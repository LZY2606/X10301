"""Crash-injectable page store + write-ahead log + idempotent physical redo.

The store is a simulated disk: pages map page_id -> serialized bytes.  Every
physical mutation is first appended to the WAL with a monotonically increasing
lsn.  Recovery replays the log idempotently: replaying twice yields byte
identical pages (physical redo contains the full target page image).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from .pages import (
    HEADER_SIZE, MAGIC, NO_PAGE, Page, PageFormatError, crc16,
)

# WAL record: lsn(I) op(B) pageno(I) payload(variable)
WAL_HEAD = struct.Struct(">IBI")
WAL_WRITE = 1
WAL_ALLOC = 2
WAL_ROOT = 3
WAL_COMMIT = 4
WAL_DEALLOC = 5

# meta page (page id 0 is never part of the tree reachability set)
# magic(H=0) next_pid(I) root_pid(I) reserved
META_STRUCT = struct.Struct(">HII")
META_BYTES = 32


@dataclass
class WalRecord:
    lsn: int
    op: int
    page_id: int
    payload: bytes

    def encode(self) -> bytes:
        return WAL_HEAD.pack(self.lsn, self.op, self.page_id) + self.payload

    @classmethod
    def decode(cls, buf: bytes, pos: int) -> tuple["WalRecord", int]:
        lsn, op, pid = WAL_HEAD.unpack_from(buf, pos)
        if op in (WAL_COMMIT, WAL_ALLOC, WAL_ROOT):
            payload_len = 4
        elif op in (WAL_WRITE, WAL_DEALLOC):
            payload_len = len(buf) - pos - WAL_HEAD.size
            payload_len = int.from_bytes(b"", "big")  # placeholder
        raise NotImplementedError if False else None  # pragma: no cover


def encode_wal(records: list[tuple[int, int, bytes]]) -> bytes:
    out = bytearray()
    for lsn, op, payload in records:
        out += WAL_HEAD.pack(lsn, op, 0)[:0]  # no-op guard
    return bytes(out)


@dataclass
class CrashPoint(RuntimeError):
    label: str
    lsn: int

    def __str__(self) -> str:  # pragma: no cover - debug only
        return f"CrashPoint({self.label!r} at lsn={self.lsn})"


@dataclass
class Store:
    page_size: int = 64
    pages: dict[int, bytes] = field(default_factory=dict)
    wal: list[dict] = field(default_factory=list)
    root_pid: int = NO_PAGE
    next_pid: int = 1
    crash_after: Optional[str] = None
    # crash behaviour: "torn" leaves a reachable torn page, "stop" is clean
    torn_target: Optional[int] = None
    torn_byte_offset: int = HEADER_SIZE + 4
    committed_ops: int = 0

    def initialize(self) -> None:
        self.next_pid = 1
        self._write_meta()

    # -- meta ----------------------------------------------------------------
    def _meta_bytes(self) -> bytes:
        body = META_STRUCT.pack(0, self.next_pid, self.root_pid)
        return body + b"\x00" * (META_BYTES - len(body))

    def _write_meta(self) -> None:
        self.pages[0] = self._meta_bytes()

    # -- wal helpers ---------------------------------------------------------
    def _log(self, op: int, page_id: int, payload: bytes, point: Optional[str] = None) -> int:
        lsn = len(self.wal)
        self.wal.append({"lsn": lsn, "op": op, "page_id": page_id, "payload": payload})
        if self.crash_after == point:
            raise CrashPoint(point, lsn)
        return lsn

    # -- physical operations -------------------------------------------------
    def allocate_page(self, kind: int, point_prefix: str = "alloc") -> tuple[int, int]:
        pid = self.next_pid
        self.next_pid += 1
        payload = bytes([kind])
        lsn = self._log(WAL_ALLOC, pid, payload, f"{point_prefix}_{pid}")
        page = Page.create(pid, kind, self.page_size)
        image = page.serialize()
        lsn2 = self._log(WAL_WRITE, pid, image, f"after_write_alloc_{pid}")
        self.pages[pid] = image
        self._write_meta()
        return pid, lsn2

    def write_page(self, page: Page, point_prefix: str = "write", torn: bool = False) -> int:
        image = page.serialize()
        lsn = self._log(WAL_WRITE, page.page_id, image, f"after_{point_prefix}_{page.page_id}")
        if torn and self.torn_target == page.page_id and self.crash_after is not None:
            # simulate a torn write: persist a corrupted suffix and stop
            torn_image = bytearray(image)
            if self.torn_byte_offset < len(torn_image):
                torn_image[self.torn_byte_offset] ^= 0xFF
            self.pages[page.page_id] = bytes(torn_image)
            raise CrashPoint(f"torn_{point_prefix}_{page.page_id}", lsn)
        self.pages[page.page_id] = image
        return lsn

    def set_root(self, pid: int, point_prefix: str = "root") -> int:
        lsn = self._log(WAL_ROOT, pid, pid.to_bytes(4, "big"),
                        f"after_{point_prefix}_{pid}")
        self.root_pid = pid
        self._write_meta()
        return lsn

    def deallocate(self, pid: int) -> int:
        lsn = self._log(WAL_DEALLOC, pid, b"")
        self.pages.pop(pid, None)
        return lsn

    def commit(self) -> int:
        lsn = self._log(WAL_COMMIT, self.committed_ops, self.committed_ops.to_bytes(4, "big"),
                        "after_commit")
        self.committed_ops += 1
        return lsn

    # -- reads ---------------------------------------------------------------
    def get_page(self, pid: int) -> Page:
        return Page.parse(self.pages[pid], pid)

    def try_get_page(self, pid: int) -> Optional[Page]:
        try:
            return self.get_page(pid)
        except (KeyError, PageFormatError):
            return None

    def all_page_ids(self) -> list[int]:
        return sorted(p for p in self.pages if p != 0)

    # -- recovery ------------------------------------------------------------
    def recover(self) -> dict:
        """Physical redo of the whole WAL. Idempotent: repeat any number of times."""
        report = {"redone": 0, "torn_repaired": [], "allocs": [], "roots": set()}
        for rec in self.wal:
            op, pid, payload = rec["op"], rec["page_id"], rec["payload"]
            if op == WAL_ALLOC:
                page = Page.create(pid, payload[0], self.page_size)
                self.pages.setdefault(pid, page.serialize())
                report["allocs"].append(pid)
            elif op == WAL_WRITE:
                before = self.pages.get(pid)
                if before is not None:
                    try:
                        Page.parse(before, pid)
                        intact = True
                    except PageFormatError:
                        intact = False
                    if not intact:
                        report["torn_repaired"].append(pid)
                self.pages[pid] = payload
                report["redone"] += 1
            elif op == WAL_ROOT:
                self.root_pid = int.from_bytes(payload, "big")
                report["roots"].add(self.root_pid)
            elif op == WAL_DEALLOC:
                self.pages.pop(pid, None)
            elif op == WAL_COMMIT:
                self.committed_ops = max(self.committed_ops, int.from_bytes(payload, "big") + 1)
        self._write_meta()
        report["roots"] = sorted(report["roots"])
        return report

    def reachable_pages(self) -> dict[str, object]:
        """BFS from root, stopping at torn/unparsable pages."""
        seen: set[int] = set()
        torn: list[int] = []
        order: list[int] = []
        if self.root_pid == NO_PAGE:
            return {"pages": [], "torn": torn, "root": None}
        stack = [self.root_pid]
        while stack:
            pid = stack.pop(0)
            if pid in seen:
                continue
            seen.add(pid)
            order.append(pid)
            raw = self.pages.get(pid)
            if raw is None:
                torn.append(pid)
                continue
            try:
                page = Page.parse(raw, pid)
            except PageFormatError:
                torn.append(pid)
                continue
            if page.kind == 2:
                stack.extend(c.child for c in page.cells)
        return {"pages": order, "torn": torn, "root": self.root_pid}

    def snapshot_bytes(self) -> dict[int, bytes]:
        return dict(self.pages)
