"""Paged store: meta frame, WAL-first durability, buffer pool, crash injection."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from . import page as P
from .record import (LogRecord, BEGIN, COMMIT, ALLOC_PAGE, UPDATE_PAGE,
                     FREE_PAGE, CLR, encode_all, decode_all)

META_PID = 0
META_FORMAT = ">iiH"
META_LEN = struct.calcsize(META_FORMAT)  # 10
META_CHECKSUM_AT = 8


def meta_checksum(buf: bytes) -> int:
    data = bytearray(buf[:META_LEN])
    data[META_CHECKSUM_AT:META_CHECKSUM_AT + 2] = b"\x00\x00"
    total = 0
    for i in range(0, META_LEN - 2, 2):
        total += (data[i] << 8) | data[i + 1]
    return (~total) & 0xFFFF


def encode_meta(page_size: int, root_pid: int, next_pid: int) -> bytes:
    body = struct.pack(META_FORMAT, page_size, root_pid, next_pid)
    body = body[:8] + struct.pack(">H", meta_checksum(body))
    return body + b"\x00" * (page_size - META_LEN)


def decode_meta(frame: bytes) -> dict:
    page_size, root_pid, next_pid = struct.unpack_from(META_FORMAT, frame, 0)
    stored = struct.unpack_from(">H", frame, META_CHECKSUM_AT)[0]
    return {"page_size": page_size, "root_pid": root_pid, "next_pid": next_pid,
            "valid": stored == meta_checksum(frame)}


@dataclass
class Pager:
    page_size: int = 256
    disk: dict[int, bytes] = field(default_factory=dict)
    wal: bytearray = field(default_factory=lambda: bytearray())
    next_txn_id: int = 1
    _lsn: int = 0

    def fresh(self) -> None:
        self.disk = {META_PID: encode_meta(self.page_size, root_pid=-1, next_pid=1)}

    @property
    def records(self) -> list[LogRecord]:
        return decode_all(bytes(self.wal))

    def next_lsn(self) -> int:
        return self._lsn + 1

    def append_log(self, rec: LogRecord) -> LogRecord:
        rec.lsn = self._lsn + 1
        self._lsn = rec.lsn
        self.wal += rec.encode()
        return rec

    def read_disk(self, pid: int) -> bytes | None:
        return self.disk.get(pid)

    def write_disk(self, pid: int, frame: bytes, torn_bytes: int | None = None) -> None:
        if torn_bytes is None:
            self.disk[pid] = frame
            return
        # interrupted page write: a prefix lands and the rest reads as zeros
        partial = frame[:torn_bytes] + b"\x00" * (self.page_size - torn_bytes)
        self.disk[pid] = partial

    def remove_disk(self, pid: int) -> None:
        self.disk.pop(pid, None)

    def meta(self) -> dict:
        return decode_meta(self.disk[META_PID])

    def write_meta(self) -> None:
        m = self.meta()
        self.disk[META_PID] = encode_meta(self.page_size, m["root_pid"],
                                         m["next_pid"])

    def set_root(self, root_pid: int) -> None:
        m = self.meta()
        self.disk[META_PID] = encode_meta(self.page_size, root_pid, m["next_pid"])


@dataclass
class Txn:
    txn_id: int
    pager: Pager
    dirty: dict[int, bytes] = field(default_factory=dict)  # final desired frames
    last_lsn: int = -1
    committed: bool = False

    def begin(self) -> None:
        self.pager.append_log(LogRecord(lsn=0, kind=BEGIN, txn=self.txn_id,
                                        prev_lsn=self.last_lsn))

    def read(self, pid: int) -> bytes:
        if pid in self.dirty:
            return self.dirty[pid]
        frame = self.pager.read_disk(pid)
        if frame is None:
            raise KeyError(f"page {pid} not allocated")
        return frame

    def alloc(self, kind: int) -> tuple[int, bytes]:
        meta = self.pager.meta()
        pid = meta["next_pid"]
        lsn = self.pager.next_lsn()
        new_page = P.frame(self.pager.page_size, kind, pid, page_lsn=lsn)
        rec = LogRecord(lsn=0, kind=ALLOC_PAGE, txn=self.txn_id, pid=pid,
                        prev_lsn=self.last_lsn, page_kind=kind, after=new_page)
        self.pager.append_log(rec)
        self.last_lsn = lsn
        self.dirty[pid] = new_page
        self._set_next_pid(pid + 1)
        return pid, new_page

    def _set_next_pid(self, next_pid: int) -> None:
        m = self.pager.meta()
        lsn = self.pager.next_lsn()
        new_meta = encode_meta(self.pager.page_size, m["root_pid"], next_pid)
        self.pager.append_log(LogRecord(
            lsn=0, kind=UPDATE_PAGE, txn=self.txn_id, pid=META_PID,
            prev_lsn=self.last_lsn, before=self.pager.disk[META_PID],
            after=new_meta))
        self.last_lsn = lsn
        self.dirty[META_PID] = new_meta

    def modify(self, pid: int, build) -> bytes:
        """``build(lsn)`` returns the new frame; logs before/after image."""
        before = self.read(pid)
        lsn = self.pager.next_lsn()
        after = build(lsn)
        if P.header(after).page_lsn != lsn:
            raise ValueError("rebuilt frame must carry new page LSN")
        self.pager.append_log(LogRecord(
            lsn=0, kind=UPDATE_PAGE, txn=self.txn_id, pid=pid,
            prev_lsn=self.last_lsn, page_lsn_from=P.header(before).page_lsn,
            before=before, after=after))
        self.last_lsn = lsn
        self.dirty[pid] = after
        return after

    def free(self, pid: int) -> None:
        before = self.read(pid)
        lsn = self.pager.next_lsn()
        self.pager.append_log(LogRecord(
            lsn=0, kind=FREE_PAGE, txn=self.txn_id, pid=pid,
            prev_lsn=self.last_lsn, before=before))
        self.last_lsn = lsn
        self.dirty.pop(pid, None)

    def set_root(self, root_pid: int) -> None:
        m = self.pager.meta()
        lsn = self.pager.next_lsn()
        new_meta = encode_meta(self.pager.page_size, root_pid, m["next_pid"])
        self.pager.append_log(LogRecord(
            lsn=0, kind=UPDATE_PAGE, txn=self.txn_id, pid=META_PID,
            prev_lsn=self.last_lsn, before=self.pager.disk[META_PID],
            after=new_meta))
        self.last_lsn = lsn
        self.dirty[META_PID] = new_meta

    def commit(self, flush_order: str = "child_first") -> None:
        self.pager.append_log(LogRecord(lsn=0, kind=COMMIT, txn=self.txn_id,
                                        prev_lsn=self.last_lsn))
        self.committed = True
        self._flush(flush_order, torn_pid=None, interrupt=False)

    def _tree_pids(self) -> list[int]:
        return [pid for pid in self.dirty if pid != META_PID]

    def _flush(self, order: str, torn_pid: int | None,
               interrupt: bool) -> list[dict]:
        writes: list[dict] = []
        pids = self._tree_pids()
        if order == "child_first":
            pids = sorted(pids, key=lambda p: (P.header(self.dirty[p]).kind, p))
        elif order == "parent_first":
            pids = sorted(pids, key=lambda p: (-P.header(self.dirty[p]).kind, p))
        elif order == "none":
            pids = []
        for pid in pids:
            frame = self.dirty[pid]
            if pid == torn_pid:
                self.pager.write_disk(pid, frame, torn_bytes=1)
                writes.append({"pid": pid, "torn": True})
                if interrupt:
                    return writes
            self.pager.write_disk(pid, frame)
            writes.append({"pid": pid, "torn": False})
        if META_PID in self.dirty:
            self.pager.disk[META_PID] = self.dirty[META_PID]
            writes.append({"pid": META_PID, "torn": False})
        return writes
