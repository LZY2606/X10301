"""Fixed-page-size B+tree engine with byte-layout pages, splits and WAL recovery.

The engine is deliberately deterministic so that a case bundle replays to the
exact same page ids, slot directories, log sequence numbers and layouts.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Callable, Optional

PAGE_SIZE = 256
HEADER_SIZE = 16
SLOT_SIZE = 6
INVALID = 0xFFFF
LSN_NONE = 0

# header layout (16 bytes):
# magic H, kind B, flags B, free H, next_slot H, free_end H, first H, lsn Q
MAGIC = 0x5047  # "PG"
KIND_LEAF = 1
KIND_INTERNAL = 2
FLAG_TORN = 0x01


class EngineError(Exception):
    """Hard error (e.g. record larger than a page). Import-safe diagnostics."""


@dataclass
class Entry:
    key: bytes
    value: bytes
    child: int = INVALID  # internal entry child pointer

    def payload(self, kind: int) -> int:
        if kind == KIND_INTERNAL:
            return len(self.key) + 2 + 4
        return len(self.key) + 2 + len(self.value) + 2


@dataclass
class Page:
    pid: int
    kind: int = KIND_LEAF
    flags: int = 0
    # internal: first child pointer; leaf: right sibling link
    link: int = INVALID
    entries: list[Entry] = field(default_factory=list)
    lsn: int = LSN_NONE

    def used(self) -> int:
        body = sum(e.payload(self.kind) for e in self.entries)
        return HEADER_SIZE + SLOT_SIZE * len(self.entries) + body

    def free(self) -> int:
        return PAGE_SIZE - self.used()

    def serialize(self) -> bytes:
        """Pack entries back-to-back from the page end; slots from the header."""
        n = len(self.entries)
        free_end = PAGE_SIZE
        for e in self.entries:
            free_end -= e.payload(self.kind)
        free_space = free_end - (HEADER_SIZE + SLOT_SIZE * n)
        out = bytearray(PAGE_SIZE)
        struct.pack_into(
            "<HBBHHHHQ",
            out,
            0,
            MAGIC,
            self.kind,
            self.flags,
            free_space,
            HEADER_SIZE + SLOT_SIZE * n,
            free_end,
            self.link,
            self.lsn,
        )
        # slot directory: (offset H, key_len H, tail_len H)
        off = HEADER_SIZE
        cursor = PAGE_SIZE
        for e in self.entries:
            cursor -= e.payload(self.kind)
            if self.kind == KIND_INTERNAL:
                tail_len = 4
            else:
                tail_len = len(e.value) + 2
            struct.pack_into("<HHH", out, off, cursor, len(e.key), tail_len)
            off += SLOT_SIZE
            cursor2 = cursor
            if self.kind == KIND_INTERNAL:
                out[cursor2:cursor2 + len(e.key)] = e.key
                cursor2 += len(e.key)
                struct.pack_into("<H", out, cursor2, len(e.value)); cursor2 += 2
                struct.pack_into("<H", out, cursor2, e.child); cursor2 += 4
            else:
                out[cursor2:cursor2 + len(e.key)] = e.key
                cursor2 += len(e.key)
                struct.pack_into("<H", out, cursor2, len(e.value)); cursor2 += 2
                out[cursor2:cursor2 + len(e.value)] = e.value
                cursor2 += len(e.value)
                struct.pack_into("<H", out, cursor2, e.child); cursor2 += 2
        return bytes(out)

    def slots(self) -> list[dict]:
        off = HEADER_SIZE
        res = []
        for idx, e in enumerate(self.entries):
            res.append(
                {
                    "slot": idx,
                    "dir_offset": off,
                    "record_offset": PAGE_SIZE - sum(
                        x.payload(self.kind) for x in self.entries[idx:]
                    ),
                    "key_len": len(e.key),
                    "tail_len": 4 if self.kind == KIND_INTERNAL else len(e.value) + 2,
                }
            )
            off += SLOT_SIZE
        return res

    @staticmethod
    def parse(data: bytes, pid: int) -> "Page":
        if len(data) != PAGE_SIZE:
            raise EngineError(f"page {pid}: expected {PAGE_SIZE} bytes, got {len(data)}")
        magic, kind, flags, _free, slot_end, _free_end, link, lsn = (
            struct.unpack_from("<HBBHHHHQ", data, 0)
        )
        if magic != MAGIC or kind not in (KIND_LEAF, KIND_INTERNAL):
            raise EngineError(f"page {pid}: torn or corrupt header")
        n = (slot_end - HEADER_SIZE) // SLOT_SIZE
        entries: list[Entry] = []
        for i in range(n):
            roff, klen, tail = struct.unpack_from(
                "<HHH", data, HEADER_SIZE + i * SLOT_SIZE
            )
            if roff < slot_end or roff + klen + 2 + tail > PAGE_SIZE:
                raise EngineError(f"page {pid}: slot {i} points outside free space")
            key = bytes(data[roff:roff + klen])
            p = roff + klen
            (stored_vlen,) = struct.unpack_from("<H", data, p); p += 2
            if kind == KIND_INTERNAL:
                if tail != 4 or stored_vlen != 0:
                    raise EngineError(f"page {pid}: slot {i} bad internal tail")
                (child,) = struct.unpack_from("<H", data, p)
                entries.append(Entry(key=key, value=b"", child=child))
            else:
                if tail != stored_vlen + 2:
                    raise EngineError(f"page {pid}: slot {i} length mismatch")
                value = bytes(data[p:p + stored_vlen])
                p += stored_vlen
                (child,) = struct.unpack_from("<H", data, p)
                if child != INVALID:
                    raise EngineError(f"page {pid}: leaf slot carries child pointer")
                entries.append(Entry(key=key, value=value, child=INVALID))
        return Page(pid, kind, flags, link, entries, lsn)


def _idx_leaf(p: Page, key: bytes) -> int:
    lo, hi = 0, len(p.entries)
    while lo < hi:
        mid = (lo + hi) // 2
        if p.entries[mid].key < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _find_leaf(state: "State", key: bytes) -> tuple[list[int], Page]:
    pid = state.root
    path = [pid]
    page = state.pages[pid]
    while page.kind == KIND_INTERNAL:
        idx = _idx_leaf(page, key)
        if idx == 0:
            nxt = page.link
        else:
            nxt = page.entries[idx - 1].child
        pid = nxt
        path.append(pid)
        page = state.pages[pid]
    return path, page


@dataclass
class WalRecord:
    lsn: int
    op: str  # alloc | page_write | dealloc | root
    pid: int = INVALID
    payload: bytes = b""


@dataclass
class State:
    root: int = 0
    pages: dict[int, Page] = field(default_factory=dict)
    alloc_next: int = 0
    next_lsn: int = 1
    wal: list[WalRecord] = field(default_factory=list)
    dup_policy: str = "reject"
    events: list[dict] = field(default_factory=list)
    split_resolver: Optional[Callable[[dict], Optional[int]]] = None

    def new_lsn(self) -> int:
        lsn = self.next_lsn
        self.next_lsn += 1
        return lsn

    def alloc_page(self, kind: int) -> Page:
        pid = self.alloc_next
        self.alloc_next += 1
        p = Page(pid=pid, kind=kind, first_child=INVALID)
        self.pages[pid] = p
        return p

    def log(self, kind: str, **kw) -> dict:
        ev = {"type": kind, **kw}
        self.events.append(ev)
        return ev

    # ------------------------------------------------------------------ init
    def initialize(self) -> None:
        root = self.alloc_page(KIND_LEAF)
        self.root = root.pid
        root.lsn = self.new_lsn()
        records = [
            WalRecord(self.new_lsn(), "alloc", root.pid, root.serialize()),
            WalRecord(self.new_lsn(), "page_write", root.pid, root.serialize()),
            WalRecord(self.new_lsn(), "root", self.root),
        ]
        self.wal.extend(records)
        self.log("initialize", root_pid=root.pid)

    # --------------------------------------------------------------- inserts
    def _split_candidates(self, page: Page) -> list[dict]:
        """All byte-feasible split points. Decided by bytes, not key count."""
        import numpy as np

        n = len(page.entries)
        payloads = np.array([e.payload(page.kind) for e in page.entries], dtype=np.int64)
        body_total = int(payloads.sum())
        candidates: list[dict] = []
        for i in range(1, n):
            left_body = int(payloads[:i].sum())
            right_body = body_total - left_body
            left_used = HEADER_SIZE + SLOT_SIZE * i + left_body
            right_used = HEADER_SIZE + SLOT_SIZE * (n - i) + right_body
            if left_used <= PAGE_SIZE and right_used <= PAGE_SIZE:
                candidates.append(
                    {
                        "split_at": i,
                        "left_used_bytes": left_used,
                        "right_used_bytes": right_used,
                        "imbalance_bytes": abs(left_used - right_used),
                        "left_key_count": i,
                        "right_key_count": n - i,
                    }
                )
        return candidates

    def _choose_split(self, page: Page, candidates: list[dict]) -> int:
        split_id = f"split:pid{page.pid}:keys{len(page.entries)}"
        chosen = None
        if self.split_resolver is not None:
            chosen = self.split_resolver({"split_id": split_id, "candidates": candidates})
        if chosen is None:
            chosen = min(
                candidates,
                key=lambda c: (c["imbalance_bytes"], c["split_at"]),
            )["split_at"]
        self.log(
            "split_decision",
            split_id=split_id,
            pid=page.pid,
            candidates=candidates,
            chosen_split_at=chosen,
        )
        return chosen

    def _new_child_page(self, kind: int) -> Page:
        p = self.alloc_page(kind)
        self.log("page_allocated", pid=p.pid, kind=kind)
        return p

    def _insert_leaf(self, leaf: Page, key: bytes, value: bytes) -> Optional[dict]:
        idx = _idx_leaf(leaf, key)
        if idx < len(leaf.entries) and leaf.entries[idx].key == key:
            if self.dup_policy == "reject":
                self.log(
                    "duplicate_key",
                    pid=leaf.pid,
                    key_hex=key.hex(),
                    policy="reject",
                    accepted=False,
                )
                return None
            old = leaf.entries[idx]
            leaf.entries[idx] = Entry(key, value, INVALID)
            self.log(
                "duplicate_key",
                pid=leaf.pid,
                key_hex=key.hex(),
                policy="overwrite",
                accepted=True,
                old_value_hex=old.value.hex(),
            )
            return {"kind": "dirty", "pages": [leaf.pid], "allocs": [], "deallocs": []}
        leaf.entries.insert(idx, Entry(key, value, INVALID))
        mut = {"kind": "dirty", "pages": [leaf.pid], "allocs": [], "deallocs": []}
        if leaf.used() > PAGE_SIZE:
            mut = self._split_leaf(leaf)
        return mut

    def _split_leaf(self, leaf: Page) -> dict:
        candidates = self._split_candidates(leaf)
        if not candidates:
            raise EngineError(f"page {leaf.pid}: no byte-feasible split point")
        i = self._choose_split(leaf, candidates)
        right = self._new_child_page(KIND_LEAF)
        right.entries = leaf.entries[i:]
        leaf.entries = leaf.entries[:i]
        right.link = leaf.link
        leaf.link = right.pid
        sep = right.entries[0].key
        self.log(
            "page_split",
            pid=leaf.pid,
            new_pid=right.pid,
            level="leaf",
            separator_hex=sep.hex(),
            separator_rule="copy",
            split_at=i,
        )
        return {
            "kind": "split",
            "left": leaf.pid,
            "right": right.pid,
            "separator": sep,
            "level": "leaf",
            "pages": [leaf.pid, right.pid],
            "allocs": [right.pid],
            "deallocs": [],
        }

    def _insert_into_parent(self, left: int, right: int, sep: bytes, level: str) -> dict:
        if left == self.root:
            return self._new_root(left, right, sep, level)
        parent_pid = self._parent_of(left)
        parent = self.pages[parent_pid]
        idx = _idx_leaf(parent, sep)
        parent.entries.insert(idx, Entry(sep, b"", right))
        self.log(
            "separator_promoted",
            pid=parent.pid,
            child_left=left,
            child_right=right,
            separator_hex=sep.hex(),
            separator_rule="copy" if level == "leaf" else "move",
        )
        mut = {
            "kind": "dirty",
            "pages": [parent.pid],
            "allocs": [],
            "deallocs": [],
        }
        if parent.used() > PAGE_SIZE:
            mut = self._split_internal(parent)
        return mut

    def _parent_of(self, child: int) -> int:
        for pid, p in self.pages.items():
            if p.kind != KIND_INTERNAL:
                continue
            if p.link == child or any(e.child == child for e in p.entries):
                return pid
        raise EngineError(f"orphan child page {child}")

    def _new_root(self, left: int, right: int, sep: bytes, level: str) -> dict:
        root = self._new_child_page(KIND_INTERNAL)
        root.link = left
        root.entries = [Entry(sep, b"", right)]
        old_root = self.root
        self.root = root.pid
        self.log(
            "root_changed",
            old_root=old_root,
            new_root=root.pid,
            cause="root_split",
            separator_hex=sep.hex(),
            separator_rule="copy" if level == "leaf" else "move",
        )
        return {
            "kind": "new_root",
            "left": left,
            "right": right,
            "separator": sep,
            "level": level,
            "pages": [root.pid],
            "allocs": [root.pid],
            "deallocs": [],
        }

    def _split_internal(self, page: Page) -> dict:
        candidates = self._split_candidates(page)
        if not candidates:
            raise EngineError(f"internal page {id(page)}: no feasible split point")
        i = self._choose_split(page, candidates)
        right = self._new_child_page(KIND_INTERNAL)
        up = page.entries[i - 1].key
        right.link = page.entries[i - 1].child
        right.entries = page.entries[i:]
        page.entries = page.entries[: i - 1]
        self.log(
            "page_split",
            pid=page.pid,
            new_pid=right.pid,
            level="internal",
            separator_hex=up.hex(),
            separator_rule="move",
            split_at=i,
        )
        if page.pid == self.root:
            return self._new_root(page.pid, right.pid, up, "internal")
        return self._insert_into_parent(page.pid, right.pid, up, "internal")
