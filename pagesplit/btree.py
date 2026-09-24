"""B-tree page model: on-page byte layout, WAL, splits, crash recovery.

Page layout (leaf and internal share a 28-byte header)::

    0  magic u16        LEAF=0xB7EE INTERNAL=0xB7EF META=0xB7ED
    2  page_id u32
    6  type-specific    leaf: next_sibling u32 / internal: leftmost u32
    10 flags u16        bit0: root
    12 nslots u16
    14 free_start u16   first free byte (cells grow down from here)
    16 free_end u16     last free byte (slot dir grows up from here)
    18 lsn u32          log sequence number of last modifying record
    22 cell_bytes u16   bytes occupied by cells
    24 checksum u32     FNV-1a over bytes [0:24]
    28 ... slot directory: u16 offsets, slot i at 28+2*i
    cells packed from page_size downward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MAGIC_META = 0xB7ED
MAGIC_LEAF = 0xB7EE
MAGIC_INTERNAL = 0xB7EF
HEADER_SIZE = 28
SLOT_SIZE = 2

T_LEAF = "leaf"
T_INTERNAL = "internal"
T_META = "meta"

# WAL record types
W_ALLOC = "alloc"
W_UPDATE = "update"
W_COMMIT = "commit"
W_ABORT = "abort"
W_CHECKPOINT = "checkpoint"


def fnv1a(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _u16(b: bytes, o: int) -> int:
    return int.from_bytes(b[o:o + 2], "big")


def _u32(b: bytes, o: int) -> int:
    return int.from_bytes(b[o:o + 4], "big")


def _lp(n: int) -> bytes:
    if n <= 255:
        return bytes((n,))
    return b"\x00" + n.to_bytes(2, "big")


def _lp_get(b: bytes, o: int) -> tuple[int, int]:
    if b[o] != 0:
        return b[o], 1
    return int.from_bytes(b[o + 1:o + 3], "big"), 3


class TornPageError(Exception):
    """Checksum mismatch: a half-written page reached on disk."""


@dataclass
class Cell:
    key: bytes
    value: bytes = b""
    child: int | None = None  # internal: right-child page id

    def encode(self) -> bytes:
        if self.child is None:
            return _lp(len(self.key)) + self.key + _lp(len(self.value)) + self.value
        return _lp(len(self.key)) + self.key + self.child.to_bytes(4, "big")

    @property
    def cell_size(self) -> int:
        return len(self.encode())

    def to_dict(self) -> dict:
        d = {"key_hex": self.key.hex(), "key_utf8": _try_utf8(self.key),
             "nbytes": self.cell_size}
        if self.child is None:
            d["value_hex"] = self.value.hex()
            d["value_utf8"] = _try_utf8(self.value)
        else:
            d["child"] = self.child
        return d


def _try_utf8(b: bytes) -> str | None:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def decode_cell(data: bytes, page_type: str) -> Cell:
    klen, n = _lp_get(data, 0)
    key = data[n:n + klen]
    p = n + klen
    if page_type == T_INTERNAL:
        child = _u32(data, p)
        return Cell(key=key, child=child)
    vlen, n2 = _lp_get(data, p)
    value = data[p + n2:p + n2 + vlen]
    return Cell(key=key, value=value)


class Page:
    def __init__(self, page_id: int, page_type: str, page_size: int,
                 cells: list[Cell] | None = None, link: int = 0,
                 is_root: bool = False, lsn: int = 0):
        self.page_id = page_id
        self.page_type = page_type
        self.page_size = page_size
        self.cells = cells or []
        self.link = link  # leaf: next sibling; internal: leftmost child
        self.is_root = is_root
        self.lsn = lsn

    # ---- identity / sizing -------------------------------------------------
    @property
    def magic(self) -> int:
        return {T_META: MAGIC_META, T_LEAF: MAGIC_LEAF,
                T_INTERNAL: MAGIC_INTERNAL}[self.page_type]

    @property
    def slot_bytes(self) -> int:
        return SLOT_SIZE * len(self.cells)

    @property
    def cell_payload_bytes(self) -> int:
        return sum(c.cell_size for c in self.cells)

    @property
    def used_bytes(self) -> int:
        return HEADER_SIZE + self.slot_bytes + self.cell_payload_bytes

    @property
    def free_space(self) -> int:
        return self.page_size - self.used_bytes

    def slot_directory(self) -> list[int]:
        """Offsets of each cell, slot order == insertion order (unsorted)."""
        offsets = []
        end = self.page_size
        for c in self.cells:
            end -= c.cell_size
            offsets.append(end)
        return offsets

    # ---- serialisation -----------------------------------------------------
    def encode(self) -> bytes:
        buf = bytearray(self.page_size)
        buf[0:2] = self.magic.to_bytes(2, "big")
        buf[2:6] = self.page_id.to_bytes(4, "big")
        buf[6:10] = (self.link & 0xFFFFFFFF).to_bytes(4, "big")
        flags = 1 if self.is_root else 0
        buf[10:12] = flags.to_bytes(2, "big")
        n = len(self.cells)
        buf[12:14] = n.to_bytes(2, "big")
        blob = b"".join(c.encode() for c in self.cells)
        free_start = self.page_size - len(blob)
        free_end = HEADER_SIZE + 2 * n
        buf[14:16] = free_start.to_bytes(2, "big")
        buf[16:18] = free_end.to_bytes(2, "big")
        buf[18:22] = (self.lsn & 0xFFFFFFFF).to_bytes(4, "big")
        buf[22:24] = len(blob).to_bytes(2, "big")
        buf[24:28] = fnv1a(bytes(buf[0:24])).to_bytes(4, "big")
        # slot directory: insertion order, points at packed cells
        end = self.page_size
        for i, c in enumerate(self.cells):
            end -= c.cell_size
            buf[HEADER_SIZE + 2 * i:HEADER_SIZE + 2 * i + 2] = end.to_bytes(2, "big")
        buf[free_start:self.page_size] = blob
        return bytes(buf)

    @staticmethod
    def decode(data: bytes, page_size: int, verify: bool = True) -> "Page":
        if len(data) != page_size:
            raise TornPageError(f"short page: {len(data)} != {page_size}")
        magic = _u16(data, 0)
        stored = _u32(data, 24)
        if verify and fnv1a(data[0:24]) != stored:
            raise TornPageError("checksum mismatch")
        page_id = _u32(data, 2)
        page_type = {MAGIC_META: T_META, MAGIC_LEAF: T_LEAF,
                     MAGIC_INTERNAL: T_INTERNAL}.get(magic)
        if page_type is None:
            raise TornPageError(f"bad magic {magic:#x}")
        link = _u32(data, 6)
        is_root = bool(_u16(data, 10) & 1)
        n = _u16(data, 12)
        lsn = _u32(data, 18)
        p = Page(page_id, page_type, page_size, link=link, is_root=is_root, lsn=lsn)
        for i in range(n):
            off = _u16(data, HEADER_SIZE + 2 * i)
            p.cells.append(decode_cell(data[off:], page_type))
        return p

    # ---- layout view -------------------------------------------------------
    def regions(self) -> list[dict]:
        blob = self.cell_payload_bytes
        n = len(self.cells)
        return [
            {"name": "header", "start": 0, "end": HEADER_SIZE},
            {"name": "slot_directory", "start": HEADER_SIZE,
             "end": HEADER_SIZE + 2 * n, "nslots": n},
            {"name": "free_space", "start": HEADER_SIZE + 2 * n,
             "end": self.page_size - blob, "bytes": self.free_space},
            {"name": "cells", "start": self.page_size - blob,
             "end": self.page_size},
        ]


@dataclass
class WalRecord:
    lsn: int
    txn: int
    kind: str
    page_id: int = 0
    before: bytes | None = None      # full before-image (update/alloc seed)
    after: bytes | None = None       # after-image for redo
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"lsn": self.lsn, "txn": self.txn, "kind": self.kind,
                "page_id": self.page_id,
                "before_hex": self.before.hex() if self.before is not None else None,
                "after_hex": self.after.hex() if self.after is not None else None,
                "payload": self.payload}


@dataclass
class Meta:
    root: int = 1
    max_page_id: int = 1
    lsn: int = 0


class Engine:
    """In-memory pages + WAL + simulated paged file with crash injection."""

    def __init__(self, page_size: int = 128):
        self.page_size = page_size
        self.pages: dict[int, Page] = {}
        self.disk: dict[int, bytes] = {}
        self.wal: list[WalRecord] = []
        self.meta = Meta()
        self.lsn = 0
        self.txn = 0
        self.checkpoint_lsn = 0
        # crash injection configuration (set inside a transaction)
        self.crash_after_disk_pages: set[int] | None = None
        self.crash_tear_page: int | None = None
        self.crash_after_wal: bool = False
        self.crashed = False
        self._new_pages: set[int] = set()
        self._dirty: set[int] = set()
        self._pending: dict[int, tuple[bytes, bool]] = {}
        self._init_db()

    def _init_db(self) -> None:
        root = Page(1, T_LEAF, self.page_size, is_root=True)
        self.pages[1] = root
        self._new_pages.add(1)
        self._append(0, W_ALLOC, 1, after=b"", payload={"page_type": T_LEAF})
        root.lsn = self.lsn
        self._flush_page(1)
        meta = Page(0, T_META, self.page_size)
        meta.cells = [Cell(b"root", root.page_id.to_bytes(4, "big")),
                      Cell(b"maxpid", self.meta.max_page_id.to_bytes(4, "big"))]
        self.pages[0] = meta
        self._new_pages.add(0)
        self._append(0, W_ALLOC, 0, after=b"", payload={"page_type": T_META})
        meta.lsn = self.lsn
        self._flush_page(0)
        self._append(0, W_CHECKPOINT, payload={"root": 1, "max_page_id": 1})
        self.checkpoint_lsn = self.lsn

    # ---- low level ---------------------------------------------------------
    def _next_lsn(self) -> int:
        self.lsn += 1
        return self.lsn

    def _append(self, txn: int, kind: str, page_id: int = 0,
                before: bytes | None = None, after: bytes | None = None,
                payload: dict | None = None) -> WalRecord:
        rec = WalRecord(self._next_lsn(), txn, kind, page_id, before, after,
                        payload or {})
        self.wal.append(rec)
        if self.crash_after_wal:
            self.crashed = True
            raise CrashInjection("after_wal_flush", rec)
        return rec

    def _alloc_log_seed(self, pid: int, page: Page) -> None:
        pass  # initial seed is logged through _new_page in txns

    def _new_page(self, txn: int, page_type: str, link: int = 0,
                  is_root: bool = False) -> Page:
        pid = self.meta.max_page_id + 1
        self.meta.max_page_id = pid
        page = Page(pid, page_type, self.page_size, link=link, is_root=is_root)
        self.pages[pid] = page
        self._new_pages.add(pid)
        self._dirty.add(pid)
        return page

    def _flush_page(self, pid: int) -> None:
        page = self.pages[pid]
        if self.crash_tear_page == pid:
            # simulated torn write: only the first half of the page reaches
            # durable storage, zero-filling the remainder => bad checksum.
            full = page.encode()
            half = self.page_size // 2
            torn = full[:half] + b"\x00" * (self.page_size - half)
            self.disk[pid] = torn
            self.crashed = True
            raise CrashInjection("torn_page_write", pid)
        self.disk[pid] = page.encode()
        if self.crash_after_disk_pages is not None:
            self.crash_after_disk_pages.discard(pid)
            if not self.crash_after_disk_pages:
                self.crashed = True
                raise CrashInjection("after_page_write", pid)

    def _log_update(self, txn: int, pid: int, before: bytes) -> None:
        after = self.pages[pid].encode()
        rec = self._append(txn, W_UPDATE, pid, before=before, after=after)
        self.pages[pid].lsn = rec.lsn

    # ---- transactions ------------------------------------------------------
    def begin(self) -> int:
        self.txn += 1
        self._new_pages = set()
        self._dirty = set()
        return self.txn

    def commit(self, txn: int, crash: dict | None = None) -> WalRecord:
        """Write all dirty pages (oldest->youngest within this txn's order is
        preserved by log order), then COMMIT.

        crash spec: {"after_disk_pages": [ids...]} raises after those pages
        reached disk; {"after_wal": true} raises when COMMIT hits the WAL.
        """
        if crash:
            ids = crash.get("after_disk_pages")
            self.crash_after_disk_pages = set(ids) if ids else None
            self.crash_tear_page = crash.get("tear_page")
            self.crash_after_wal = bool(crash.get("after_wal"))
        dirty = sorted(self._dirty)
        for pid in dirty:
            after = self.pages[pid].encode()
            if pid in self._new_pages:
                rec = self._append(txn, W_ALLOC, pid, after=after,
                                   payload={"final_image": True})
            else:
                rec = self._append(txn, W_UPDATE, pid,
                                   before=self.disk.get(pid, b""), after=after)
            self.pages[pid].lsn = rec.lsn
            self._flush_page(pid)
        rec = self._append(txn, W_COMMIT, payload={})
        self.crash_after_disk_pages = None
        self.crash_tear_page = None
        self.crash_after_wal = False
        self._write_meta()
        if self.lsn - self.checkpoint_lsn >= 6:
            self.checkpoint()
        self._new_pages = set()
        self._dirty = set()
        return rec

    def abort(self, txn: int) -> None:
        for rec in self.wal:
            pass
        # roll back in-memory changes using before images of this txn
        for rec in reversed(self.wal):
            if rec.txn != txn:
                continue
            if rec.kind == W_UPDATE and rec.before is not None:
                self.pages[rec.page_id] = Page.decode(rec.before, self.page_size)
            elif rec.kind == W_ALLOC:
                self.pages.pop(rec.page_id, None)
                self.meta.max_page_id = max(
                    [0] + [p for p in self.pages if p != 0])
        self._append(txn, W_ABORT)
        self._new_pages = set()

    def _page_disk_lsn(self, pid: int) -> int:
        raw = self.disk.get(pid)
        if raw is None:
            return -1
        return _u32(raw, 18)

    def _write_meta(self) -> None:
        meta = self.pages[0]
        meta.cells = [Cell(b"root", self.meta.root.to_bytes(4, "big")),
                      Cell(b"maxpid", self.meta.max_page_id.to_bytes(4, "big"))]
        meta.lsn = self.lsn
        self._flush_page(0)

    def checkpoint(self) -> None:
        for pid, page in self.pages.items():
            self.disk[pid] = page.encode()
        self.checkpoint_lsn = self.lsn
        self._append(0, W_CHECKPOINT, payload={
            "root": self.meta.root, "max_page_id": self.meta.max_page_id})

    # ---- tree operations ---------------------------------------------------
    def _find_leaf(self, key: bytes) -> Page:
        pid = self.meta.root
        while True:
            page = self.pages[pid]
            if page.page_type == T_LEAF:
                return page
            nxt = page.link
            for cell in page.cells:
                if key < cell.key:
                    nxt = cell.child
                    break
            pid = nxt

    def insert(self, txn: int, key: bytes, value: bytes,
               dup: str = "last_wins", separator_policy: str = "copy") -> dict:
        """Insert into the tree. Returns a trace with split events/candidates."""
        trace = {"op": "insert", "key_hex": key.hex(), "splits": [],
                 "root_split": False}
        right_id, sep, promoted = self._insert_recursive(
            txn, self.meta.root, key, value, dup, separator_policy, trace, 0)
        if sep is not None:
            left = self.pages[self.meta.root]
            left.is_root = False
            root = self._new_page(txn, T_INTERNAL, link=left.page_id,
                                  is_root=True)
            root.cells = [Cell(sep, child=right_id)]
            self.meta.root = root.page_id
            trace["root_split"] = True
            trace["splits"].append({
                "depth": 0, "page_id": left.page_id, "new_page_id": root.page_id,
                "kind": "root", "separator_key_hex": sep.hex(),
                "separator_policy": ("promote" if promoted else "copy")})
        return trace

    def _insert_recursive(self, txn: int, pid: int, key: bytes, value: bytes,
                          dup: str, sep_policy: str, trace: dict, depth: int):
        """Returns (right_or_self_page_id, separator|None, promoted)."""
        page = self.pages[pid]
        if page.page_type == T_LEAF:
            self._leaf_put(page, key, value, dup)
            if self._needs_split(page):
                rid, sep, prom = self._split_leaf(txn, page, sep_policy, trace, depth)
                return rid, sep, prom
            return pid, None, False
        # children: [link] + [cell.child ...]; pick child-index ci
        ci = 0
        for cell in page.cells:
            if key < cell.key:
                break
            ci += 1
        child = page.link if ci == 0 else page.cells[ci - 1].child
        rid, sep, promoted = self._insert_recursive(
            txn, child, key, value, dup, sep_policy, trace, depth + 1)
        if sep is not None:
            # old child keeps left subtree (same id); new cell names the right
            page.cells.insert(ci, Cell(sep, child=rid))
            if self._needs_split(page):
                return self._split_internal(txn, page, sep_policy, trace, depth)
        return pid, None, False

    # ---- leaf / internal helpers ------------------------------------------
    def _needs_split(self, page: Page) -> bool:
        return page.free_space < 0 or self._overflow_cell(page)

    def _overflow_cell(self, page: Page) -> bool:
        # slot directory and cells overlap when the cell region collides
        return HEADER_SIZE + 2 * (len(page.cells)) > self.page_size - page.cell_payload_bytes

    def _touch(self, page: Page) -> None:
        self._dirty.add(page.page_id)

    def _leaf_put(self, page: Page, key: bytes, value: bytes, dup: str) -> None:
        existing = [i for i, c in enumerate(page.cells) if c.key == key]
        if existing:
            i = existing[-1]
            if dup == "last_wins":
                page.cells[i] = Cell(key, value)
            elif dup == "append_version":
                page.cells.append(Cell(key, value))
            # reject: keep first value untouched
        else:
            page.cells.append(Cell(key, value))
        # slot directory follows deterministic key order on a leaf
        page.cells.sort(key=lambda c: c.key)
        self._touch(page)

    def _split_index(self, page: Page) -> int:
        """Byte-space split index.

        Invariant: decided by cumulative *bytes*, never key count.  Candidates
        are every prefix position that yields two pages each fitting the page;
        the chosen one is the legal candidate closest to a balanced payload.
        """
        sizes = [c.cell_size for c in page.cells]
        total = sum(sizes)
        n = len(page.cells)
        legal = []
        for i in range(1, n):
            left = sum(sizes[:i])
            right = total - left
            if (HEADER_SIZE + 2 * i + left <= self.page_size and
                    HEADER_SIZE + 2 * (n - i) + right <= self.page_size):
                legal.append(i)
        if not legal:  # pathological: a single cell overflows
            legal = [max(1, n // 2)]
        best = min(legal, key=lambda i: abs(sum(sizes[:i]) - (total - sum(sizes[:i]))))
        page._split_candidates = {
            "by": "bytes", "total_cell_bytes": total,
            "prefix_bytes": [sum(sizes[:i]) for i in range(n + 1)],
            "legal_indexes": legal, "chosen_index": best}
        return best

    def _split_leaf(self, txn: int, page: Page, sep_policy: str,
                    trace: dict, depth: int):
        idx = self._split_index(page)
        left_cells, right_cells = page.cells[:idx], page.cells[idx:]
        sep = right_cells[0].key
        right = self._new_page(txn, T_LEAF, link=page.link)
        if sep_policy == "promote":
            # lift separator out of the leaf: it lives only in the parent
            promoted = [c for c in right_cells if c.key != sep]
            right.cells = promoted
            promoted_flag = True
        else:
            right.cells = list(right_cells)  # copy: sep stays in right leaf
            promoted_flag = False
        page.cells = list(left_cells)
        page.link = right.page_id
        self._touch(page)
        trace["splits"].append({
            "depth": depth, "page_id": page.page_id,
            "new_page_id": right.page_id, "kind": "leaf",
            "separator_key_hex": sep.hex(), "separator_policy": sep_policy,
            "separator_in_right_subtree": sep_policy == "copy",
            "candidates": getattr(page, "_split_candidates", {})})
        return right.page_id, sep, promoted_flag

    def _split_internal(self, txn: int, page: Page, sep_policy: str,
                        trace: dict, depth: int):
        # Reuse page.page_id as LEFT so the parent's existing child pointer
        # stays valid; allocate only the new RIGHT page.
        idx = self._split_index(page)
        boundary = page.cells[idx - 1]
        children = [page.link] + [c.child for c in page.cells]
        sep = boundary.key
        # LEFT:  link=children[0],   keys cells[:idx-1], children up to idx-1
        # RIGHT: link=children[idx], keys cells[idx:],   children idx+1 ...
        left_cells = [Cell(c.key, child=c.child) for c in page.cells[:idx - 1]]
        right_cells = [Cell(c.key, child=c.child) for c in page.cells[idx:]]
        page.link = children[0]
        page.cells = left_cells
        right = self._new_page(txn, T_INTERNAL, link=children[idx])
        right.cells = right_cells
        self._touch(page)
        trace["splits"].append({
            "depth": depth, "page_id": page.page_id,
            "new_page_id": right.page_id, "kind": "internal",
            "separator_key_hex": sep.hex(), "separator_policy": sep_policy,
            "separator_in_right_subtree": sep_policy == "copy",
            "candidates": getattr(page, "_split_candidates", {})})
        return right.page_id, sep, sep_policy == "promote"

    def delete(self, txn: int, key: bytes) -> dict:
        trace = {"op": "delete", "key_hex": key.hex(), "emptied_page": None}
        leaf = self._find_leaf(key)
        before = len(leaf.cells)
        leaf.cells = [c for c in leaf.cells if c.key != key]
        if before > 0 and not leaf.cells and leaf.page_id != self.meta.root:
            trace["emptied_page"] = leaf.page_id
            # empty non-root leaf stays on disk but is unlinked from its parent
            self._unlink_child(txn, leaf.page_id)
        return trace

    def _unlink_child(self, txn: int, child_id: int) -> None:
        pid = self.meta.root
        while self.pages[pid].page_type == T_INTERNAL:
            page = self.pages[pid]
            if page.link == child_id:
                page.link = page.cells[0].child
                page.cells = page.cells[1:]
                return
            for i, cell in enumerate(page.cells):
                if cell.child == child_id:
                    page.cells.pop(i)
                    return
            pid = page.link if page.link != child_id else page.cells[0].child

    # ---- scans / snapshots -------------------------------------------------
    def reachable_page_ids(self) -> set[int]:
        seen = {0}
        stack = [self.meta.root]
        while stack:
            pid = stack.pop()
            if pid in seen or pid not in self.pages:
                continue
            seen.add(pid)
            page = self.pages[pid]
            if page.page_type == T_INTERNAL:
                stack.append(page.link)
                stack.extend(c.child for c in page.cells)
        return seen

    def all_leaf_keys(self) -> list[bytes]:
        pid = self.meta.root
        keys = []
        # descend via internal pointers
        while self.pages[pid].page_type == T_INTERNAL:
            pid = self.pages[pid].link
        while pid:
            page = self.pages.get(pid)
            if page is None:
                break
            keys.extend(c.key for c in page.cells)
            pid = page.link
        return keys

    def tree_view(self) -> dict:
        def node(pid: int, depth: int) -> dict:
            page = self.pages.get(pid)
            if page is None:
                return {"page_id": pid, "missing": True}
            d = {"page_id": pid, "type": page.page_type, "root": page.is_root,
                 "lsn": page.lsn, "free_space": page.free_space,
                 "nslots": len(page.cells),
                 "sibling_or_leftmost": page.link, "cells": page.cells and
                 [c.to_dict() for c in page.cells]}
            if page.page_type == T_INTERNAL:
                d["children"] = [node(page.link, depth + 1)] +                     [node(c.child, depth + 1) for c in page.cells]
            return d
        return {"root": self.meta.root, "tree": node(self.meta.root, 0),
                "max_page_id": self.meta.max_page_id,
                "lsn": self.lsn, "page_size": self.page_size}


class CrashInjection(Exception):
    def __init__(self, point: str, ref):
        super().__init__(f"crash injected: {point} ({ref})")
        self.point = point
        self.ref = ref


def recover(disk: dict[int, bytes], wal: list[WalRecord],
            page_size: int, checkpoint_lsn: int) -> dict:
    """Replay WAL onto the paged file, then undo losers.

    * torn pages (checksum failures) are reconstructed from committed
      after-images / before-images;
    * a page flushed by a transaction that never commits is restored from its
      before-image, so no reachable half-page survives;
    * running recovery a second time changes nothing (idempotent).
    """
    report = {"torn_pages": [], "redone": [], "undone": [], "orphans": [],
              "idempotent": False, "reachable_after": []}
    state = {pid: (raw if fnv1a(raw[0:24]) == _u32(raw, 24) else None)
             for pid, raw in disk.items()}
    for pid, raw in disk.items():
        if state[pid] is None:
            report["torn_pages"].append(pid)

    # find checkpoint / committed txns / winner lsn set
    committed: set[int] = set()
    start_lsn = checkpoint_lsn
    for rec in wal:
        if rec.kind == W_CHECKPOINT:
            start_lsn = rec.lsn
        elif rec.kind == W_COMMIT:
            committed.add(rec.txn)

    def apply(rec: WalRecord) -> None:
        if rec.kind in (W_UPDATE, W_ALLOC) and rec.after:
            state[rec.page_id] = rec.after

    # redo from last checkpoint forward (all records incl. losers)
    for rec in wal:
        if rec.lsn <= start_lsn:
            continue
        if rec.kind in (W_UPDATE, W_ALLOC):
            cur = state.get(rec.page_id)
            cur_lsn = _u32(cur, 18) if cur else -1
            if cur_lsn < rec.lsn and rec.after:
                apply(rec)
                report["redone"].append(rec.lsn)
        elif rec.kind == W_COMMIT:
            committed.add(rec.txn)

    # undo losers in reverse lsn using before-images
    loser_pages: set[int] = set()
    for rec in reversed(wal):
        if rec.lsn <= start_lsn:
            continue
        if rec.kind in (W_UPDATE, W_ALLOC) and rec.txn not in committed:
            loser_pages.add(rec.page_id)
            if rec.kind == W_UPDATE and rec.before is not None:
                state[rec.page_id] = rec.before
            else:  # alloc of a page whose txn lost: page disappears
                state.pop(rec.page_id, None)
            report["undone"].append(rec.lsn)

    # rebuild engine purely from recovered images for reachability
    eng = Engine._from_state(state, page_size)
    reachable = eng.reachable_page_ids()
    report["reachable_after"] = sorted(reachable)
    for pid in list(state):
        if pid != 0 and pid not in reachable:
            report["orphans"].append(pid)
    # invariant: every reachable page must be a valid, non-torn page
    report["no_reachable_half_page"] = all(
        state.get(pid) is not None and
        fnv1a(state[pid][0:24]) == _u32(state[pid], 24) for pid in reachable)
    return {"engine": eng, "disk": state, "report": report}


def _engine_from_state(self, state, page_size):  # pragma: no cover - replaced
    raise NotImplementedError
Engine._from_state = classmethod(
    lambda cls, state, page_size: _build_from_state(state, page_size))


def _build_from_state(state: dict[int, bytes], page_size: int) -> Engine:
    eng = Engine.__new__(Engine)
    eng.page_size = page_size
    eng.pages = {}
    eng.disk = dict(state)
    eng.wal = []
    eng.lsn = 0
    eng.txn = 0
    eng.checkpoint_lsn = 0
    eng.crash_after_disk_pages = None
    eng.crash_after_wal = False
    eng.crashed = False
    eng._new_pages = set()
    eng._pending = {}
    root = 1
    maxpid = 0
    for pid, raw in state.items():
        if raw is None:
            continue
        page = Page.decode(raw, page_size)
        eng.pages[pid] = page
        eng.lsn = max(eng.lsn, page.lsn)
        maxpid = max(maxpid, pid)
    if 0 in eng.pages:
        for cell in eng.pages[0].cells:
            if cell.key == b"root":
                root = int.from_bytes(cell.value, "big")
    eng.meta = Meta(root=root, max_page_id=maxpid, lsn=eng.lsn)
    return eng
