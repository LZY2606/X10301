"""Fixed-size B-tree page encoding.

Page byte layout (page_size bytes, default 256):

    offset 0..39   page header
    offset 32..    slot directory (2 bytes/slot, grows toward high offsets)
    offset ..page_size-1  record cells (grow downward from the end)

Header (little endian):
    magic u16, type u8, flags u8, free_start u16, slot_end u16,
    lsn u64, page_id u32, prev_page i32, next_page i32, rightmost_child i32,
    nslots u16, crc32 u32

All free-space accounting is byte based: splits in this project are decided
by byte space, never by key count.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

MAGIC = 0x4254  # 'BT'
PAGE_HEADER_SIZE = 38
SLOT_ENTRY_SIZE = 4

TYPE_LEAF = 1
TYPE_INTERNAL = 2

FLAG_DIRTY = 0x01
FLAG_HALF_PAGE = 0x02  # page write interrupted before the WAL flush

NO_PAGE = -1

_HEADER = struct.Struct("<HBBHHQIiiiHI")
assert _HEADER.size == PAGE_HEADER_SIZE
_SLOT = struct.Struct("<HH")

# Record cell header: length u16, key_len u16 (payload length follows).
_CELL = struct.Struct("<HH")


@dataclass
class Slot:
    offset: int
    length: int


@dataclass
class Cell:
    key: bytes
    value: bytes = b""
    child: int = NO_PAGE  # internal entry: left child; rightmost lives in header


class PageFormatError(Exception):
    pass


def cell_size(key: bytes, value: bytes = b"", *, internal: bool = False) -> int:
    if internal:
        # internal cell stores child (4B) + key_len (2B) + key; value unused
        return 4 + 2 + len(key)
    return _CELL.size + len(key) + len(value)


class Page:
    """An in-memory decoded page with byte-accurate layout bookkeeping."""

    def __init__(self, page_id: int, page_size: int, page_type: int = TYPE_LEAF,
                 prev_page: int = NO_PAGE, next_page: int = NO_PAGE,
                 rightmost_child: int = NO_PAGE, lsn: int = 0,
                 flags: int = 0):
        self.page_size = page_size
        self.page_id = page_id
        self.page_type = page_type
        self.flags = flags
        self.free_start = page_size  # high-water of the record area
        self.slot_end = PAGE_HEADER_SIZE  # end of slot directory
        self.lsn = lsn
        self.prev_page = prev_page
        self.next_page = next_page
        self.rightmost_child = rightmost_child
        self.slots: List[Slot] = []
        self.cells: List[Cell] = []  # index aligned with slots

    # ------------------------------------------------------------------ sizes
    @property
    def free_space(self) -> int:
        """Usable bytes between the record high-water and the slot directory."""
        return self.free_start - self.slot_end

    @property
    def used_bytes(self) -> int:
        return self.page_size - self.free_space

    def can_fit(self, size: int) -> bool:
        return self.free_space >= size + SLOT_ENTRY_SIZE

    # ------------------------------------------------------------------ cells
    def _cell_bytes(self, cell: Cell) -> bytes:
        if self.page_type == TYPE_INTERNAL:
            return struct.pack("<IH", cell.child & 0xFFFFFFFF, len(cell.key)) + cell.key
        return _CELL.pack(len(cell.key), len(cell.value)) + cell.key + cell.value

    @staticmethod
    def _decode_cell(buf: bytes, page_type: int) -> Cell:
        if page_type == TYPE_INTERNAL:
            child, klen = struct.unpack_from("<IH", buf, 0)
            signed_child = child if child < 0x80000000 else child - 0x100000000
            key = buf[6:6 + klen]
            return Cell(key=key, child=signed_child)
        klen, vlen = struct.unpack_from("<HH", buf, 0)
        key = buf[4:4 + klen]
        value = buf[4 + klen:4 + klen + vlen]
        return Cell(key=key, value=value)

    def append_cell(self, cell: Cell) -> None:
        body = self._cell_bytes(cell)
        size = len(body)
        if not self.can_fit(size):
            raise PageFormatError(
                f"page {self.page_id}: need {size + SLOT_ENTRY_SIZE}B, "
                f"free {self.free_space}B")
        offset = self.free_start - size
        self.free_start = offset
        self.slots.append(Slot(offset, size))
        self.cells.append(cell)

    def rebuild(self) -> None:
        """Re-serialize all cells, rebuilding the slot directory in place.

        Cells are laid out in their current list order; the caller is
        responsible for sorting. Used after insert/replace/delete/split.
        """
        bodies = [self._cell_bytes(c) for c in self.cells]
        needed = sum(len(b) + SLOT_ENTRY_SIZE for b in bodies)
        if needed > self.page_size - PAGE_HEADER_SIZE:
            raise PageFormatError(
                f"page {self.page_id}: {needed}B do not fit in "
                f"{self.page_size - PAGE_HEADER_SIZE}B usable bytes")
        self.slots = []
        self.free_start = self.page_size
        self.slot_end = PAGE_HEADER_SIZE + SLOT_ENTRY_SIZE * len(bodies)
        cursor = self.page_size
        for body in bodies:
            cursor -= len(body)
            self.slots.append(Slot(cursor, len(body)))
        self.free_start = cursor

    # ---------------------------------------------------------------- encode
    def encode(self) -> bytes:
        buf = bytearray(self.page_size)
        for i, slot in enumerate(self.slots):
            _SLOT.pack_into(buf, PAGE_HEADER_SIZE + 4 * i, slot.offset, slot.length)
        for slot, cell in zip(self.slots, self.cells):
            buf[slot.offset:slot.offset + slot.length] = self._cell_bytes(cell)
        prevp = self.prev_page if self.prev_page >= 0 else NO_PAGE
        nextp = self.next_page if self.next_page >= 0 else NO_PAGE
        rchild = self.rightmost_child
        _HEADER.pack_into(
            buf, 0, MAGIC, self.page_type, self.flags, self.free_start,
            self.slot_end, self.lsn, self.page_id,
            prevp, nextp, rchild, len(self.slots), 0)
        crc = zlib.crc32(bytes(buf[:34]) + bytes(buf[38:])) & 0xFFFFFFFF
        struct.pack_into("<I", buf, 34, crc)
        return bytes(buf)

    @classmethod
    def decode(cls, raw: bytes) -> "Page":
        (magic, ptype, flags, free_start, slot_end, lsn, pid,
         prevp, nextp, rchild, nslots, crc) = _HEADER.unpack_from(raw, 0)
        if magic != MAGIC:
            raise PageFormatError("bad magic")
        expect = zlib.crc32(raw[:34] + raw[38:]) & 0xFFFFFFFF
        if crc != expect:
            raise PageFormatError(f"crc mismatch on page {pid}: torn page")
        prev = prevp if prevp != 0xFFFFFFFF else NO_PAGE
        nxt = nextp if nextp != 0xFFFFFFFF else NO_PAGE
        page = cls(
            page_id=pid if pid < 0x80000000 else pid - 0x100000000,
            page_size=len(raw), page_type=ptype, prev_page=prev,
            next_page=nxt, rightmost_child=rchild, lsn=lsn, flags=flags)
        page.free_start = free_start
        page.slot_end = slot_end
        for i in range(nslots):
            off, length = _SLOT.unpack_from(raw, PAGE_HEADER_SIZE + 4 * i)
            page.slots.append(Slot(off, length))
            page.cells.append(cls._decode_cell(bytes(raw[off:off + length]), ptype))
        return page

    def half_page(self) -> "Page":
        """Return a copy flagged as an interrupted (torn) write."""
        torn = Page(self.page_id, self.page_size, self.page_type,
                    self.prev_page, self.next_page, self.rightmost_child,
                    self.lsn, self.flags | FLAG_HALF_PAGE)
        torn.free_start = self.free_start
        torn.slot_end = self.slot_end
        torn.slots = list(self.slots)
        torn.cells = list(self.cells)
        return torn


# --------------------------------------------------------------------- split
@dataclass
class SplitPlan:
    index: int                 # first slot that moves to the new page
    left_bytes: int
    right_bytes: int
    candidates: list  # list[int] legal first-slot indices
    ambiguous: bool


def byte_split_plan(sizes: List[int], bias: str = "balanced",
                    *, internal: bool = False) -> SplitPlan:
    """Choose a split point using byte space.

    bias="balanced" picks the cut minimizing |left-right| used bytes.
    bias="left_heavy" prefers the larger left side (v2 parameter revision).
    Every non-empty side is legal; ties produce multiple candidates and an
    ambiguous verdict so the archive keeps all legal options.
    """
    arr = np.asarray(sizes, dtype=np.int64)
    n = len(arr)
    if n < 2:
        raise PageFormatError("need at least 2 cells to split")
    prefix = np.cumsum(arr)
    total = int(prefix[-1])
    cuts = np.arange(1, n)
    left = prefix[:-1]
    right = total - left
    diff = np.abs(left - right)
    best = int(diff.min())
    legal = [int(c) for c, d in zip(cuts, diff) if int(d) == best]
    if bias == "left_heavy":
        index = max(legal)
    elif bias == "balanced":
        index = legal[0]
    else:
        raise ValueError(f"unknown split bias: {bias!r}")
    ambiguous = len(legal) > 1
    return SplitPlan(index=index, left_bytes=int(prefix[index - 1]),
                     right_bytes=total - int(prefix[index - 1]),
                     candidates=legal, ambiguous=ambiguous)


def leaf_separator_key(keys: List[bytes], cut: int, policy: str) -> bytes:
    """Leaf split separator rule (must stay consistent with promotion)."""
    if policy not in ("copy", "promote"):
        raise ValueError(f"unknown separator policy: {policy!r}")
    return bytes(keys[cut])  # first key of the right leaf


def verify_separator_rule(page: Page, promoted_keys: List[bytes],
                          policy: str) -> Optional[str]:
    """Return None when the separator rule is consistent, else a diagnosis."""
    if page.page_type != TYPE_INTERNAL:
        return None
    keys = [c.key for c in page.cells]
    if keys != promoted_keys:
        return (f"separator mismatch: internal {keys!r} vs promoted "
                f"{promoted_keys!r} under policy {policy!r}")
    if policy not in ("copy", "promote"):
        return f"unknown separator policy {policy!r}"
    return None
