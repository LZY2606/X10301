"""Fixed-size page binary layout for the teaching B-tree.

Byte map (all integers big-endian), ``HEADER_SIZE = 40``::

    off  size  field
      0   2    magic 0xB7EE
      2   1    format version
      3   1    page type  (1 = leaf, 2 = internal, 0 = freshly allocated)
      4   4    page id
      8   4    parent page id      (0xFFFFFFFF = none / root)
     12   4    prev sibling pid    (leaves only; 0xFFFFFFFF = none)
     16   4    next sibling pid    (leaves only; 0xFFFFFFFF = none)
     20   4    leftmost child pid  (internal only; 0xFFFFFFFF for leaves)
     24   4    page LSN
     28   2    free_start          (first free byte after packed cells)
     30   2    free_end            (first byte of the slot directory)
     32   2    slot count
     34   2    flags               (bit0 = empty/deleted leaf)
     36   4    reserved (zero)

Cells grow down from the end of the page; the slot directory grows up from
offset 40.  Slot ``i`` is a ``uint16`` absolute byte offset of cell ``i``.
Leaf cell:  ``u16 klen | u16 vlen | key | value``.
Internal cell: ``u16 klen | u32 child | separator key``; an internal page
with N separators has N+1 children, the leftmost child living in the header.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

MAGIC = 0xB7EE
VERSION = 1
LEAF = 1
INTERNAL = 2
FRESH = 0
HEADER_SIZE = 40
NONE = 0xFFFFFFFF
FLAG_EMPTY = 0x0001
DEFAULT_PAGE_SIZE = 256


class PageFormatError(ValueError):
    pass


def _enc_parts(buf: bytearray, offset: int, size: int, value: int) -> None:
    buf[offset:offset + size] = int(value).to_bytes(size, "big")


@dataclass
class Page:
    pid: int
    ptype: int = FRESH
    page_size: int = DEFAULT_PAGE_SIZE
    parent: int = NONE
    prev_sibling: int = NONE
    next_sibling: int = NONE
    leftmost: int = NONE
    page_lsn: int = 0
    flags: int = 0
    # leaf items: list[(key: bytes, value: bytes)]
    # internal items: list[(sep: bytes, child: int)]
    items: list = field(default_factory=list)

    # ------------------------------------------------------------------ basic
    @property
    def is_leaf(self) -> bool:
        return self.ptype == LEAF

    @property
    def is_internal(self) -> bool:
        return self.ptype == INTERNAL

    @property
    def empty_flag(self) -> bool:
        return bool(self.flags & FLAG_EMPTY)

    def mark_empty(self, on: bool = True) -> None:
        if on:
            self.flags |= FLAG_EMPTY
        else:
            self.flags &= ~FLAG_EMPTY

    def cell_size(self, item) -> int:
        if self.is_leaf:
            key, value = item
            return 4 + len(key) + len(value)
        if self.is_internal:
            sep, _child = item
            return 2 + 4 + len(sep)
        raise PageFormatError(f"page {self.pid} has no usable type")

    @property
    def slot_count(self) -> int:
        return len(self.items)

    @property
    def used_bytes(self) -> int:
        return HEADER_SIZE + 2 * len(self.items) + sum(
            self.cell_size(it) for it in self.items
        )

    @property
    def free_bytes(self) -> int:
        free = self.page_size - self.used_bytes
        if self.ptype == FRESH:
            return self.page_size - HEADER_SIZE
        return free

    # ------------------------------------------------------------- serialize
    def serialize(self) -> bytes:
        buf = bytearray(self.page_size)
        _enc_parts(buf, 0, 2, MAGIC)
        buf[2] = VERSION
        buf[3] = self.ptype
        _enc_parts(buf, 4, 4, self.pid)
        _enc_parts(buf, 8, 4, self.parent)
        _enc_parts(buf, 12, 4, self.prev_sibling)
        _enc_parts(buf, 16, 4, self.next_sibling)
        _enc_parts(buf, 20, 4, self.leftmost)
        _enc_parts(buf, 24, 4, self.page_lsn)
        # pack cells downwards
        cursor = self.page_size
        slots: list[int] = []
        for item in self.items:
            if self.is_leaf:
                key, value = item
                cell = (
                    len(key).to_bytes(2, "big")
                    + len(value).to_bytes(2, "big")
                    + key
                    + value
                )
            else:
                sep, child = item
                cell = (
                    len(sep).to_bytes(2, "big")
                    + int(child).to_bytes(4, "big")
                    + sep
                )
            cursor -= len(cell)
            buf[cursor:cursor + len(cell)] = cell
            slots.append(cursor)
        free_end = cursor
        free_start = HEADER_SIZE
        _enc_parts(buf, 28, 2, free_start)
        _enc_parts(buf, 30, 2, free_end)
        _enc_parts(buf, 32, 2, len(self.items))
        _enc_parts(buf, 34, 2, self.flags)
        # slot directory grows upwards; slot order == logical key order
        for i, off in enumerate(slots):
            _enc_parts(buf, HEADER_SIZE + 2 * i, 2, off)
        return bytes(buf)

    @classmethod
    def parse(cls, raw: bytes, page_size: int = DEFAULT_PAGE_SIZE) -> "Page":
        if len(raw) != page_size:
            raise PageFormatError(
                f"page blob is {len(raw)} bytes, expected {page_size}"
            )
        magic = int.from_bytes(raw[0:2], "big")
        if magic != MAGIC:
            raise PageFormatError(f"bad magic 0x{magic:04x}")
        ptype = raw[3]
        pid = int.from_bytes(raw[4:8], "big")
        page = cls(
            pid=pid,
            ptype=ptype,
            page_size=page_size,
            parent=int.from_bytes(raw[8:12], "big"),
            prev_sibling=int.from_bytes(raw[12:16], "big"),
            next_sibling=int.from_bytes(raw[16:20], "big"),
            leftmost=int.from_bytes(raw[20:24], "big"),
            page_lsn=int.from_bytes(raw[24:28], "big"),
            flags=int.from_bytes(raw[34:36], "big"),
        )
        if ptype == FRESH:
            return page
        nslots = int.from_bytes(raw[32:34], "big")
        items: list = []
        for i in range(nslots):
            off = int.from_bytes(
                raw[HEADER_SIZE + 2 * i:HEADER_SIZE + 2 * i + 2], "big"
            )
            if page.is_leaf:
                klen = int.from_bytes(raw[off:off + 2], "big")
                vlen = int.from_bytes(raw[off + 2:off + 4], "big")
                key = bytes(raw[off + 4:off + 4 + klen])
                value = bytes(raw[off + 4 + klen:off + 4 + klen + vlen])
                items.append((key, value))
            elif page.is_internal:
                klen = int.from_bytes(raw[off:off + 2], "big")
                child = int.from_bytes(raw[off + 2:off + 6], "big")
                sep = bytes(raw[off + 6:off + 6 + klen])
                items.append((sep, child))
            else:
                raise PageFormatError(f"page {pid} unknown type {ptype}")
        page.items = items
        return page

    # ----------------------------------------------------------- navigation
    def leaf_index(self, key: bytes) -> tuple[int, bool]:
        """Return (insert/find position, exact?) for a leaf key."""
        lo, hi = 0, len(self.items)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.items[mid][0] < key:
                lo = mid + 1
            else:
                hi = mid
        exact = lo < len(self.items) and self.items[lo][0] == key
        return lo, exact

    def child_for(self, key: bytes) -> int:
        """Pick the child pointer for ``key`` on an internal page."""
        child = self.leftmost
        for sep, cand in self.items:
            if key < sep:
                return child
            child = cand
        return child

    def layout_regions(self) -> list[dict]:
        """Annotate every byte region of the page for the layout view."""
        raw = self.serialize()
        regions: list[dict] = []
        header_fields = [
            (0, 2, "magic"), (2, 1, "version"), (3, 1, "type"),
            (4, 4, "page_id"), (8, 4, "parent"),
            (12, 4, "prev_sibling"), (16, 4, "next_sibling"),
            (20, 4, "leftmost"), (24, 4, "page_lsn"),
            (28, 2, "free_start"), (30, 2, "free_end"),
            (32, 2, "slot_count"), (34, 2, "flags"), (36, 4, "reserved"),
        ]
        for start, size, name in header_fields:
            regions.append({
                "kind": "header", "start": start, "end": start + size,
                "label": name, "bytes": raw[start:start + size].hex(),
            })
        slot_end = HEADER_SIZE + 2 * len(self.items)
        if HEADER_SIZE < slot_end:
            regions.append({
                "kind": "slot_dir", "start": HEADER_SIZE, "end": slot_end,
                "label": "slot directory",
            })
        for i, off in enumerate(int.from_bytes(
                raw[HEADER_SIZE + 2 * i:HEADER_SIZE + 2 * i + 2], "big")
                for i in range(len(self.items))):
            end = off + self.cell_size(self.items[i])
            regions.append({
                "kind": "cell", "start": off, "end": end,
                "label": f"cell@slot{i}", "slot": i,
            })
        free_end = int.from_bytes(raw[30:32], "big")
        if slot_end < free_end:
            regions.append({
                "kind": "free", "start": slot_end, "end": free_end,
                "label": "free space",
            })
        regions.sort(key=lambda r: (r["start"], r["end"]))
        return regions


def fresh_page(pid: int, page_size: int = DEFAULT_PAGE_SIZE) -> Page:
    return Page(pid=pid, ptype=FRESH, page_size=page_size)


def leaf_page(pid: int, page_size: int = DEFAULT_PAGE_SIZE, **kw) -> Page:
    return Page(pid=pid, ptype=LEAF, page_size=page_size, **kw)


def internal_page(pid: int, page_size: int = DEFAULT_PAGE_SIZE, **kw) -> Page:
    return Page(pid=pid, ptype=INTERNAL, page_size=page_size, **kw)
