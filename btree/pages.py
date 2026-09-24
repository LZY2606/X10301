"""On-disk page byte layout.

Coordinate basis (all offsets are byte offsets inside a fixed-size page):

  [0 .. 2)   magic        uint16
  [2 .. 6)   page_id      uint32
  [6]        kind         uint8  (1=leaf, 2=internal)
  [7 .. 9)   nslots       uint16
  [9 .. 11)  free_start   uint16  first free byte after the cell area
  [11 .. 15) right_sibling uint32 (0xFFFFFFFF = none)
  [15 .. 19) parent       uint32 (0xFFFFFFFF = root)
  [19]       reserved     uint8
  [19 .. 21) wait -- see HEADER layout below

Cell area grows left-to-right from HEADER_SIZE; the slot directory grows
right-to-left from PAGE_SIZE as uint16 cell offsets, ordered by key bytes.

Leaf cell:     keylen(uint16) key bytes vallen(uint32) value bytes
Internal cell: keylen(uint16) key bytes child(uint32)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

MAGIC = 0xB7EE
KIND_LEAF = 1
KIND_INTERNAL = 2
NO_PAGE = 0xFFFFFFFF
META_PAGE_ID = 0

# magic H, page_id I, kind B, nslots H, free_start H,
# sibling I, parent I, reserved B, crc H
_HEADER = struct.Struct(">HIBHHIIBH")
HEADER_SIZE = _header_size = _HEADER.size  # 22 bytes
CRC_OFFSET = 19 + 1  # position of crc field

KEYLEN = struct.Struct(">H")
VALLEN = struct.Struct(">I")
CHILD = struct.Struct(">I")
SLOT = struct.Struct(">H")


class PageFormatError(Exception):
    """Structural corruption: bad magic or checksum mismatch (torn page)."""


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if (crc & 1) else (crc >> 1)
    return (~crc) & 0xFFFF


def encode_cell(key: bytes, value: Optional[bytes] = None, child: Optional[int] = None) -> bytes:
    head = KEYLEN.pack(len(key)) + key
    if child is not None:
        return head + CHILD.pack(child)
    assert value is not None
    return head + VALLEN.pack(len(value)) + value


def cell_size(key: bytes, value: Optional[bytes] = None, child: Optional[int] = None) -> int:
    base = KEYLEN.size + len(key)
    return base + CHILD.size if child is not None else base + VALLEN.size + len(value or b"")


@dataclass
class Cell:
    key: bytes
    value: Optional[bytes] = None
    child: Optional[int] = None
    offset: int = 0
    length: int = 0

    @property
    def kind(self) -> str:
        return "internal" if self.child is not None else "leaf"

    def encode(self) -> bytes:
        return encode_cell(self.key, self.value, self.child)

    def key_repr(self) -> str:
        try:
            text = self.key.decode("utf-8")
            if all(32 <= ord(c) < 127 for c in text):
                return text
        except UnicodeDecodeError:
            pass
        return "0x" + self.key.hex()


@dataclass
class Page:
    page_id: int
    kind: int
    cells: list[Cell] = field(default_factory=list)
    right_sibling: int = NO_PAGE
    parent: int = NO_PAGE
    leftmost: int = NO_PAGE
    page_size: int = 0

    # -- construction -------------------------------------------------------
    @classmethod
    def create(cls, page_id: int, kind: int, page_size: int,
               parent: int = NO_PAGE, sibling: int = NO_PAGE,
               leftmost: int = NO_PAGE) -> "Page":
        page = cls(page_id, kind, [], sibling, parent, page_size)
        page.leftmost = leftmost
        return page

    # -- parsing ------------------------------------------------------------
    @classmethod
    def parse(cls, buf: bytes, page_id: int) -> "Page":
        if len(buf) < HEADER_SIZE or struct.unpack(">H", buf[0:2])[0] != MAGIC:
            raise PageFormatError(f"page {page_id}: bad magic / short read")
        body_for_crc = buf[:CRC_OFFSET] + buf[CRC_OFFSET + 2:]
        magic, pid, kind, nslots, free_start, sibling, parent, leftmost, crc = _HEADER.unpack(
            buf[:HEADER_SIZE])
        if crc16(body_for_crc) != crc:
            raise PageFormatError(f"page {page_id}: crc mismatch (torn write)")
        page = cls(pid, kind, [], sibling, parent, len(buf))
        page.leftmost = leftmost
        slot_dir_start = len(buf) - SLOT.size * nslots
        for i in range(nslots):
            off = SLOT.unpack_from(buf, slot_dir_start + i * SLOT.size)[0]
            page._append_cell_from(buf, off)
        if page.cells and any(
            page.cells[i].key > page.cells[i + 1].key for i in range(len(page.cells) - 1)):
            raise PageFormatError(f"page {page_id}: slot directory not key ordered")
        return page

    def _append_cell_from(self, buf: bytes, off: int) -> None:
        klen = KEYLEN.unpack_from(buf, off)[0]
        key = bytes(buf[off + KEYLEN.size: off + KEYLEN.size + klen])
        p = off + KEYLEN.size + klen
        if self.kind == KIND_INTERNAL:
            child = CHILD.unpack_from(buf, p)[0]
            length = (p + CHILD.size) - off
            self.cells.append(Cell(key, None, child, off, length))
        else:
            vlen = VALLEN.unpack_from(buf, p)[0]
            val = bytes(buf[p + VALLEN.size: p + VALLEN.size + vlen])
            length = (p + VALLEN.size + vlen) - off
            self.cells.append(Cell(key, val, None, off, length))

    # -- serialization ------------------------------------------------------
    def serialize(self) -> bytes:
        buf = bytearray(self.page_size)
        off = HEADER_SIZE
        for cell in self.cells:
            data = cell.encode()
            buf[off:off + len(data)] = data
            cell.offset = off
            cell.length = len(data)
            off += len(data)
        free_start = off
        slot_dir_start = self.page_size - SLOT.size * len(self.cells)
        for i, cell in enumerate(sorted(self.cells, key=lambda c: c.key)):
            SLOT.pack_into(buf, slot_dir_start + i * SLOT.size, cell.offset)
        header = _HEADER.pack(
            MAGIC, self.page_id, self.kind, len(self.cells), free_start,
            self.right_sibling, self.parent, self.leftmost, 0)
        buf[:HEADER_SIZE] = header
        crc = crc16(bytes(buf[:CRC_OFFSET]) + bytes(buf[CRC_OFFSET + 2:]))
        SLOT.pack_into(buf, CRC_OFFSET, crc)
        return bytes(buf)

    # -- derived metrics ----------------------------------------------------
    def used_bytes(self) -> int:
        return sum(c.length if c.length else cell_size(c.key, c.value, c.child)
                   for c in self.cells) + SLOT.size * len(self.cells)

    def free_bytes(self) -> int:
        return self.page_size - HEADER_SIZE - self.used_bytes()

    def free_interval(self) -> tuple[int, int]:
        """Byte coordinate [free_start, slot_directory_start) of free space."""
        free_start = HEADER_SIZE + sum(
            c.length if c.length else cell_size(c.key, c.value, c.child)
            for c in self.cells)
        return free_start, self.page_size - SLOT.size * len(self.cells)

    def slot_offsets_in_key_order(self) -> list[int]:
        tmp = sorted(self.cells, key=lambda c: c.key)
        return [c.offset for c in tmp]

    def find_cell_index(self, key: bytes) -> int:
        for i, c in enumerate(self.cells):
            if c.key == key:
                return i
        return -1
