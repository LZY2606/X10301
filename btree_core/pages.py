"""Fixed-size B-tree page layout.

Physical layout (all integers big-endian):
  header (22 bytes):
    page_id       u32
    ptype         u8   (1=leaf, 2=internal)
    flags         u8
    n_slots       u16
    free_start    u16  (end of slot directory)
    free_end      u16  (start of cell area)
    right_sibling u32  (leaf sibling link, 0 = none)
    left_child    u32  (internal nodes only, 0 for leaf)
    checksum      u16  (numpy byte-sum over image with this field zeroed)
  slot directory: n_slots * u16 offsets, growing upward from HEADER_SIZE
  cell area: cells growing downward from page_size

Quantities carried here are always in unit "byte" with coordinate basis
"page_offset" (0-based offset from the start of the page image).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import struct

import numpy as np

LEAF = 1
INTERNAL = 2
HEADER_FORMAT = ">IBBHHHIIH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 22
SLOT_SIZE = 2
UNIT_BYTE = "byte"
BASIS_PAGE_OFFSET = "page_offset"


class PageCorrupt(ValueError):
    """Raised when a page image fails structural validation (e.g. torn write)."""


@dataclass
class Cell:
    key: bytes
    value: bytes = b""   # leaf payload
    child: int = 0       # internal child page id

    def size(self, ptype: int) -> int:
        if ptype == LEAF:
            return 2 + len(self.key) + 2 + len(self.value)
        return 2 + len(self.key) + 4


def encode_cell(cell: Cell, ptype: int) -> bytes:
    if ptype == LEAF:
        return struct.pack(">H", len(cell.key)) + cell.key + struct.pack(">H", len(cell.value)) + cell.value
    return struct.pack(">H", len(cell.key)) + cell.key + struct.pack(">I", cell.child)


def decode_cell(buf: bytes, off: int, ptype: int) -> Cell:
    (klen,) = struct.unpack_from(">H", buf, off)
    key = bytes(buf[off + 2: off + 2 + klen])
    if len(key) != klen:
        raise PageCorrupt("cell key overruns page image")
    pos = off + 2 + klen
    if ptype == LEAF:
        (vlen,) = struct.unpack_from(">H", buf, pos)
        value = bytes(buf[pos + 2: pos + 2 + vlen])
        if len(value) != vlen:
            raise PageCorrupt("cell value overruns page image")
        return Cell(key=key, value=value)
    (child,) = struct.unpack_from(">I", buf, pos)
    return Cell(key=key, child=child)


@dataclass
class Page:
    page_id: int
    ptype: int
    right_sibling: int = 0
    left_child: int = 0
    lsn: int = 0
    cells: list = field(default_factory=list)

    def keys(self):
        return [c.key for c in self.cells]

    def used_bytes(self) -> int:
        return HEADER_SIZE + SLOT_SIZE * len(self.cells) + sum(c.size(self.ptype) for c in self.cells)

    def free_space(self, page_size: int) -> int:
        """Free bytes between slot directory and cell area (unit: byte)."""
        return page_size - self.used_bytes()

    def slot_directory(self, page_size: int) -> list:
        """Slot offsets as they would appear in the serialized image."""
        offsets = []
        pos = page_size
        for cell in self.cells:
            pos -= cell.size(self.ptype)
            offsets.append(pos)
        return offsets

    def to_bytes(self, page_size: int) -> bytes:
        buf = bytearray(page_size)
        pos = page_size
        offsets = []
        for cell in self.cells:
            data = encode_cell(cell, self.ptype)
            pos -= len(data)
            if pos < HEADER_SIZE + SLOT_SIZE * len(self.cells):
                raise PageCorrupt("page overflow during serialization")
            buf[pos:pos + len(data)] = data
            offsets.append(pos)
        free_start = HEADER_SIZE + SLOT_SIZE * len(self.cells)
        free_end = pos
        struct.pack_into(
            HEADER_FORMAT, buf, 0,
            self.page_id, self.ptype, 0, len(self.cells),
            free_start, free_end, self.right_sibling, self.left_child, 0,
        )
        for i, off in enumerate(offsets):
            struct.pack_into(">H", buf, HEADER_SIZE + SLOT_SIZE * i, off)
        struct.pack_into(">H", buf, HEADER_SIZE - 2, image_checksum(bytes(buf)))
        return bytes(buf)

    @classmethod
    def from_bytes(cls, img: bytes, page_size: int, lsn: int = 0) -> "Page":
        if len(img) != page_size:
            raise PageCorrupt(f"image length {len(img)} != page_size {page_size}")
        page_id, ptype, _flags, n_slots, free_start, free_end, right_sibling, left_child, checksum = struct.unpack_from(
            HEADER_FORMAT, img, 0
        )
        zeroed = bytearray(img)
        struct.pack_into(">H", zeroed, HEADER_SIZE - 2, 0)
        if image_checksum(bytes(zeroed)) != checksum:
            raise PageCorrupt("page checksum mismatch (torn or partial write)")
        if ptype not in (LEAF, INTERNAL):
            raise PageCorrupt(f"unknown page type {ptype}")
        if not (HEADER_SIZE <= free_start <= free_end <= page_size):
            raise PageCorrupt("free space bounds outside page")
        if free_start != HEADER_SIZE + SLOT_SIZE * n_slots:
            raise PageCorrupt("slot directory size mismatch")
        cells = []
        for i in range(n_slots):
            (off,) = struct.unpack_from(">H", img, HEADER_SIZE + SLOT_SIZE * i)
            if not (free_end <= off < page_size):
                raise PageCorrupt(f"slot {i} offset {off} outside cell area")
            cells.append(decode_cell(img, off, ptype))
        keys = [c.key for c in cells]
        if keys != sorted(keys):
            raise PageCorrupt("cells out of key order")
        return cls(
            page_id=page_id, ptype=ptype, right_sibling=right_sibling,
            left_child=left_child, lsn=lsn, cells=cells,
        )


def image_checksum(img: bytes) -> int:
    """Checksum over the raw image (numpy over the byte buffer)."""
    arr = np.frombuffer(img, dtype=np.uint8)
    return int(arr.sum(dtype=np.uint64) % 65521)


def validate_image(img: bytes, page_size: int) -> list:
    """Return a list of diagnostics; empty list means the page is structurally whole."""
    try:
        Page.from_bytes(img, page_size)
        return []
    except PageCorrupt as exc:
        return [str(exc)]


class Pager:
    """Logical in-memory page pool plus page-id allocation log."""

    def __init__(self, page_size: int):
        self.page_size = page_size
        self.pages = {}
        self.next_page_id = 1
        self.freed = []
        self.alloc_log = []

    def alloc(self, ptype: int) -> Page:
        page = Page(page_id=self.next_page_id, ptype=ptype)
        self.alloc_log.append({"action": "alloc", "page_id": page.page_id, "ptype": ptype})
        self.next_page_id += 1
        self.pages[page.page_id] = page
        return page

    def free(self, page_id: int) -> None:
        self.freed.append(page_id)
        self.alloc_log.append({"action": "free", "page_id": page_id})
        self.pages.pop(page_id, None)

    def snapshot(self) -> dict:
        return {pid: p.to_bytes(self.page_size).hex() for pid, p in sorted(self.pages.items())}
