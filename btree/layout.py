"""Fixed-size page byte layout with slot directory, free space and CRC.

Layout (all integers little-endian):
  header 24B: page_id u32 | flags u8 (bit0=leaf) | reserved u8 |
              slot_count u16 | free_start u16 | free_end u16 |
              right_sibling u32 | left_child u32 | lsn u32
  slot directory: slot_count * 4B (offset u16, length u16), grows upward
  cells: stored from (page_size - CRC_SIZE) downward
  crc16: last 2 bytes of the page, over bytes [0, page_size-2)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

HEADER_SIZE = 24
SLOT_SIZE = 4
CRC_SIZE = 2
NO_PAGE = 0xFFFFFFFF
FLAG_LEAF = 0x01

_HDR = struct.Struct("<IBBHHHIII")


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc & 0xFFFF


@dataclass
class Cell:
    key: bytes
    value: bytes | None = None   # leaf cells carry a value
    child: int | None = None     # internal cells carry a child page id

    def encoded_size(self) -> int:
        if self.child is not None:
            return 2 + len(self.key) + 4
        return 2 + len(self.key) + 2 + len(self.value or b"")

    def encode(self) -> bytes:
        if self.child is not None:
            return struct.pack("<H", len(self.key)) + self.key + struct.pack("<I", self.child)
        v = self.value or b""
        return struct.pack("<H", len(self.key)) + self.key + struct.pack("<H", len(v)) + v

    @staticmethod
    def decode(buf: bytes, is_leaf: bool) -> "Cell":
        klen = struct.unpack_from("<H", buf, 0)[0]
        key = buf[2:2 + klen]
        if is_leaf:
            vlen = struct.unpack_from("<H", buf, 2 + klen)[0]
            return Cell(key=key, value=buf[4 + klen:4 + klen + vlen])
        child = struct.unpack_from("<I", buf, 2 + klen)[0]
        return Cell(key=key, child=child)


@dataclass
class Page:
    page_id: int
    is_leaf: bool
    right_sibling: int = NO_PAGE
    left_child: int = NO_PAGE
    lsn: int = 0
    cells: list = field(default_factory=list)

    def used_bytes(self) -> int:
        return HEADER_SIZE + SLOT_SIZE * len(self.cells) + sum(c.encoded_size() for c in self.cells)

    def free_bytes(self, page_size: int) -> int:
        return page_size - CRC_SIZE - self.used_bytes()

    def fits(self, cell: Cell, page_size: int) -> bool:
        return self.free_bytes(page_size) >= cell.encoded_size() + SLOT_SIZE

    def encode(self, page_size: int) -> bytes:
        buf = np.zeros(page_size, dtype=np.uint8)
        free_start = HEADER_SIZE + SLOT_SIZE * len(self.cells)
        cursor = page_size - CRC_SIZE
        slots = []
        for cell in self.cells:
            raw = cell.encode()
            cursor -= len(raw)
            buf[cursor:cursor + len(raw)] = np.frombuffer(raw, dtype=np.uint8)
            slots.append((cursor, len(raw)))
        free_end = cursor
        flags = FLAG_LEAF if self.is_leaf else 0
        hdr = _HDR.pack(self.page_id, flags, 0, len(self.cells), free_start, free_end,
                        self.right_sibling, self.left_child, self.lsn)
        buf[0:HEADER_SIZE] = np.frombuffer(hdr, dtype=np.uint8)
        for i, (off, ln) in enumerate(slots):
            buf[HEADER_SIZE + i * SLOT_SIZE: HEADER_SIZE + i * SLOT_SIZE + SLOT_SIZE] = \
                np.frombuffer(struct.pack("<HH", off, ln), dtype=np.uint8)
        crc = crc16(bytes(buf[:page_size - CRC_SIZE]))
        buf[page_size - CRC_SIZE:] = np.frombuffer(struct.pack("<H", crc), dtype=np.uint8)
        return bytes(buf)

    @staticmethod
    def decode(buf: bytes, page_size: int) -> "Page":
        if len(buf) != page_size:
            raise ValueError(f"page length {len(buf)} != {page_size}")
        want = struct.unpack_from("<H", buf, page_size - CRC_SIZE)[0]
        if crc16(buf[:page_size - CRC_SIZE]) != want:
            raise ValueError("crc mismatch (torn page)")
        page_id, flags, _r, slot_count, _fs, _fe, right, left_child, lsn = _HDR.unpack_from(buf, 0)
        page = Page(page_id=page_id, is_leaf=bool(flags & FLAG_LEAF),
                    right_sibling=right, left_child=left_child, lsn=lsn)
        for i in range(slot_count):
            off, ln = struct.unpack_from("<HH", buf, HEADER_SIZE + i * SLOT_SIZE)
            page.cells.append(Cell.decode(buf[off:off + ln], page.is_leaf))
        return page

    def segments(self, page_size: int) -> list[dict]:
        """Byte map of the encoded page for the review UI."""
        segs = [{"name": "header", "start": 0, "end": HEADER_SIZE,
                 "detail": f"page_id={self.page_id} leaf={self.is_leaf} lsn={self.lsn} "
                           f"right_sibling={self.right_sibling} left_child={self.left_child}"}]
        cursor = page_size - CRC_SIZE
        for i, cell in enumerate(self.cells):
            raw = cell.encode()
            cursor -= len(raw)
            segs.append({"name": f"cell[{i}]", "start": cursor, "end": cursor + len(raw),
                         "detail": f"key={cell.key!r} value={cell.value!r} child={cell.child}"})
        free_end = cursor
        free_start = HEADER_SIZE + SLOT_SIZE * len(self.cells)
        for i in range(len(self.cells)):
            segs.append({"name": f"slot[{i}]", "start": HEADER_SIZE + i * SLOT_SIZE,
                         "end": HEADER_SIZE + (i + 1) * SLOT_SIZE, "detail": "slot directory entry"})
        if free_end > free_start:
            segs.append({"name": "free", "start": free_start, "end": free_end,
                         "detail": f"free space {free_end - free_start} bytes"})
        segs.append({"name": "crc16", "start": page_size - CRC_SIZE, "end": page_size,
                     "detail": "torn-write detection"})
        segs.sort(key=lambda s: s["start"])
        return segs
