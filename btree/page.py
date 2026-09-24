"""Fixed-size page physical layout.

Frame (``PAGE_SIZE`` bytes, origin = page start)::

    [ common header 29B ][ slot directory -> ][ free space ][ <- cells ]

Header fields::

    kind u8, flags u16(ROOT=1), pid i32, parent i32, prev_sib i32, next_sib i32,
    nslots u16, cell_end u16, page_lsn i32, checksum u16

Leaf cells::  [klen u16][key bytes][vlen u32][value bytes][version u32]
Internal cells:: [child_pid i32][klen u16][key bytes]
Rightmost internal child is the trailing sentinel cell with klen=SENTINEL (0xFFFF).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator

KIND_FREE = 0
KIND_LEAF = 1
KIND_INTERNAL = 2

FLAG_ROOT = 0x0001

HEADER_FORMAT = ">BHiiiiHHIH"
HEADER_LEN = struct.calcsize(HEADER_FORMAT)  # 29
SENTINEL = 0xFFFF


@dataclass(frozen=True)
class Header:
    kind: int
    flags: int
    pid: int
    parent: int
    prev_sib: int
    next_sib: int
    nslots: int
    cell_end: int
    page_lsn: int
    checksum: int

    @property
    def is_root(self) -> bool:
        return bool(self.flags & FLAG_ROOT)


def checksum16(buf: bytes) -> int:
    """Additive 16-bit checksum over all words except the checksum field."""
    data = bytearray(buf)
    data[27:29] = b"\x00\x00"
    if len(data) % 2:
        data.append(0)
    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) | data[i + 1]
    while total > 0xFFFF:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def valid_checksum(page: bytes) -> bool:
    stored = struct.unpack_from(">H", page, 27)[0]
    return checksum16(page) == stored


def header(page: bytes) -> Header:
    (kind, flags, pid, parent, prev_sib, next_sib, nslots, cell_end,
     page_lsn, csum) = struct.unpack_from(HEADER_FORMAT, page, 0)
    return Header(kind, flags, pid, parent, prev_sib, next_sib, nslots,
                  cell_end, page_lsn, csum)


def slot_offsets(page: bytes) -> list[int]:
    h = header(page)
    return [struct.unpack_from(">H", page, HEADER_LEN + 2 * i)[0]
            for i in range(h.nslots)]


def cell_bytes(page: bytes) -> Iterator[tuple[int, bytes]]:
    h = header(page)
    for i, off in enumerate(slot_offsets(page)):
        ln = struct.unpack_from(">H", page, off)[0]
        # cells: for leaf klen then content; length varies, decode to find end
        if h.kind == KIND_LEAF:
            end = _leaf_cell_end(page, off)
        elif h.kind == KIND_INTERNAL:
            end = _int_cell_end(page, off)
        else:
            raise ValueError("cannot read cells of free page")
        yield i, page[off:end]


def _leaf_cell_end(page: bytes, off: int) -> int:
    pos = off + 2
    (klen,) = struct.unpack_from(">H", page, pos)
    pos += 2 + klen
    (vlen,) = struct.unpack_from(">I", page, pos)
    return pos + 4 + vlen + 4  # version


def _int_cell_end(page: bytes, off: int) -> int:
    pos = off + 4
    (klen,) = struct.unpack_from(">H", page, pos)
    if klen == SENTINEL:
        return pos + 2
    return pos + 2 + klen


def leaf_cell(key: bytes, value: bytes, version: int) -> bytes:
    return (struct.pack(">H", len(key)) + key
            + struct.pack(">I", len(value)) + value
            + struct.pack(">I", version))


def decode_leaf_cell(cell: bytes) -> dict:
    pos = 2
    (klen,) = struct.unpack_from(">H", cell, 0)
    key = cell[pos:pos + klen]
    pos += klen
    (vlen,) = struct.unpack_from(">I", cell, pos)
    pos += 4
    value = cell[pos:pos + vlen]
    version = struct.unpack_from(">I", cell[pos + vlen:pos + vlen + 4])[0]
    return {"key": key, "value": value, "version": version,
            "body_len": len(cell)}


def internal_cell(child: int, key: bytes | None) -> bytes:
    if key is None:  # rightmost sentinel
        return struct.pack(">iH", child, SENTINEL)
    return struct.pack(">iH", child, len(key)) + key


def decode_internal_cell(cell: bytes) -> dict:
    (child,) = struct.unpack_from(">i", cell, 0)
    (klen,) = struct.unpack_from(">H", cell, 4)
    if klen == SENTINEL:
        return {"child": child, "key": None, "body_len": len(cell)}
    return {"child": child, "key": cell[6:6 + klen],
            "body_len": len(cell)}


def leaf_key(cell: bytes) -> bytes:
    return decode_leaf_cell(cell)["key"]


def internal_key(cell: bytes) -> bytes | None:
    return decode_internal_cell(cell)["key"]


def frame(page_size: int, kind: int, pid: int, parent: int = -1,
          prev_sib: int = -1, next_sib: int = -1, root: bool = False,
          page_lsn: int = 0, cells: list[bytes] | None = None) -> bytes:
    """Pack a full page with cells right-aligned and slots left-aligned."""
    cells = cells or []
    cell_end = page_size
    offsets: list[int] = []
    for cell in cells:
        cell_end -= len(cell)
        offsets.append(cell_end)
    # slots are ordered key 0..n-1; cell 0 lives at the highest address
    blob = b"".join(cells[::-1])
    flags = FLAG_ROOT if root else 0
    head = struct.pack(HEADER_FORMAT, kind, flags, pid, parent, prev_sib,
                       next_sib, len(cells), cell_end, page_lsn, 0)
    slots = b"".join(struct.pack(">H", o) for o in offsets)
    body = bytearray(page_size)
    body[0:len(head)] = head
    pos = HEADER_LEN
    body[pos:pos + len(slots)] = slots
    pos += len(slots)
    free_end = cell_end
    body[free_end:free_end + len(blob)] = blob
    page = bytes(body)
    cs = checksum16(page)
    return page[:27] + struct.pack(">H", cs) + page[29:]


def free_bytes_of(page: bytes, page_size: int) -> int:
    """Free byte space: between the slot directory end and cell area start."""
    h = header(page)
    return h.cell_end - (HEADER_LEN + 2 * h.nslots)


def min_free_insert(page: bytes, cell_len: int, page_size: int) -> bool:
    """Whether ``cell_len`` bytes plus one slot still fit (byte-space decision)."""
    return free_bytes_of(page, page_size) >= cell_len + 2


def key_count(page: bytes) -> int:
    return header(page).nslots
