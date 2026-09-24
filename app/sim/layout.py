"""B-tree page byte layout.

All domain quantities carry a unit or coordinate basis:
  - offsets: bytes, absolute from page start (offset 0 = first byte of the page)
  - lengths: bytes
  - key length: bytes of the UTF-8 encoded key (raw text is preserved in the
    raw material layer; the derived layer normalizes to UTF-8 bytes)

Page layout (little-endian), page_size bytes per page::

    offset  size  field
    0       1     page_type (1=meta, 2=leaf, 3=internal)
    1       4     page_id (uint32)
    5       4     left_sibling (uint32, 0 = none)
    9       4     right_sibling (uint32, 0 = none)
    13      4     page_lsn (uint32)
    17      4     cell_count (uint32)
    21      4     first_child (uint32, internal pages only, else 0)
    25      4     flags (uint32; bit0 = freed/empty page)
    29      4     checksum crc32 of bytes [0:29] + [33:page_size]
    33      7     reserved
    40      ..    slot directory: cell_count * uint16 absolute offsets,
                  growing upward from offset 40
    ..      ..    cells, packed downward from page_size

free_space_bytes = cell_area_start - (40 + 2 * cell_count)
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

PAGE_META = 1
PAGE_LEAF = 2
PAGE_INTERNAL = 3

HEADER_SIZE = 40
CHECKSUM_OFFSET = 29
SLOT_OFFSET = HEADER_SIZE
META_BODY_OFFSET = HEADER_SIZE  # meta page body: root(4) + next_page_id(4)

TYPE_NAMES = {PAGE_META: "meta", PAGE_LEAF: "leaf", PAGE_INTERNAL: "internal"}


@dataclass
class Cell:
    key: bytes
    value: bytes = b""        # leaf cells
    child: int = 0            # internal cells: child to the RIGHT of key
    sep: bytes = b""          # internal cells under "copy" policy: original key


@dataclass
class Page:
    page_id: int
    page_type: int
    page_lsn: int = 0
    left_sibling: int = 0
    right_sibling: int = 0
    first_child: int = 0
    flags: int = 0
    cells: list = field(default_factory=list)  # list[Cell]
    root_id: int = 0          # meta pages only
    next_page_id: int = 0     # meta pages only

    # ---- byte accounting (all in bytes) ----
    def cell_bytes(self, cell: Cell) -> int:
        if self.page_type == PAGE_LEAF:
            return 4 + len(cell.key) + len(cell.value)
        if self.page_type == PAGE_INTERNAL:
            return 8 + len(cell.key) + (len(cell.sep) if cell.sep else 0)
        return 0

    def used_bytes(self) -> int:
        body = sum(self.cell_bytes(c) for c in self.cells)
        return HEADER_SIZE + 2 * len(self.cells) + body

    def free_space_bytes(self, page_size: int) -> int:
        return page_size - self.used_bytes()

    def fits(self, cell: Cell, page_size: int) -> bool:
        return self.used_bytes() + 2 + self.cell_bytes(cell) <= page_size


def encode_page(page: Page, page_size: int) -> bytes:
    buf = bytearray(page_size)
    struct.pack_into("<BIIIIIII", buf, 0,
                     page.page_type, page.page_id, page.left_sibling,
                     page.right_sibling, page.page_lsn, len(page.cells),
                     page.first_child, page.flags)
    cell_area = page_size
    slots = []
    for cell in page.cells:
        if page.page_type == PAGE_LEAF:
            blob = struct.pack("<HH", len(cell.key), len(cell.value)) + cell.key + cell.value
        elif page.page_type == PAGE_INTERNAL:
            sep = cell.sep or b""
            blob = struct.pack("<IHH", cell.child, len(cell.key), len(sep)) + cell.key + sep
        else:
            blob = b""
        cell_area -= len(blob)
        if cell_area < SLOT_OFFSET + 2 * len(page.cells):
            raise ValueError(f"page {page.page_id} overflow during encode")
        buf[cell_area:cell_area + len(blob)] = blob
        slots.append(cell_area)
    for i, off in enumerate(slots):
        struct.pack_into("<H", buf, SLOT_OFFSET + 2 * i, off)
    if page.page_type == PAGE_META:
        struct.pack_into("<II", buf, META_BODY_OFFSET, page.root_id, page.next_page_id)
    crc = zlib.crc32(bytes(buf[0:CHECKSUM_OFFSET]) + bytes(buf[CHECKSUM_OFFSET + 4:page_size]))
    struct.pack_into("<I", buf, CHECKSUM_OFFSET, crc)
    return bytes(buf)


def decode_page(data: bytes, page_size: int) -> Page:
    if len(data) != page_size:
        raise ValueError(f"page image must be {page_size} bytes, got {len(data)}")
    stored_crc = struct.unpack_from("<I", data, CHECKSUM_OFFSET)[0]
    crc = zlib.crc32(bytes(data[0:CHECKSUM_OFFSET]) + bytes(data[CHECKSUM_OFFSET + 4:page_size]))
    if stored_crc != crc:
        raise ValueError("checksum mismatch (torn or corrupt page)")
    (ptype, pid, left, right, lsn, ncells, first_child, flags) = struct.unpack_from("<BIIIIIII", data, 0)
    page = Page(page_id=pid, page_type=ptype, page_lsn=lsn, left_sibling=left,
                right_sibling=right, first_child=first_child, flags=flags)
    if ptype == PAGE_META:
        page.root_id, page.next_page_id = struct.unpack_from("<II", data, META_BODY_OFFSET)
        return page
    for i in range(ncells):
        off = struct.unpack_from("<H", data, SLOT_OFFSET + 2 * i)[0]
        if ptype == PAGE_LEAF:
            klen, vlen = struct.unpack_from("<HH", data, off)
            key = bytes(data[off + 4: off + 4 + klen])
            val = bytes(data[off + 4 + klen: off + 4 + klen + vlen])
            page.cells.append(Cell(key=key, value=val))
        elif ptype == PAGE_INTERNAL:
            child, klen, slen = struct.unpack_from("<IHH", data, off)
            key = bytes(data[off + 8: off + 8 + klen])
            sep = bytes(data[off + 8 + klen: off + 8 + klen + slen])
            page.cells.append(Cell(key=key, child=child, sep=sep))
    return page


def describe_page(image: bytes, page_size: int) -> dict:
    """Decode a page image into a display structure with byte coordinates.

    Every span carries {offset, length} in bytes, absolute from page start.
    """
    page = decode_page(image, page_size)
    spans = [{"field": "header", "offset": 0, "length": HEADER_SIZE, "unit": "byte"}]
    n = len(page.cells)
    spans.append({"field": "slot_directory", "offset": SLOT_OFFSET,
                  "length": 2 * n, "unit": "byte"})
    slots = []
    cursor = page_size
    cells_desc = []
    for i, cell in enumerate(page.cells):
        size = page.cell_bytes(cell)
        cursor -= size
        slot_off = struct.unpack_from("<H", image, SLOT_OFFSET + 2 * i)[0]
        assert slot_off == cursor, "slot offset must match packed cell offset"
        slots.append({"slot_index": i, "offset": slot_off, "unit": "byte"})
        cells_desc.append({
            "slot_index": i,
            "offset": cursor,
            "length": size,
            "unit": "byte",
            "key_hex": cell.key.hex(),
            "key_text": cell.key.decode("utf-8", "replace"),
            "value_hex": cell.value.hex() if page.page_type == PAGE_LEAF else None,
            "child": cell.child if page.page_type == PAGE_INTERNAL else None,
            "sep_hex": cell.sep.hex() if (page.page_type == PAGE_INTERNAL and cell.sep) else None,
        })
        spans.append({"field": f"cell[{i}]", "offset": cursor, "length": size, "unit": "byte"})
    free_off = SLOT_OFFSET + 2 * n
    free_len = cursor - free_off
    spans.append({"field": "free_space", "offset": free_off, "length": free_len, "unit": "byte"})
    return {
        "page_id": page.page_id,
        "page_type": TYPE_NAMES.get(page.page_type, str(page.page_type)),
        "page_lsn": page.page_lsn,
        "left_sibling": page.left_sibling,
        "right_sibling": page.right_sibling,
        "first_child": page.first_child,
        "flags": page.flags,
        "cell_count": n,
        "used_bytes": page.used_bytes(),
        "free_space_bytes": free_len,
        "slots": slots,
        "cells": cells_desc,
        "spans": spans,
        "coordinate_basis": "byte offset from page start (offset 0 = first byte)",
    }


def free_space_map(images: dict, page_size: int) -> np.ndarray:
    """Vectorized free-space (bytes) per page id, for the layout panel."""
    ids = sorted(images)
    arr = np.zeros(len(ids), dtype=np.int64)
    for i, pid in enumerate(ids):
        try:
            page = decode_page(images[pid], page_size)
            arr[i] = page.free_space_bytes(page_size)
        except ValueError:
            arr[i] = -1  # torn/corrupt page
    return arr
