"""Fixed-size binary page layout.

Coordinate basis (all offsets are byte offsets from page start, unit: byte):

  [ header (HEADER_SIZE bytes) ][ free space ][ cells (growing downward) ]
                                                          ^
                                                  slot directory grows upward
                                              from the end of the page.

Header (16 bytes, struct '>HBBHHHBH' + 3 reserved):
  magic  uint16   0x4254 ("BT")
  kind   uint8    PAGE_META(0) / PAGE_LEAF(1) / PAGE_INTERNAL(2)
  flags  uint8    bit0 = used (allocated), bit1 = has_prev, bit2 = has_next
  nslots uint16
  free   uint16   free-space pointer (first byte of the free region)
  left   uint32   leaf: prev sibling; internal: leftmost child pid
  right  uint32   leaf: next sibling; meta: root pid; internal: reserved(0)
  crc    uint16   CRC-16/CCITT over all bytes except the crc field

Slot entry (4 bytes, '>H H'):
  cell_offset uint16  start offset of the cell inside the page
  extra       uint16  leaf: cell length in bytes; internal: child pid

Cell:
  leaf      = keylen(1) key(val bytes) vallen(1) value
  internal  = child(4) keylen(1) key
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

PAGE_SIZE = 128
HEADER_SIZE = 16
SLOT_SIZE = 4

PAGE_META = 0
PAGE_LEAF = 1
PAGE_INTERNAL = 2

META_PID = 0
INITIAL_ROOT_PID = 1

_MAGIC = 0x4254
_HEADER = struct.Struct(">HBBHHHBH")
assert _HEADER.size == HEADER_SIZE
_SLOT = struct.Struct(">HH")
_CRC_OFFSET = 14

FLAG_USED = 0x01
FLAG_HAS_PREV = 0x02
FLAG_HAS_NEXT = 0x04


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE. Unitless checksum used to detect torn pages."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


@dataclass
class Cell:
    key: bytes
    value: bytes = b""
    child: Optional[int] = None  # internal only: child pid for the key's right subtree

    @property
    def length(self) -> int:
        """On-disk cell length in bytes."""
        if self.child is None:
            return 1 + len(self.key) + 1 + len(self.value)
        return 4 + 1 + len(self.key)


@dataclass
class Page:
    pid: int
    kind: int = PAGE_LEAF
    used: bool = True
    cells: List[Cell] = field(default_factory=list)
    # 4-byte slot entries parallel to cells after a rebuild; populated by encode
    slot_entries: List[tuple] = field(default_factory=list)
    has_prev: bool = False
    has_next: bool = False
    left: int = 0
    right: int = 0
    # populated on decode/encode: where each cell starts (byte coordinate)
    cell_offsets: List[int] = field(default_factory=list)
    free_ptr: int = HEADER_SIZE
    checksum_ok: bool = True
    raw: Optional[bytes] = None

    def is_leaf(self) -> bool:
        return self.kind == PAGE_LEAF

    def is_internal(self) -> bool:
        return self.kind == PAGE_INTERNAL

    def free_space(self) -> int:
        """Usable bytes between the header and the slot directory / cells."""
        used_cells = sum(c.length for c in self.cells)
        return PAGE_SIZE - HEADER_SIZE - len(self.cells) * SLOT_SIZE - used_cells

    def find_leaf(self, key: bytes) -> int:
        """Index of the cell with the first key strictly greater than key."""
        import bisect

        keys = [c.key for c in self.cells]
        return bisect.bisect_right(keys, key)

    def find_internal(self, key: bytes) -> int:
        """Return child pid to follow for key."""
        import bisect

        keys = [c.key for c in self.cells]
        idx = bisect.bisect_right(keys, key)
        if idx == 0:
            return self.left
        return self.cells[idx - 1].child

    def get(self, key: bytes) -> Optional[bytes]:
        for c in self.cells:
            if c.key == key:
                return c.value
        return None


def raw_free_space(free_ptr: int, nslots: int) -> int:
    return PAGE_SIZE - free_ptr - nslots * SLOT_SIZE


def slot_directory_bytes(nslots: int) -> int:
    return nslots * SLOT_SIZE


def _encode_cell(cell: Cell) -> bytes:
    if cell.child is None:
        return bytes([len(cell.key)]) + cell.key + bytes([len(cell.value)]) + cell.value
    return struct.pack(">I", cell.child) + bytes([len(cell.key)]) + cell.key


def _decode_cell(kind: int, blob: bytes, off: int) -> tuple:
    if kind == PAGE_LEAF:
        klen = blob[off]
        key = blob[off + 1 : off + 1 + klen]
        p = off + 1 + klen
        vlen = blob[p]
        value = blob[p + 1 : p + 1 + vlen]
        return Cell(key=key, value=value), p + 1 + vlen
    child = struct.unpack(">I", blob[off : off + 4])[0]
    klen = blob[off + 4]
    key = blob[off + 5 : off + 5 + klen]
    return Cell(key=key, child=child), off + 5 + klen


def encode_page(page: Page) -> bytes:
    """Serialize a page, compacting cells against the slot directory."""
    n = len(page.cells)
    slots_len = n * SLOT_SIZE
    buf = bytearray(PAGE_SIZE)
    # cells are placed from the end, before the slot directory.
    cursor = PAGE_SIZE - slots_len
    offsets: List[int] = []
    slot_entries: List[tuple] = []
    # place cells in key order; slot index == cell index.
    ordered = sorted(range(n), key=lambda i: page.cells[i].key)
    placed = {}
    for new_idx, old_idx in enumerate(ordered):
        cell = page.cells[old_idx]
        enc = _encode_cell(cell)
        start = cursor - len(enc)
        buf[start : start + len(enc)] = enc
        extra = len(enc) if page.kind == PAGE_LEAF else (cell.child or 0)
        placed[old_idx] = (new_idx, start, extra)
        cursor = start
    free_ptr = cursor
    # rebuild cells/offsets in stored (sorted) order
    sorted_cells = [page.cells[i] for i in ordered]
    offsets = [placed[i][1] for i in ordered]
    slot_entries = [(placed[i][1], placed[i][2]) for i in ordered]
    slot_start = PAGE_SIZE - slots_len
    for idx, (off, extra) in enumerate(slot_entries):
        struct.pack_into(">HH", buf, slot_start + idx * SLOT_SIZE, off, extra)

    flags = FLAG_USED if page.used else 0
    if page.has_prev:
        flags |= FLAG_HAS_PREV
    if page.has_next:
        flags |= FLAG_HAS_NEXT
    right = page.right
    if page.kind == PAGE_META:
        right = page.right  # root pid stored in right
    _HEADER.pack_into(buf, 0, _MAGIC, page.kind, flags, n, free_ptr, page.left, right, 0)
    crc = crc16(bytes(buf[:_CRC_OFFSET]) + bytes(buf[_CRC_OFFSET + 2 :]))
    struct.pack_into(">H", buf, _CRC_OFFSET, crc)

    page.cells = sorted_cells
    page.cell_offsets = offsets
    page.slot_entries = slot_entries
    page.free_ptr = free_ptr
    page.checksum_ok = True
    raw = bytes(buf)
    page.raw = raw
    return raw


def decode_page(pid: int, raw: bytes) -> Page:
    """Parse a page. A CRC mismatch marks it as a torn half-written page."""
    if len(raw) != PAGE_SIZE:
        raise ValueError(f"page {pid}: expected {PAGE_SIZE} bytes, got {len(raw)}")
    magic, kind, flags, nslots, free_ptr, left, right, crc = _HEADER.unpack_from(raw, 0)
    checksum_ok = crc16(raw[:_CRC_OFFSET] + raw[_CRC_OFFSET + 2 :]) == crc
    page = Page(
        pid=pid,
        kind=kind if magic == _MAGIC else -1,
        used=bool(flags & FLAG_USED),
        has_prev=bool(flags & FLAG_HAS_PREV),
        has_next=bool(flags & FLAG_HAS_NEXT),
        left=left,
        right=right,
        free_ptr=free_ptr,
        checksum_ok=checksum_ok,
        raw=bytes(raw),
    )
    if not checksum_ok or magic != _MAGIC:
        return page
    slot_start = PAGE_SIZE - nslots * SLOT_SIZE
    cells: List[Cell] = []
    offsets: List[int] = []
    entries: List[tuple] = []
    for i in range(nslots):
        off, extra = _SLOT.unpack_from(raw, slot_start + i * SLOT_SIZE)
        entries.append((off, extra))
        if 0 <= off < PAGE_SIZE:
            cell, _ = _decode_cell(kind, raw, off)
            cells.append(cell)
            offsets.append(off)
    page.cells = cells
    page.cell_offsets = offsets
    page.slot_entries = entries
    return page


def meta_page(root_pid: int) -> Page:
    page = Page(pid=META_PID, kind=PAGE_META, used=True, right=root_pid)
    page.has_prev = page.has_next = False
    return page


def leaf_page(pid: int) -> Page:
    return Page(pid=pid, kind=PAGE_LEAF)


def internal_page(pid: int, leftmost: int) -> Page:
    page = Page(pid=pid, kind=PAGE_INTERNAL)
    page.left = leftmost
    return page
