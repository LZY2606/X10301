"""Fixed-size page byte layout with a slot directory.

Page image (``page_size`` bytes, power of two, e.g. 128/4096)::

    header (32B) | slot directory (2B * ncells, grows toward high addr)
                  | free space | packed cells (grow toward low addr)

Header fields (little-endian):

    0  u8   page type        (1 = leaf, 2 = internal)
    1  u8   flags            (bit0 = torn)
    2  u16  cell count
    4  u16  free_start       (lowest offset used by packed cells)
    8  u64  page_lsn
    16 u32  hi_child         (internal only; 0xFFFFFFFF for leaves)
    20 u32  page_id
    28 u32  checksum (FNV-1a over header[0:28] + slot dir + cell area)

Leaf cell:  u16 klen | key bytes | u16 vlen | value bytes
Internal:   u32 child | u16 klen | separator bytes   (rightmost child in header)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

LEAF = 1
INTERNAL = 2
NONE_PID = 0xFFFFFFFF
TORN_FLAG = 0x01
HEADER_SIZE = 32
_SLOT = struct.Struct("<H")
_CRC = struct.Struct("<I")
_MASK = 0xFFFFFFFF
_OFFSET_BITS = 0xFFFF


class PageFormatError(ValueError):
    pass


class PageChecksumError(PageFormatError):
    def __init__(self, page_id: int, expected: int, actual: int):
        super().__init__(
            f"page {page_id}: checksum mismatch "
            f"(stored {expected:#010x}, computed {actual:#010x})"
        )
        self.page_id = page_id
        self.expected = expected
        self.actual = actual


@dataclass
class Cell:
    key: bytes
    value: Optional[bytes] = None
    child: Optional[int] = None

    def leaf_size(self) -> int:
        return 2 + len(self.key) + 2 + len(self.value or b"")

    def internal_size(self) -> int:
        return 4 + 2 + len(self.key)


@dataclass
class BasePage:
    page_id: int
    cells: List[Cell] = field(default_factory=list)
    page_lsn: int = 0
    flags: int = 0

    @property
    def ncells(self) -> int:
        return len(self.cells)

    def slot_bytes(self) -> int:
        return 2 * len(self.cells)


@dataclass
class LeafPage(BasePage):
    hi_child: int = NONE_PID
    kind: int = LEAF


@dataclass
class InternalPage(BasePage):
    hi_child: int = NONE_PID
    kind: int = INTERNAL


def cell_size(cell: Cell, page_kind: int) -> int:
    return cell.leaf_size() if page_kind == LEAF else cell.internal_size()


def cells_payload_bytes(cells: List[Cell], page_kind: int) -> int:
    return sum(cell_size(c, page_kind) for c in cells)


def page_used_bytes(cells: List[Cell], page_kind: int) -> int:
    """Header + slot directory + packed cells."""
    return HEADER_SIZE + 2 * len(cells) + cells_payload_bytes(cells, page_kind)


def page_usable_bytes(page_size: int) -> int:
    return page_size - HEADER_SIZE


def page_free_bytes(page_size: int, cells: List[Cell], page_kind: int) -> int:
    free_start = page_size
    for c in cells:
        free_start -= cell_size(c, page_kind)
    return free_start - (HEADER_SIZE + 2 * len(cells))


def fnv1a(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & _MASK
    return h


def _encode_kind(page: BasePage, page_size: int, kind: int, torn: bool) -> bytes:
    n = len(page.cells)
    free_start = page_size
    for c in page.cells:
        free_start -= cell_size(c, kind)
    if free_start < HEADER_SIZE + 2 * n:
        raise PageFormatError(f"page {page.page_id} overflows {page_size} bytes")

    buf = bytearray(page_size)
    hi = getattr(page, "hi_child", NONE_PID)
    flags = page.flags | (TORN_FLAG if torn else 0)
    struct.pack_into(
        "<BBHH4sQII4s",
        buf,
        0,
        kind,
        flags,
        n,
        free_start,
        b"\x00\x00\x00\x00",
        page.page_lsn,
        hi & _MASK,
        page.page_id & _MASK,
        b"\x00\x00\x00\x00",
    )

    cursor = free_start
    for c in page.cells:
        if kind == LEAF:
            payload = struct.pack("<H", len(c.key)) + c.key
            payload += struct.pack("<H", len(c.value or b"")) + (c.value or b"")
        else:
            if c.child is None:
                raise PageFormatError("internal cell missing child pointer")
            payload = struct.pack("<I", c.child) + struct.pack("<H", len(c.key)) + c.key
        buf[cursor : cursor + len(payload)] = payload
        cursor += len(payload)

    for i in range(n):
        # slot directory offsets are filled after the body; physical order
        # matches logical key order because cells are appended in order.
        pass

    physical = free_start
    for i, c in enumerate(page.cells):
        struct.pack_into("<H", buf, HEADER_SIZE + 2 * i, physical)
        physical += cell_size(c, kind)

    checksum = fnv1a(bytes(buf[: HEADER_SIZE - 4]) + bytes(buf[HEADER_SIZE:page_size]))
    struct.pack_into("<I", buf, HEADER_SIZE - 4, checksum)
    return bytes(buf)


def encode_page(page: BasePage, page_size: int, torn: bool = False) -> bytes:
    kind = getattr(page, "kind", LEAF if isinstance(page, LeafPage) else INTERNAL)
    return _encode_kind(page, page_size, kind, torn)


def page_checksum(buf: bytes) -> tuple[int, int]:
    (stored,) = _CRC.unpack_from(buf, HEADER_SIZE - 4)
    computed = fnv1a(bytes(buf[: HEADER_SIZE - 4]) + bytes(buf[HEADER_SIZE:]))
    return stored, computed


def decode_page(buf: bytes, verify: bool = True) -> BasePage:
    kind, flags, n, free_start = struct.unpack_from("<BBHH", buf, 0)
    (_, _, _, _, _, page_lsn, hi_child, page_id, _) = struct.unpack_from(
        "<BBHH4sQII4s", buf, 0
    )
    stored, computed = page_checksum(buf)
    torn = bool(flags & TORN_FLAG) or stored != computed
    if verify and stored != computed:
        raise PageChecksumError(page_id, stored, computed)
    cells: List[Cell] = []
    for i in range(n):
        (off,) = _SLOT.unpack_from(buf, HEADER_SIZE + 2 * i)
        if off < HEADER_SIZE or off > _OFFSET_BITS:
            raise PageFormatError(f"page {page_id}: bad slot offset {off}")
        if kind == LEAF:
            (klen,) = struct.unpack_from("<H", buf, off)
            key = bytes(buf[off + 2 : off + 2 + klen])
            (vlen,) = struct.unpack_from("<H", buf, off + 2 + klen)
            value = bytes(buf[off + 4 + klen : off + 4 + klen + vlen])
            cells.append(Cell(key=key, value=value))
        else:
            (child,) = struct.unpack_from("<I", buf, off)
            (klen,) = struct.unpack_from("<H", buf, off + 4)
            key = bytes(buf[off + 6 : off + 6 + klen])
            cells.append(Cell(key=key, child=child))
    if kind == INTERNAL:
        page = InternalPage(page_id=page_id, cells=cells, page_lsn=page_lsn,
                            flags=flags, hi_child=hi_child)
    else:
        page = LeafPage(page_id=page_id, cells=cells, page_lsn=page_lsn, flags=flags)
    page.flags |= TORN_FLAG if torn and not (flags & TORN_FLAG) else 0
    return page


def legal_split_points(cells: List[Cell], page_size: int, kind: int) -> List[int]:
    """All byte-space-feasible split points.

    A point ``i`` (left gets ``cells[:i]``) is legal when *both* halves fit
    independently in a fresh page. Feasibility is decided by packed byte
    space, never by the number of keys -- a variable-length key that is much
    larger than its neighbours can force an off-center cut.
    """
    points: List[int] = []
    total = len(cells)
    for i in range(1, total):
        left_ok = page_free_bytes(page_size, cells[:i], kind) >= 0
        right_ok = page_free_bytes(page_size, cells[i:], kind) >= 0
        if left_ok and right_ok:
            points.append(i)
    return points


def best_split_point(cells: List[Cell], page_size: int, kind: int) -> Optional[int]:
    """Deterministic choice: minimize absolute byte imbalance of the halves."""
    candidates = legal_split_points(cells, page_size, kind)
    if not candidates:
        return None

    def imbalance(i: int) -> int:
        left = cells_payload_bytes(cells[:i], kind)
        right = cells_payload_bytes(cells[i:], kind)
        return abs(left - right)

    return min(candidates, key=lambda i: (imbalance(i), i))
