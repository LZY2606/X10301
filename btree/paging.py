"""固定页大小的页内字节布局。

页结构（自顶向下）::

    +-------------------------+
    | page header (定长)       |
    +- - - - - - - - - - - - +|
    | slot directory (向前增长)|  每项 2 字节：单元格在页内的字节偏移
    |                         |
    |        free space       |
    |                         |
    | cells (向页尾方向增长)   |  key/value 都是变长
    +-------------------------+

free_space = cell_end - (HEADER_SIZE + 2 * nslots)，即槽目录末尾到首个单元格
之间连续可用的字节数。页分裂以 *字节空间* 为准，而不是键数。
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Iterable

MAGIC = b"PB"
PAGE_VERSION = 1

LEAF = 1
INTERNAL = 2

# magic, version, ptype, free, nslots, cell_end, lsn, next_leaf, first_child,
# checksum, flags
HEADER_FORMAT = ">2sH B HHH I ii I H"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

SLOT_FORMAT = ">H"
SLOT_SIZE = 2

# cell: key_len(H) + val_len(I) + first_child(i) + key + value
CELL_FORMAT = ">HI i"
CELL_HEAD = struct.calcsize(CELL_FORMAT)  # 2 + 4 + 4 = 10

FLAG_TORN = 0x1

MAX_KEY = 0xFFFF

_NO_CHILD = -1


# checksum 字段的拆分位置（前 9 个字段 | checksum | flags）
_PREFIX = ">2sH B HHH I ii "
_SUFFIX = "I H"
_PREFIX_F = _PREFIX.replace(" ", "")
_SUFFIX_F = _SUFFIX.replace(" ", "")


def crc(page: bytes) -> int:
    """对“checksum 字段置零”的整页字节计算 CRC32。"""
    masked = bytearray(page)
    masked[_CK_OFF : _CK_OFF + 4] = b"\x00\x00\x00\x00"
    return zlib.crc32(bytes(masked)) & 0xFFFFFFFF


# checksum 在页头中的字节区间
_CK_OFF = struct.calcsize(_PREFIX_F)


@dataclass(frozen=True)
class Cell:
    key: bytes
    value: bytes
    first_child: int = _NO_CHILD

    @property
    def byte_len(self) -> int:
        return CELL_HEAD + len(self.key) + len(self.value)


def encode_cell(cell: Cell) -> bytes:
    if len(cell.key) > MAX_KEY:
        raise ValueError(f"键过长：{len(cell.key)} > {MAX_KEY} 字节")
    return (
        struct.pack(CELL_FORMAT, len(cell.key), len(cell.value), cell.first_child)
        + cell.key
        + cell.value
    )


def _header_values(
    ptype: int,
    free: int,
    nslots: int,
    cell_end: int,
    lsn: int,
    next_leaf: int,
    first_child: int,
    checksum: int,
    flags: int,
) -> bytes:
    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        PAGE_VERSION,
        ptype,
        free,
        nslots,
        cell_end,
        lsn,
        next_leaf,
        first_child,
        checksum,
        flags,
    )


def build_page(
    page_size: int,
    ptype: int,
    cells: Iterable[Cell],
    *,
    lsn: int = 0,
    next_leaf: int = _NO_CHILD,
    first_child: int = _NO_CHILD,
    flags: int = 0,
) -> bytes:
    """按逻辑顺序（调用方保证已排序）重建成一整页。"""
    cells = list(cells)
    buf = bytearray(page_size)
    end = page_size
    offsets: list[int] = []
    for cell in cells:
        blob = encode_cell(cell)
        end -= len(blob)
        buf[end : end + len(blob)] = blob
        offsets.append(end)
    slots_end = HEADER_SIZE + SLOT_SIZE * len(cells)
    if end < slots_end:
        raise PageOverflow(page_size, cells, end, slots_end)
    # 槽目录按逻辑顺序排列，槽 i 指向第 i 个逻辑单元格（物理位置可乱序）
    for i, off in enumerate(offsets):
        base = HEADER_SIZE + i * SLOT_SIZE
        buf[base : base + SLOT_SIZE] = struct.pack(SLOT_FORMAT, off)
    body = bytes(buf[HEADER_SIZE:])
    # 先以 checksum=0 组装整页，再对整页计算并回填 CRC
    buf[0:HEADER_SIZE] = _header_values(
        ptype, end - slots_end, len(cells), end,
        lsn, next_leaf, first_child, 0, flags,
    )
    checksum = zlib.crc32(bytes(buf)) & 0xFFFFFFFF
    buf[0:HEADER_SIZE] = _header_values(
        ptype, end - slots_end, len(cells), end,
        lsn, next_leaf, first_child, checksum, flags,
    )
    return bytes(buf)


class PageOverflow(Exception):
    def __init__(self, page_size, cells, cell_end, slots_end):
        super().__init__(
            f"页溢出：单元格区起点 {cell_end} < 槽目录终点 {slots_end}"
            f"（页大小 {page_size} 字节，{len(cells)} 个单元格）"
        )
        self.page_size = page_size
        self.cells = cells


def empty_leaf(page_size: int, *, lsn: int = 0, next_leaf: int = _NO_CHILD) -> bytes:
    return build_page(page_size, LEAF, [], lsn=lsn, next_leaf=next_leaf)


def header(page: bytes) -> dict:
    (
        magic,
        version,
        ptype,
        free,
        nslots,
        cell_end,
        lsn,
        next_leaf,
        first_child,
        checksum,
        flags,
    ) = struct.unpack(HEADER_FORMAT, page[:HEADER_SIZE])
    return {
        "magic": magic,
        "version": version,
        "ptype": ptype,
        "free_space": free,
        "nslots": nslots,
        "cell_end": cell_end,
        "lsn": lsn,
        "next_leaf": next_leaf,
        "first_child": first_child,
        "checksum": checksum,
        "flags": flags,
    }


def page_ok(page: bytes, page_size: int) -> tuple[bool, str | None]:
    """结构与校验和是否完好（用于检测写页中断/torn page）。"""
    if len(page) != page_size:
        return False, f"页长度 {len(page)} != {page_size}"
    h = header(page)
    if h["magic"] != MAGIC:
        return False, "magic 不匹配"
    if h["version"] != PAGE_VERSION:
        return False, f"版本 {h['version']} 不受支持"
    want = crc(page)
    if want != h["checksum"]:
        return False, f"校验和损坏（读到 {h['checksum']:#010x}，应为 {want:#010x}）"
    if h["flags"] & FLAG_TORN:
        return False, "页头带有写中断标记（torn bit）"
    return True, None


def slots(page: bytes) -> list[int]:
    h = header(page)
    out = []
    for i in range(h["nslots"]):
        base = HEADER_SIZE + i * SLOT_SIZE
        (off,) = struct.unpack(SLOT_FORMAT, page[base : base + SLOT_SIZE])
        out.append(off)
    return out


def decode_cell(page: bytes, offset: int) -> Cell:
    klen, vlen, first_child = struct.unpack(
        CELL_FORMAT, page[offset : offset + CELL_HEAD]
    )
    k0 = offset + CELL_HEAD
    key = bytes(page[k0 : k0 + klen])
    v0 = k0 + klen
    value = bytes(page[v0 : v0 + vlen])
    return Cell(key, value, first_child)


def cells(page: bytes) -> list[Cell]:
    """按槽目录的逻辑顺序返回单元格。"""
    return [decode_cell(page, off) for off in slots(page)]


def logical_cells(page: bytes) -> list[Cell]:
    return cells(page)


def cell_needs(kl: int, vl: int) -> int:
    return CELL_HEAD + kl + vl + SLOT_SIZE


def can_fit(page: bytes, kl: int, vl: int) -> bool:
    h = header(page)
    return h["free_space"] >= cell_needs(kl, vl)


def insert_cell(page: bytes, cell: Cell, *, lsn: int) -> bytes:
    """在保持逻辑有序的前提下物理追加一个单元格（用于体现槽目录的意义）。"""
    h = header(page)
    page_size = len(page)
    old = cells(page)
    keys = [c.key for c in old]
    lo = bisect_left(keys, cell.key)
    if lo < len(old) and keys[lo] == cell.key:
        raise ValueError("insert_cell 遇到重复键，重复键策略应在调用层处理")
    new = old[:lo] + [cell] + old[lo:]
    return build_page(
        page_size,
        h["ptype"],
        new,
        lsn=lsn,
        next_leaf=h["next_leaf"],
        first_child=h["first_child"],
        flags=h["flags"] & ~FLAG_TORN,
    )


def replace_cells(
    page: bytes, new_cells: list[Cell], *, lsn: int, first_child: int | None = None
) -> bytes:
    h = header(page)
    return build_page(
        len(page),
        h["ptype"],
        new_cells,
        lsn=lsn,
        next_leaf=h["next_leaf"],
        first_child=h["first_child"] if first_child is None else first_child,
        flags=h["flags"] & ~FLAG_TORN,
    )


def set_meta(
    page: bytes,
    *,
    lsn: int | None = None,
    next_leaf: int | None = None,
    first_child: int | None = None,
) -> bytes:
    h = header(page)
    return build_page(
        len(page),
        h["ptype"],
        cells(page),
        lsn=h["lsn"] if lsn is None else lsn,
        next_leaf=h["next_leaf"] if next_leaf is None else next_leaf,
        first_child=h["first_child"] if first_child is None else first_child,
        flags=h["flags"] & ~FLAG_TORN,
    )


def bisect_left(keys: list[bytes], key: bytes) -> int:
    lo, hi = 0, len(keys)
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def tornify(page: bytes) -> bytes:
    """模拟写页中断：翻转页尾一个数据字节并置 torn bit，校验和必然失效。"""
    buf = bytearray(page)
    buf[-1] ^= 0xFF
    h = header(bytes(buf))
    torn = build_page(
        len(page),
        h["ptype"],
        cells(bytes(buf)),
        lsn=h["lsn"],
        next_leaf=h["next_leaf"],
        first_child=h["first_child"],
        flags=h["flags"] | FLAG_TORN,
    )
    # 重建会重新计算 CRC，因此再破坏一个数据字节制造“半页”
    t = bytearray(torn)
    t[-1] ^= 0xFF
    return bytes(t)


def render_key(key: bytes) -> str:
    try:
        return key.decode("utf-8")
    except UnicodeDecodeError:
        return key.hex()
