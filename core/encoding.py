"""固定页大小下的页内字节布局。

页布局（page_size 固定）::

    [header 19B][cell0][cell1]...[   free space   ][slot dir][crc16]

- cell 自头部后向高地址顺序打包（按键序，即槽序）。
- slot directory 自页尾向低地址增长，每项 uint16 cell 偏移，按键序排列。
- crc16(CRC-CCITT) 覆盖整页（crc 字段置零），用于识别写页中断造成的半页。
- free space 以字节为单位：page_size - (free_start + 2*ncells) - 2(crc)
"""
from __future__ import annotations

from dataclasses import dataclass

MAGIC = b"BTP1"
LEAF = 1
INTERNAL = 2
HEADER_SIZE = 19  # magic4 kind1 pid4 ncells2 free_start2 lsn4 crc? crc 在页尾
SLOT_SIZE = 2


class TornPageError(Exception):
    """页镜像 CRC 或结构不合法：写页中断留下的半页。"""


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class DecodedCell:
    key: bytes
    value: bytes
    child: int | None = None
    length: int = 0
    offset: int = 0


@dataclass(frozen=True)
class DecodedPage:
    pid: int
    kind: int
    lsn: int
    free_start: int
    cells: tuple[DecodedCell, ...]
    free_space: int


def leaf_cell_len(key: bytes, value: bytes) -> int:
    return 4 + len(key) + len(value)


def internal_cell_len(key: bytes, value: bytes) -> int:
    return 8 + len(key) + len(value)  # child4 + klen2 + vlen2


def encode_page(pid: int, kind: int, lsn: int, cells: list[tuple], page_size: int) -> bytes:
    """cells: leaf -> [(key,value)]; internal -> [(child,key,value)]，必须按 key 升序。"""
    page = bytearray(page_size)
    page[0:4] = MAGIC
    page[4] = kind
    page[5:9] = pid.to_bytes(4, "big")
    ncells = len(cells)
    page[9:11] = ncells.to_bytes(2, "big")

    offset = HEADER_SIZE
    slots: list[int] = []
    for cell in cells:
        slots.append(offset)
        if kind == LEAF:
            key, value = cell
            klen, vlen = len(key), len(value)
            body = klen.to_bytes(2, "big") + vlen.to_bytes(2, "big") + key + value
        else:
            child, key, value = cell
            body = (
                child.to_bytes(4, "big")
                + len(key).to_bytes(2, "big")
                + len(value).to_bytes(2, "big")
                + key
                + value
            )
        page[offset : offset + len(body)] = body
        offset += len(body)

    page[11:13] = offset.to_bytes(2, "big")
    page[13:17] = (lsn & 0xFFFFFFFF).to_bytes(4, "big")

    dir_start = page_size - 2 - SLOT_SIZE * ncells
    for i, cell_offset in enumerate(slots):
        pos = dir_start + i * SLOT_SIZE
        page[pos : pos + 2] = cell_offset.to_bytes(2, "big")

    crc = crc16_ccitt(bytes(page))
    page[page_size - 2 : page_size] = crc.to_bytes(2, "big")
    return bytes(page)


def decode_page(blob: bytes) -> DecodedPage:
    page_size = len(blob)
    if len(blob) < HEADER_SIZE or blob[0:4] != MAGIC:
        raise TornPageError("bad magic/header")
    stored_crc = int.from_bytes(blob[page_size - 2 : page_size], "big")
    if crc16_ccitt(blob[: page_size - 2] + b"\x00\x00") != stored_crc:
        raise TornPageError("crc mismatch: torn page")
    kind = blob[4]
    if kind not in (LEAF, INTERNAL):
        raise TornPageError("bad page kind")
    pid = int.from_bytes(blob[5:9], "big")
    ncells = int.from_bytes(blob[9:11], "big")
    free_start = int.from_bytes(blob[11:13], "big")
    lsn = int.from_bytes(blob[13:17], "big")
    dir_start = page_size - 2 - SLOT_SIZE * ncells
    if dir_start < free_start:
        raise TornPageError("slot directory overlaps cells")

    cells: list[DecodedCell] = []
    for i in range(ncells):
        cell_off = int.from_bytes(blob[dir_start + i * 2 : dir_start + i * 2 + 2], "big")
        end = (
            int.from_bytes(blob[dir_start + (i + 1) * 2 : dir_start + (i + 1) * 2 + 2], "big")
            if i + 1 < ncells
            else free_start
        )
        if not (HEADER_SIZE <= cell_off < end <= free_start):
            raise TornPageError(f"bad slot offset at {i}")
        if kind == LEAF:
            klen = int.from_bytes(blob[cell_off : cell_off + 2], "big")
            vlen = int.from_bytes(blob[cell_off + 2 : cell_off + 4], "big")
            key = blob[cell_off + 4 : cell_off + 4 + klen]
            value = blob[cell_off + 4 + klen : cell_off + 4 + klen + vlen]
            length = leaf_cell_len(key, value)
            cells.append(DecodedCell(key, value, None, length, cell_off))
        else:
            child = int.from_bytes(blob[cell_off : cell_off + 4], "big")
            klen = int.from_bytes(blob[cell_off + 4 : cell_off + 6], "big")
            vlen = int.from_bytes(blob[cell_off + 6 : cell_off + 8], "big")
            key = blob[cell_off + 8 : cell_off + 8 + klen]
            value = blob[cell_off + 8 + klen : cell_off + 8 + klen + vlen]
            length = internal_cell_len(key, value)
            cells.append(DecodedCell(key, value, child, length, cell_off))
        if cell_off + cells[-1].length != end:
            raise TornPageError(f"cell {i} length mismatch")

    free_space = page_size - 2 - SLOT_SIZE * ncells - free_start
    return DecodedPage(pid=pid, kind=kind, lsn=lsn, free_start=free_start,
                       cells=tuple(cells), free_space=free_space)
