"""固定页大小的页内布局：header + slot directory + 保留区 + cell area。

字节坐标基准（basis=pageN）：页首偏移 0；槽位目录紧随页头向后生长；
internal 页在目录后固定保留 4 字节存放 rightmost child；cell 区自页尾
向前打包；中间为 free space。

页头（33 字节）：
    magic u32 | page_id u32 | type u8 | free_start u16 | ncell u16
    | next u32 | parent u32 | page_lsn u64 | checksum u32
checksum 覆盖除 checksum 字段外的整页；写页中断（截断/半页）必然导致
长度、magic 或 checksum 失败 -> TornPageError。
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAGIC = 0x50_544731  # "PTG1"
LEAF = 1
INTERNAL = 2
HEADER_SIZE = 33
SLOT_SIZE = 2
RIGHTMOST_SIZE = 4
NONE = 0xFFFF_FFFF

_HEADER = struct.Struct("<IIBHHIIQ")
assert _HEADER.size == 29


class TornPageError(Exception):
    """页镜像不完整：长度不足、magic 不符或 checksum 不匹配（写页中断）。"""


@dataclass
class LeafCell:
    key: bytes
    value: bytes

    def size(self) -> int:
        return 4 + len(self.key) + len(self.value)


@dataclass
class InternalCell:
    key: bytes  # separator（按构造规则复制或提升而来）
    left_child: int

    def size(self) -> int:
        return 6 + len(self.key)


@dataclass
class Page:
    page_id: int
    kind: int
    cells: list[Any] = field(default_factory=list)  # LeafCell | InternalCell，槽位升序
    rightmost: int = NONE  # 仅 internal：>= 最后一个 separator 的子页
    next: int = NONE       # leaf sibling link
    parent: int = NONE
    page_lsn: int = 0
    page_size: int = 128

    def reserved(self) -> int:
        return RIGHTMOST_SIZE if self.kind == INTERNAL else 0

    def slot_dir_end(self) -> int:
        return HEADER_SIZE + SLOT_SIZE * len(self.cells) + self.reserved()

    def used_cell_bytes(self) -> int:
        arr = np.asarray([c.size() for c in self.cells], dtype=np.int64)
        return int(arr.sum()) if len(arr) else 0

    def free_bytes(self) -> int:
        used = HEADER_SIZE + SLOT_SIZE * len(self.cells) + self.reserved() + self.used_cell_bytes()
        return self.page_size - used

    def free_start(self) -> int:
        return self.page_size - self.used_cell_bytes()

    def fits_extra(self, extra_cell_size: int) -> bool:
        return self.free_bytes() >= (extra_cell_size + SLOT_SIZE)

    def serialize(self) -> bytes:
        n = len(self.cells)
        free_start = self.free_start()
        buf = bytearray(self.page_size)
        buf[:29] = _HEADER.pack(
            MAGIC, self.page_id, self.kind, free_start, n,
            self.next, self.parent, self.page_lsn,
        )
        if self.kind == INTERNAL:
            struct.pack_into("<I", buf, HEADER_SIZE + 2 * n, self.rightmost)
        cursor = self.page_size
        for idx, cell in enumerate(self.cells):
            if self.kind == LEAF:
                blob = struct.pack("<HH", len(cell.key), len(cell.value)) + cell.key + cell.value
            else:
                blob = struct.pack("<HI", len(cell.key), cell.left_child) + cell.key
            cursor -= len(blob)
            buf[cursor:cursor + len(blob)] = blob
            struct.pack_into("<H", buf, HEADER_SIZE + 2 * idx, cursor)
        chk = zlib.crc32(bytes(buf[:29]) + bytes(buf[33:])) & 0xFFFF_FFFF
        struct.pack_into("<I", buf, 29, chk)
        return bytes(buf)


def parse_page(data: bytes, expected_size: int) -> Page:
    """从（可能被写页中断截断的）字节镜像解析页。"""
    if len(data) != expected_size:
        raise TornPageError(f"页长度 {len(data)} != 页大小 {expected_size}（疑似写页中断）")
    magic, pid, kind, free_start, n, nxt, parent, lsn = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise TornPageError(f"magic=0x{magic:08x} 非法（疑似半页）")
    chk = struct.unpack_from("<I", data, 29)[0]
    expect = zlib.crc32(bytes(data[:29]) + bytes(data[33:])) & 0xFFFF_FFFF
    if chk != expect:
        raise TornPageError(f"checksum 不匹配 {chk:08x}!={expect:08x}（疑似半页）")
    page = Page(page_id=pid, kind=kind, page_size=expected_size,
                next=nxt, parent=parent, page_lsn=lsn)
    for i in range(n):
        off = struct.unpack_from("<H", data, HEADER_SIZE + 2 * i)[0]
        if kind == LEAF:
            klen, vlen = struct.unpack_from("<HH", data, off)
            key = data[off + 4:off + 4 + klen]
            value = data[off + 4 + klen:off + 4 + klen + vlen]
            page.cells.append(LeafCell(key, value))
        else:
            klen, left = struct.unpack_from("<HI", data, off)
            key = data[off + 6:off + 6 + klen]
            page.cells.append(InternalCell(key, left))
    if kind == INTERNAL:
        page.rightmost = struct.unpack_from("<I", data, HEADER_SIZE + 2 * n)[0]
    return page


def max_key_fits(page_size: int, value_len: int = 1) -> int:
    """单 cell 最大键长（超过则任何分裂都放不下 -> 诊断 OVERSIZED）。"""
    return page_size - HEADER_SIZE - SLOT_SIZE - 4 - value_len


# ---- 按字节空间决定分裂点（需求推导规则 1）----
def byte_split_candidates(sizes: list[int], page_size: int) -> list[int]:
    """返回所有合法分裂点 i：左 [0..i-1] / 右 [i..]，两侧各自能容纳。

    判定依据是 *字节空间* 而非 cell 个数；任一侧放不下即非法。
    """
    sizes = list(sizes)
    n = len(sizes)
    prefix = np.concatenate([[0], np.cumsum(np.asarray(sizes, dtype=np.int64))])
    total = int(prefix[-1])
    candidates: list[int] = []
    for i in range(1, n):
        left_bytes = int(prefix[i])
        right_bytes = total - left_bytes
        left_used = HEADER_SIZE + SLOT_SIZE * i + left_bytes
        right_used = HEADER_SIZE + SLOT_SIZE * (n - i) + right_bytes
        if left_used <= page_size and right_used <= page_size:
            candidates.append(i)
    return candidates


def byte_balanced(sizes: list[int], candidates: list[int]) -> tuple[int | None, list[int]]:
    """从合法候选中选 |left_bytes-right_bytes| 最小者，返回 (chosen, tied)。"""
    if not candidates:
        return None, []
    total = sum(sizes)
    prefix = 0
    loads: list[tuple[int, int]] = []
    for idx, s in enumerate(sizes):
        prefix += s
        if idx + 1 in candidates:
            loads.append((idx + 1, abs(2 * prefix - total)))
    best = min(diff for _, diff in loads)
    tied = [i for i, diff in loads if diff == best]
    chosen = tied[0]
    return chosen, tied
