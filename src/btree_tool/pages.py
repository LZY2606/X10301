"""固定页大小的 B-tree 页内字节布局。

页布局（坐标基准：page_start，单位 byte）::

    [0:2]   magic 0xB7EE
    [2:4]   page type   1=leaf 2=internal
    [4:6]   ncells
    [6:10]  page LSN（日志序号）
    [10:14] free_start（slot directory 之后）
    [14:18] free_end（最低 cell 的起点）
    [18:20] high child（internal 最右子页；leaf 为 0xFFFF）
    [20:22] prev sibling（leaf 左兄弟）
    [22:24] next sibling（leaf 右兄弟）
    [24 .. free_start)   slot directory，每项 2 byte，按键序给出 cell 偏移
    [free_end .. page_size) cell 区，自页尾向前生长
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable

MAGIC = 0xB7EE
LEAF = 1
INTERNAL = 2
HEADER_SIZE = 24
SLOT_SIZE = 2
SENTINEL = 0xFFFF

_HEADER = struct.Struct(">HHHIIIHHH")


class PageFormatError(Exception):
    pass


@dataclass
class Cell:
    """页内一个 cell。

    leaf:     key/value 有值，child 为 None
    internal: key 是 separator，child 是该 separator 左侧子页 id，value 为 None
    """

    key: bytes
    value: bytes | None = None
    child: int | None = None

    def encoded_len(self) -> int:
        if self.value is not None:
            return 4 + len(self.key) + len(self.value)
        return 4 + len(self.key)

    def kind(self) -> str:
        return "leaf" if self.value is not None else "internal"


@dataclass
class Page:
    pid: int
    ptype: int = LEAF
    cells: list[Cell] = field(default_factory=list)
    high: int | None = None          # internal 最右子页
    prev: int | None = None          # leaf 左兄弟
    next: int | None = None          # leaf 右兄弟
    lsn: int = 0

    # ---- 基础属性 -----------------------------------------------------
    @property
    def is_leaf(self) -> bool:
        return self.ptype == LEAF

    def slot_dir_len(self) -> int:
        return SLOT_SIZE * len(self.cells)

    def cells_byte_len(self) -> int:
        return sum(c.encoded_len() for c in self.cells)

    def used_bytes(self) -> int:
        return HEADER_SIZE + self.slot_dir_len() + self.cells_byte_len()

    def free_space(self, page_size: int) -> int:
        """free space = free_end - free_start（字节，坐标基准 page_start）。"""
        free_start = HEADER_SIZE + self.slot_dir_len()
        free_end = page_size - self.cells_byte_len()
        return free_end - free_start

    def fits(self, page_size: int) -> bool:
        return self.used_bytes() <= page_size

    # ---- 变异（返回所需字节，供引擎做容量判定） ------------------------
    def insertion_need(self, cell: Cell) -> int:
        return SLOT_SIZE + cell.encoded_len()

    def can_insert(self, cell: Cell, page_size: int) -> bool:
        return self.free_space(page_size) >= self.insertion_need(cell)

    def put_leaf(self, key: bytes, value: bytes, *, overwrite: bool) -> tuple[bool, int]:
        """按键序写入 leaf cell。返回 (是否新增, 槽位下标)。"""
        i = self._slot_for(key)
        if i < len(self.cells) and self.cells[i].key == key:
            if not overwrite:
                return False, i
            self.cells[i] = Cell(key=key, value=value)
            return False, i
        self.cells.insert(i, Cell(key=key, value=value))
        return True, i

    def remove_leaf(self, key: bytes) -> bool:
        i = self._slot_for(key)
        if i < len(self.cells) and self.cells[i].key == key:
            del self.cells[i]
            return True
        return False

    def _slot_for(self, key: bytes) -> int:
        lo, hi = 0, len(self.cells)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cells[mid].key < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # ---- internal 路由 -------------------------------------------------
    def child_for(self, key: bytes) -> int:
        """返回 key 应进入的子页 id。"""
        assert self.ptype == INTERNAL
        for c in self.cells:
            if key < c.key:
                return c.child  # type: ignore[return-value]
        return self.high  # type: ignore[return-value]

    def children(self) -> list[int]:
        return [c.child for c in self.cells] + ([self.high] if self.high is not None else [])

    def replace_child(self, old: int, left: int, sep: bytes, right: int) -> None:
        """把子页 old 替换为 left/right，中间插入 separator。

        数组表示：separators s_0<...<s_{n-1}，children c_0..c_n
        （cells[j]=(s_j, c_j)，high=c_n）。分裂 c_i 后：
            children[i:i+1] = [left, right]
            separators[i:i] = [sep]
        若 old 是被分裂的 leaf，sep 为其首个右移键（复制上来）；
        若 old 是 internal，sep 是该 internal 被提升的键（原页不再保留）。
        """
        children = [c.child for c in self.cells]
        if self.high is not None:
            children.append(self.high)
        if old not in children:
            raise PageFormatError(f"internal {self.pid} 不含子页 {old}")
        i = children.index(old)
        children[i:i + 1] = [left, right]
        separators = [c.key for c in self.cells]
        separators[i:i] = [sep]
        self.cells = [Cell(key=s, child=c) for s, c in zip(separators, children[:-1])]
        self.high = children[-1]

    # ---- 序列化 --------------------------------------------------------
    def encode(self, page_size: int) -> bytes:
        if not self.fits(page_size):
            raise PageFormatError(
                f"page {self.pid} 溢出: used={self.used_bytes()} > page_size={page_size}")
        buf = bytearray(page_size)
        n = len(self.cells)
        free_start = HEADER_SIZE + SLOT_SIZE * n
        free_end = page_size
        offsets: list[int] = []
        for cell in self.cells:  # 按键序写 cell，位置自页尾向前
            body = _encode_cell(cell)
            free_end -= len(body)
            buf[free_end:free_end + len(body)] = body
            offsets.append(free_end)
        for i, off in enumerate(offsets):
            struct.pack_into(">H", buf, HEADER_SIZE + 2 * i, off)
        high = self.high if self.high is not None else SENTINEL
        prev = self.prev if self.prev is not None else SENTINEL
        nxt = self.next if self.next is not None else SENTINEL
        _HEADER.pack_into(buf, 0, MAGIC, self.ptype, n, self.lsn,
                          free_start, free_end, high, prev, nxt)
        return bytes(buf)

    def layout(self, page_size: int) -> dict:
        """页内字节布局（供 SVG/JSON 视图；所有坐标带基准）。"""
        n = len(self.cells)
        free_start = HEADER_SIZE + SLOT_SIZE * n
        free_end = page_size - self.cells_byte_len()
        slots = []
        cursor = free_end
        for i, cell in enumerate(self.cells):
            length = cell.encoded_len()
            slots.append({
                "slot": i,
                "offset": cursor,
                "length": length,
                "key": _preview(cell.key),
                "kind": cell.kind(),
            })
            cursor += length
        return {
            "coordinate_base": "page_start",
            "unit": "byte",
            "page_size": page_size,
            "header": {"offset": 0, "length": HEADER_SIZE},
            "slot_directory": {"offset": HEADER_SIZE, "length": SLOT_SIZE * n},
            "slots": [{"offset": HEADER_SIZE + 2 * i, "length": SLOT_SIZE,
                       "points_to": s["offset"]} for i, s in enumerate(slots)],
            "free_space": {"offset": free_start, "length": max(0, free_end - free_start)},
            "cells": list(reversed(slots)),
        }


def _encode_cell(cell: Cell) -> bytes:
    k = cell.key
    if cell.value is not None:
        return struct.pack(">HH", len(k), len(cell.value)) + k + cell.value
    return struct.pack(">HH", len(k), cell.child) + k  # type: ignore[arg-type]


def _preview(key: bytes, limit: int = 24) -> str:
    try:
        s = key.decode("utf-8")
    except UnicodeDecodeError:
        return "0x" + key[:limit].hex()
    return s if len(s) <= limit else s[:limit] + "…"


def decode_page(pid: int, raw: bytes, *, expect_size: int | None = None) -> Page:
    """从字节读回页。半页 / 魔数错误抛 PageFormatError（恢复据此判定半写页）。"""
    if expect_size is not None and len(raw) != expect_size:
        raise PageFormatError(
            f"page {pid} 字节数 {len(raw)} != 页大小 {expect_size}（疑似写页中断）")
    if len(raw) < HEADER_SIZE:
        raise PageFormatError(f"page {pid} 短于页头（疑似写页中断）")
    magic, ptype, n, lsn, free_start, free_end, high, prev, nxt = _HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise PageFormatError(f"page {pid} magic 损坏: {magic:#06x}")
    if ptype not in (LEAF, INTERNAL):
        raise PageFormatError(f"page {pid} 未知页类型 {ptype}")
    page = Page(pid=pid, ptype=ptype, lsn=lsn,
                high=None if high == SENTINEL else high,
                prev=None if prev == SENTINEL else prev,
                next=None if nxt == SENTINEL else nxt)
    # 校验 free space 记账
    if not (HEADER_SIZE + SLOT_SIZE * n <= free_start <= free_end <= len(raw)):
        raise PageFormatError(f"page {pid} free space 记账越界")
    for i in range(n):
        (off,) = struct.unpack_from(">H", raw, HEADER_SIZE + 2 * i)
        if off < free_end - 1 and False:  # pragma: no cover
            pass
        klen, v2 = struct.unpack_from(">HH", raw, off)
        if ptype == LEAF:
            vlen = v2
            end = off + 4 + klen + vlen
            if end > len(raw):
                raise PageFormatError(f"page {pid} slot {i} cell 越界")
            page.cells.append(Cell(key=raw[off + 4:off + 4 + klen],
                                   value=raw[off + 4 + klen:end]))
        else:
            child = v2
            end = off + 4 + klen
            if end > len(raw):
                raise PageFormatError(f"page {pid} slot {i} cell 越界")
            page.cells.append(Cell(key=raw[off + 4:end], child=child))
    if ptype == INTERNAL:
        keys = [c.key for c in page.cells]
        if keys != sorted(keys):
            raise PageFormatError(f"internal {self_pid(page)} separator 无序")
    else:
        keys = [c.key for c in page.cells]
        if keys != sorted(keys):
            raise PageFormatError(f"leaf {pid} 键无序")
    return page


def self_pid(page: Page) -> int:  # 内部小工具
    return page.pid


def new_leaf(pid: int, *, prev: int | None = None, next_: int | None = None) -> Page:
    return Page(pid=pid, ptype=LEAF, prev=prev, next=next_)


def new_internal(pid: int, cells: Iterable[Cell], high: int) -> Page:
    return Page(pid=pid, ptype=INTERNAL, cells=list(cells), high=high)
