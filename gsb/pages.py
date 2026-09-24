"""固定大小页的字节布局（带单位与坐标基准，派生层不丢原始表示）。

坐标基准：所有偏移单位为 byte，页大小固定 ``PAGE_SIZE`` 字节。
页布局（leaf / internal 通用头）::

    offset 0   magic (2B, 0xB7EE)
           2   type  (1B: 1=leaf, 2=internal, 0=meta)
           3   ncells (1B)
           4   free_start (2B, big-endian, cell 区起始/最低偏移)
           6   slots_end (2B, slot directory 之后第一个可用偏移)
           8   right_sibling (4B: 0xFFFFFFFF 表示无)
          12   leftmost_child (4B, internal; leaf 为 0xFFFFFFFF)
          16   page_lsn (4B, 最近写入本页的日志序号)
          20   flags (2B; bit0=torn 半写标记仅出现在故障镜像)
          22   头结束；slot directory 从 22 向高地址增长，
               每项 4B: cell_off(2B) + cell_len(2B)，按键升序
    高地址端：cell 从 PAGE_SIZE-2 向低地址堆叠，
    PAGE_SIZE-2 .. PAGE_SIZE-1: 本页整页 CRC16（对前 PAGE_SIZE-2 字节）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

PAGE_SIZE = 128
MAGIC = 0xB7EE
PAGE_META = 0
PAGE_LEAF = 1
PAGE_INTERNAL = 2
NIL = 0xFFFFFFFF
HEADER_END = 22
TRAILER_LEN = 2
SLOT_LEN = 4


def _crc16(buf: bytes) -> int:
    """CRC-16/CCITT-FALSE，逐字节计算（用 numpy 向量化批量生成也可）。"""
    crc = 0xFFFF
    arr = np.frombuffer(buf, dtype=np.uint8)
    for b in arr:
        crc ^= int(b) << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


@dataclass
class Cell:
    """一个物理 cell。leaf: (key, value); internal: (sep_key, right_child)。"""

    key: bytes
    value: bytes = b""
    right_child: Optional[int] = None  # internal 专用
    off: int = -1  # 物理坐标（在页内的字节偏移），编码后填
    length: int = 0

    def encoded_len(self) -> int:
        return 2 + len(self.key) + (2 + len(self.value) if self.right_child is None else 4)

    def encode(self) -> bytes:
        if self.right_child is None:
            return (
                len(self.key).to_bytes(2, "big")
                + self.key
                + len(self.value).to_bytes(2, "big")
                + self.value
            )
        return (
            len(self.key).to_bytes(2, "big")
            + self.key
            + int(self.right_child).to_bytes(4, "big")
        )

    @property
    def kind(self) -> str:
        return "leaf" if self.right_child is None else "internal"


@dataclass
class Page:
    pid: int
    ptype: int = PAGE_LEAF
    cells: List[Cell] = field(default_factory=list)
    right_sibling: Optional[int] = None
    leftmost_child: Optional[int] = None
    page_lsn: int = 0
    torn: bool = False  # 故障镜像的半写状态（仅模拟磁盘用）
    created_lsn: int = 0

    # ---- 结构访问 ----
    def is_leaf(self) -> bool:
        return self.ptype == PAGE_LEAF

    def child_pids(self) -> List[int]:
        if self.is_leaf():
            return []
        out = []
        if self.leftmost_child is not None:
            out.append(self.leftmost_child)
        out.extend(c.right_child for c in self.cells)
        return out

    def find_leaf(self, key: bytes) -> int:
        assert not self.is_leaf()
        if self.leftmost_child is not None and (not self.cells or key < self.cells[0].key):
            return self.leftmost_child
        for c in self.cells:
            if key < c.key:
                return c.right_child
        return self.cells[-1].right_child

    def slot_cost(self) -> int:
        return HEADER_END + SLOT_LEN * len(self.cells)

    def cell_bytes(self) -> int:
        return sum(c.encoded_len() for c in self.cells)

    def free_space(self) -> int:
        """free space（byte）= cell 区与 slot directory 之间的连续空间 - trailer。"""
        if not self.cells:
            return PAGE_SIZE - TRAILER_LEN - self.slot_cost()
        used = self.slot_cost() + self.cell_bytes() + TRAILER_LEN
        return PAGE_SIZE - used

    def fits(self) -> bool:
        return self.free_space() >= 0

    # ---- 编解码 ----
    def encode(self, mark_torn: bool = False) -> bytes:
        n = len(self.cells)
        ordered = sorted(self.cells, key=lambda c: c.key)
        body = bytearray(PAGE_SIZE)
        body[0:2] = MAGIC.to_bytes(2, "big")
        body[2] = self.ptype
        body[3] = n
        slots_end = HEADER_END + SLOT_LEN * n
        free_start = PAGE_SIZE - TRAILER_LEN
        if self.right_sibling is None:
            body[8:12] = NIL.to_bytes(4, "big")
        else:
            body[8:12] = self.right_sibling.to_bytes(4, "big")
        lm = self.leftmost_child if (not self.is_leaf() and self.leftmost_child is not None) else NIL
        body[12:16] = lm.to_bytes(4, "big")
        body[16:20] = self.page_lsn.to_bytes(4, "big")
        flags = 1 if mark_torn else 0
        body[20:22] = flags.to_bytes(2, "big")

        # cell 按键序从高地址堆叠：第 i 个逻辑 cell 坐标可由内容稳定决定
        cursor = PAGE_SIZE - TRAILER_LEN
        for idx, cell in enumerate(ordered):
            blob = cell.encode()
            off = cursor - len(blob)
            body[off:cursor] = blob
            cell.off = off
            cell.length = len(blob)
            slot_pos = HEADER_END + idx * SLOT_LEN
            body[slot_pos : slot_pos + 2] = off.to_bytes(2, "big")
            body[slot_pos + 2 : slot_pos + 4] = len(blob).to_bytes(2, "big")
            cursor = off
        free_start = cursor
        body[4:6] = free_start.to_bytes(2, "big")
        body[6:8] = slots_end.to_bytes(2, "big")
        crc = _crc16(bytes(body[: PAGE_SIZE - TRAILER_LEN]))
        body[PAGE_SIZE - TRAILER_LEN :] = crc.to_bytes(2, "big")
        return bytes(body)

    def clone(self) -> "Page":
        return decode_page(self.pid, self.encode(mark_torn=self.torn))


@dataclass
class Decoded:
    page: Optional[Page]
    valid: bool
    reason: Optional[str] = None


def decode_page(pid: int, raw: bytes, expect_torn: bool = False) -> Page:
    body = bytearray(raw)
    stored_crc = int.from_bytes(body[PAGE_SIZE - TRAILER_LEN :], "big")
    calc_crc = _crc16(bytes(body[: PAGE_SIZE - TRAILER_LEN]))
    torn = stored_crc != calc_crc
    ptype = body[2]
    ncells = body[3]
    free_start = int.from_bytes(body[4:6], "big")
    slots_end = int.from_bytes(body[6:8], "big")
    rs = int.from_bytes(body[8:12], "big")
    lm = int.from_bytes(body[12:16], "big")
    lsn = int.from_bytes(body[16:20], "big")
    flags = int.from_bytes(body[20:22], "big")
    p = Page(pid=pid, ptype=ptype, right_sibling=None if rs == NIL else rs,
             leftmost_child=None if lm == NIL else lm, page_lsn=lsn,
             torn=torn or expect_torn or bool(flags & 1))
    for i in range(ncells):
        sp = HEADER_END + i * SLOT_LEN
        off = int.from_bytes(body[sp : sp + 2], "big")
        clen = int.from_bytes(body[sp + 2 : sp + 4], "big")
        klen = int.from_bytes(body[off : off + 2], "big")
        key = bytes(body[off + 2 : off + 2 + klen])
        if ptype == PAGE_LEAF:
            vp = off + 2 + klen
            vlen = int.from_bytes(body[vp : vp + 2], "big")
            val = bytes(body[vp + 2 : vp + 2 + vlen])
            p.cells.append(Cell(key=key, value=val, off=off, length=clen))
        else:
            cp = off + 2 + klen
            child = int.from_bytes(body[cp : cp + 4], "big")
            p.cells.append(Cell(key=key, right_child=child, off=off, length=clen))
    p.cells.sort(key=lambda c: c.key)
    p.free_start_ = free_start  # type: ignore[attr-defined]
    p.slots_end_ = slots_end  # type: ignore[attr-defined]
    return p


def new_leaf(pid: int, lsn: int, right_sibling: Optional[int] = None) -> Page:
    return Page(pid=pid, ptype=PAGE_LEAF, right_sibling=right_sibling,
                page_lsn=lsn, created_lsn=lsn)


def new_internal(pid: int, lsn: int, leftmost: int) -> Page:
    return Page(pid=pid, ptype=PAGE_INTERNAL, leftmost_child=leftmost,
                page_lsn=lsn, created_lsn=lsn)
