"""固定页大小下的页内字节布局原语。

页布局（坐标基准 = 页首字节偏移 0，单位 byte）::

    leaf:     [type:1][page_id:4][parent:4][prev:4][next:4][pagelsn:4][free_end:2] = 23B
              之后 cell 区从页尾向低地址生长；slot directory 紧跟 header。
    internal: [type:1][page_id:4][parent:4][left_child:4][pagelsn:4][free_end:2]   = 19B
              之后同样为 cell 区 + slot directory。

slot directory：每项 2B（cell 起始偏移，big-endian uint16），按“键序”排列。
cell 布局：
    leaf     = [klen:2][vlen:2][key:klen][value:vlen]
    internal = [klen:2][child:4][key:klen]

free space 指 cell 区高水位（free_end）与 slot directory 尾之间的连续字节，
按 byte 计量（含碎片化时也以 repack 后的连续空间为准）。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

LEAF = 1
INTERNAL = 2

LEAF_HEADER = 23      # type1 + pid4 + parent4 + prev4 + next4 + pagelsn4 + free_end2
INTERNAL_HEADER = 19  # type1 + pid4 + parent4 + left_child4 + pagelsn4 + free_end2
SLOT = 2              # 每个 slot directory 项 2B


def _u16(b: int) -> bytes:
    return struct.pack(">H", b)


def _put_u16(buf: bytearray, off: int, val: int) -> None:
    buf[off:off + 2] = _u16(val)


def _u32(b: int) -> bytes:
    return struct.pack(">I", b)


def _get_u32(buf, off: int) -> int:
    return struct.unpack(">I", bytes(buf[off:off + 4]))[0]


def leaf_cell_size(klen: int, vlen: int) -> int:
    """leaf cell 占用字节数（含 2B klen + 2B vlen 开销）。"""
    return 4 + klen + vlen


def internal_cell_size(klen: int) -> int:
    """internal cell 占用字节数（2B klen + 4B child + key）。"""
    return 6 + klen


@dataclass
class SlotInfo:
    index: int                 # slot 序号（键序，0-based），单位：1 slot
    offset: int                # cell 起始字节偏移，单位 byte，基准页首
    length: int                # cell 长度，单位 byte
    key: bytes


@dataclass
class LeafCell:
    key: bytes
    value: bytes

    def size(self) -> int:
        return leaf_cell_size(len(self.key), len(self.value))

    def encode(self) -> bytes:
        return _u16(len(self.key)) + _u16(len(self.value)) + self.key + self.value

    @staticmethod
    def decode(buf: bytes) -> "LeafCell":
        klen, vlen = struct.unpack(">HH", buf[:4])
        key = buf[4:4 + klen]
        value = buf[4 + klen:4 + klen + vlen]
        return LeafCell(key, value)


@dataclass
class InternalCell:
    """separator entry：child 指向 key >= self.key 的子树（left_child 除外）。"""
    key: bytes
    child: int

    def size(self) -> int:
        return internal_cell_size(len(self.key))

    def encode(self) -> bytes:
        return _u16(len(self.key)) + _u32(self.child) + self.key

    @staticmethod
    def decode(buf: bytes) -> "InternalCell":
        klen = struct.unpack(">H", buf[:2])[0]
        child = _get_u32(buf, 2)
        key = bytes(buf[6:6 + klen])
        return InternalCell(key, child)


@dataclass
class Page:
    page_type: int
    page_id: int
    parent: int = 0
    # leaf only
    prev: int = 0
    next: int = 0
    # internal only
    left_child: int = 0
    # 该页最近一次落盘对应的日志序号（0 = 尚未有日志记录）
    lsn: int = 0
    leaf_cells: List[LeafCell] = field(default_factory=list)
    internal_cells: List[InternalCell] = field(default_factory=list)

    # ---- 类型便捷构造 ----
    @staticmethod
    def new_leaf(pid: int, parent: int, prev: int = 0, next: int = 0,
                 lsn: int = 0) -> "Page":
        return Page(LEAF, pid, parent, prev=prev, next=next, lsn=lsn)

    @staticmethod
    def new_internal(pid: int, parent: int, left_child: int,
                     lsn: int = 0) -> "Page":
        return Page(INTERNAL, pid, parent, left_child=left_child, lsn=lsn)

    @property
    def is_leaf(self) -> bool:
        return self.page_type == LEAF

    def header_size(self) -> int:
        return LEAF_HEADER if self.is_leaf else INTERNAL_HEADER

    # ---- 键序访问 ----
    def keys(self) -> List[bytes]:
        if self.is_leaf:
            return [c.key for c in self.leaf_cells]
        return [c.key for c in self.internal_cells]

    def child_ids(self) -> List[int]:
        if self.is_leaf:
            return []
        return [self.left_child] + [c.child for c in self.internal_cells]

    def sort(self) -> None:
        if self.is_leaf:
            self.leaf_cells.sort(key=lambda c: c.key)
        else:
            self.internal_cells.sort(key=lambda c: c.key)

    # ---- 容量（单位 byte）----
    def used(self) -> int:
        n = len(self.leaf_cells) if self.is_leaf else len(self.internal_cells)
        body = sum(c.size() for c in (self.leaf_cells if self.is_leaf else self.internal_cells))
        return self.header_size() + SLOT * n + body

    def free_space(self, page_size: int) -> int:
        """连续可用字节：page_size - 已用字节（repack 视角）。"""
        return page_size - self.used()

    def fits(self, extra_bytes: int, page_size: int) -> bool:
        return self.free_space(page_size) >= extra_bytes

    def needs_split(self, page_size: int) -> bool:
        """分裂判据完全基于字节空间，而非键的条数。"""
        return self.used() > page_size

    # ---- 序列化（坐标信息来自此布局）----
    def serialize(self, page_size: int) -> bytes:
        buf = bytearray(page_size)
        buf[0] = self.page_type
        off = 1
        buf[off:off + 4] = _u32(self.page_id)
        off += 4
        buf[off:off + 4] = _u32(self.parent)
        off += 4
        if self.is_leaf:
            buf[off:off + 4] = _u32(self.prev)
            off += 4
            buf[off:off + 4] = _u32(self.next)
            off += 4
            buf[off:off + 4] = _u32(self.lsn)
            off += 4
        else:
            buf[off:off + 4] = _u32(self.left_child)
            off += 4
            buf[off:off + 4] = _u32(self.lsn)
            off += 4
        free_end_off = off
        off += 2  # free_end 占位

        cells = self.leaf_cells if self.is_leaf else self.internal_cells
        ordered = list(cells)
        ordered.sort(key=lambda c: c.key)
        free_end = page_size
        slot_entries: List[Tuple[int, int]] = []
        for idx, cell in enumerate(ordered):
            enc = cell.encode()
            start = free_end - len(enc)
            buf[start:free_end] = enc
            slot_entries.append((off + idx * SLOT, start))
            free_end = start
        for slot_off, cell_off in slot_entries:
            _put_u16(buf, slot_off, cell_off)
        _put_u16(buf, free_end_off, free_end)
        return bytes(buf)

    @staticmethod
    def parse(raw: bytes) -> "Page":
        ptype = raw[0]
        pid = _get_u32(raw, 1)
        parent = _get_u32(raw, 5)
        if ptype == LEAF:
            prev = _get_u32(raw, 9)
            nxt = _get_u32(raw, 13)
            lsn = _get_u32(raw, 17)
            free_end = struct.unpack(">H", raw[21:23])[0]
            page = Page.new_leaf(pid, parent, prev, nxt)
            slot_start = LEAF_HEADER
        elif ptype == INTERNAL:
            left_child = _get_u32(raw, 9)
            lsn = _get_u32(raw, 13)
            free_end = struct.unpack(">H", raw[17:19])[0]
            page = Page.new_internal(pid, parent, left_child)
            slot_start = INTERNAL_HEADER
        else:
            raise ValueError(f"unknown page type {ptype} on page {pid}")
        n_slots = (free_end - slot_start) // SLOT
        for i in range(n_slots):
            cell_off = struct.unpack(">H", raw[slot_start + i * SLOT: slot_start + i * SLOT + 2])[0]
            if cell_off == 0:
                break
            if ptype == LEAF:
                klen, vlen = struct.unpack(">HH", raw[cell_off:cell_off + 4])
                cell = LeafCell.decode(raw[cell_off:cell_off + leaf_cell_size(klen, vlen)])
                page.leaf_cells.append(cell)
            else:
                klen = struct.unpack(">H", raw[cell_off:cell_off + 2])[0]
                cell = InternalCell.decode(raw[cell_off:cell_off + internal_cell_size(klen)])
                page.internal_cells.append(cell)
        page.lsn = lsn
        page.sort()
        return page

    # ---- 视图辅助 ----
    def layout(self, page_size: int) -> dict:
        """页内字节布局视图（单位 byte，坐标基准页首）。

        返回 header/slots/free_space 的稳定描述，供“页内字节布局”面板展示。
        """
        overflow = self.used() > page_size
        raw = self.serialize(max(page_size, self.used()))
        hs = self.header_size()
        n = len(self.leaf_cells) if self.is_leaf else len(self.internal_cells)
        ordered = (sorted(self.leaf_cells, key=lambda c: c.key) if self.is_leaf
                   else sorted(self.internal_cells, key=lambda c: c.key))
        free_end = struct.unpack(">H", raw[hs - 2:hs])[0]
        slots: List[SlotInfo] = []
        for i, cell in enumerate(ordered):
            off = struct.unpack(">H", raw[hs + i * SLOT: hs + i * SLOT + 2])[0]
            slots.append(SlotInfo(index=i, offset=off, length=cell.size(), key=cell.key))
        return {
            "page_id": self.page_id,
            "page_type": "leaf" if self.is_leaf else "internal",
            "page_size_bytes": page_size,
            "header_bytes": hs,
            "slot_directory_offset_bytes": hs,
            "slot_directory_bytes": SLOT * n,
            "free_end_offset_bytes": free_end,
            "free_space_bytes": self.free_space(page_size),
            "overflow_bytes": max(0, self.used() - page_size),
            "fits_page": not overflow,
            "slots": [
                {"index": s.index, "unit": "slot", "offset_bytes": s.offset,
                 "length_bytes": s.length, "key_hex": s.key.hex()}
                for s in slots
            ],
        }

    def clone(self) -> "Page":
        if self.is_leaf:
            return Page.new_leaf(self.page_id, self.parent, self.prev, self.next, self.lsn)
        return Page.new_internal(self.page_id, self.parent, self.left_child, self.lsn)
