"""固定大小 B-tree 页的二进制布局。

页坐标（basis = page_offset_0，单位字节）::

    [ header 22B ][ slot directory: n * 4B ][  free space  ][ cells 向下生长 ][ checksum 4B ]

slot directory 每项: cell_offset u16 + cell_length u16（坐标基准均为页首）。
leaf cell:  klen u16 | key | vlen u16 | value
internal cell: child u32 | slen u16 | sep
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

MAGIC = 0x5047
PAGE_LEAF = 1
PAGE_INTERNAL = 2
PAGE_META = 3
HEADER_SIZE = 26
CHECKSUM_SIZE = 4
SLOT_SIZE = 4
NO_SIBLING = 0xFFFFFFFF
TORN_FLAG = 0x01

_UNPACK_HDR = struct.Struct("<HBBHHHIHI")
# magic,type,flags,ncells,gap_start,gap_end,sibling I,page_lsn Q,rightmost_child I => 26


def checksum(data: bytes) -> int:
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
    return int(arr.sum(dtype=np.uint32))


@dataclass
class LeafCell:
    key: bytes
    value: bytes

    def encoded(self) -> bytes:
        return struct.pack("<H", len(self.key)) + self.key + struct.pack("<H", len(self.value)) + self.value


@dataclass
class InternalCell:
    child: int
    sep: bytes

    def encoded(self) -> bytes:
        return struct.pack("<I", self.child) + struct.pack("<H", len(self.sep)) + self.sep


@dataclass
class Page:
    pid: int
    page_type: int = PAGE_LEAF
    leaf_cells: list[LeafCell] = field(default_factory=list)
    internal_cells: list[InternalCell] = field(default_factory=list)
    children: list[int] = field(default_factory=list)  # len = len(internal_cells)+1
    sibling: int = NO_SIBLING
    page_lsn: int = 0
    flags: int = 0
    page_size: int = 256

    # ---- 基础派生量（带字节空间语义）----
    def cell_bytes(self) -> list[int]:
        cells = self._cells()
        return [len(c.encoded()) for c in cells]

    def used_bytes(self) -> int:
        return SLOT_SIZE * len(self._cells()) + sum(self.cell_bytes())

    def free_space(self) -> int:
        return self.page_size - HEADER_SIZE - CHECKSUM_SIZE - self.used_bytes()

    def capacity_for_cells(self) -> int:
        """新 cell（含自身 slot）可用的净字节空间。"""
        return self.free_space()

    def _cells(self) -> list:
        if self.page_type == PAGE_LEAF:
            return self.leaf_cells
        return self.internal_cells

    # ---- 序列化 ----
    def serialize(self) -> bytes:
        cells = self._cells()
        buf = bytearray(self.page_size)
        n = len(cells)
        gap_start = HEADER_SIZE + SLOT_SIZE * n
        gap_end = self.page_size - CHECKSUM_SIZE
        # cells 从尾部向下打包；slot 按 cell 顺序记录坐标
        coords: list[tuple[int, int]] = []
        cursor = gap_end
        for c in cells:
            enc = c.encoded()
            cursor -= len(enc)
            coords.append((cursor, len(enc)))
            buf[cursor : cursor + len(enc)] = enc
        off = HEADER_SIZE
        for cell_off, cell_len in coords:
            struct.pack_into("<HH", buf, off, cell_off, cell_len)
            off += SLOT_SIZE
        rightmost = self.children[-1] if (self.page_type == PAGE_INTERNAL and self.children) else 0
        hdr = _UNPACK_HDR.pack(
            MAGIC, self.page_type, self.flags, n, gap_start, gap_end,
            self.sibling, self.page_lsn, rightmost,
        )
        buf[0:HEADER_SIZE] = hdr
        cs = checksum(bytes(buf[: self.page_size - CHECKSUM_SIZE]))
        struct.pack_into("<I", buf, self.page_size - CHECKSUM_SIZE, cs)
        return bytes(buf)

    def slot_directory(self) -> list[dict]:
        """以页首坐标基准列出 slot。"""
        coords = []
        data = self.serialize()
        n = len(self._cells())
        for i in range(n):
            cell_off, cell_len = struct.unpack_from("<HH", data, HEADER_SIZE + i * SLOT_SIZE)
            coords.append(
                {
                    "slot": i,
                    "offset": cell_off,
                    "length": cell_len,
                    "basis": "page_offset_0",
                }
            )
        return coords

    def layout(self) -> dict:
        """页内字节布局（复核界面“页内字节布局”视图数据）。"""
        data = self.serialize()
        n = len(self._cells())
        gap_start = HEADER_SIZE + SLOT_SIZE * n
        gap_end = self.page_size - CHECKSUM_SIZE - sum(self.cell_bytes())
        regions = [
            {"region": "header", "offset": 0, "length": HEADER_SIZE, "basis": "page_offset_0"},
            {
                "region": "slot_directory",
                "offset": HEADER_SIZE,
                "length": SLOT_SIZE * n,
                "basis": "page_offset_0",
            },
            {
                "region": "free_space",
                "offset": gap_start,
                "length": max(0, gap_end - gap_start),
                "basis": "page_offset_0",
            },
            {
                "region": "cell_area",
                "offset": gap_end,
                "length": self.page_size - CHECKSUM_SIZE - gap_end,
                "basis": "page_offset_0",
            },
            {
                "region": "checksum",
                "offset": self.page_size - CHECKSUM_SIZE,
                "length": CHECKSUM_SIZE,
                "basis": "page_offset_0",
            },
        ]
        return {
            "page_size": self.page_size,
            "free_space": self.free_space(),
            "free_space_unit": "B",
            "slots": self.slot_directory(),
            "regions": regions,
            "checksum_valid": checksum(data[: self.page_size - CHECKSUM_SIZE])
            == struct.unpack_from("<I", data, self.page_size - CHECKSUM_SIZE)[0],
        }

    # ---- 反序列化 ----
    @classmethod
    def deserialize(cls, pid: int, data: bytes, page_size: int = 256) -> "Page":
        if len(data) != page_size:
            raise ValueError(f"页 {pid} 字节数 {len(data)} != 页大小 {page_size}")
        magic, ptype, flags, n, gap_start, gap_end, sibling, plsn, rightmost = _UNPACK_HDR.unpack_from(data, 0)
        stored_cs = struct.unpack_from("<I", data, page_size - CHECKSUM_SIZE)[0]
        actual_cs = checksum(data[: page_size - CHECKSUM_SIZE])
        if magic != MAGIC:
            raise ValueError(f"页 {pid} magic 错误")
        page = cls(pid=pid, page_type=ptype, sibling=sibling, page_lsn=plsn, flags=flags, page_size=page_size)
        if stored_cs != actual_cs:
            page.flags |= TORN_FLAG
        for i in range(n):
            cell_off, cell_len = struct.unpack_from("<HH", data, HEADER_SIZE + i * SLOT_SIZE)
            enc = data[cell_off : cell_off + cell_len]
            if ptype == PAGE_LEAF:
                klen = struct.unpack_from("<H", enc, 0)[0]
                key = enc[2 : 2 + klen]
                vlen = struct.unpack_from("<H", enc, 2 + klen)[0]
                val = enc[4 + klen : 4 + klen + vlen]
                page.leaf_cells.append(LeafCell(key, val))
            elif ptype == PAGE_INTERNAL:
                child = struct.unpack_from("<I", enc, 0)[0]
                slen = struct.unpack_from("<H", enc, 4)[0]
                sep = enc[6 : 6 + slen]
                page.internal_cells.append(InternalCell(child, sep))
        if ptype == PAGE_INTERNAL:
            page.children = [c.child for c in page.internal_cells] + [rightmost]
        return page


def torn_copy(data: bytes, torn_at_byte: int) -> bytes:
    """制造写页中断：torn_at_byte 之后全部写零（含 checksum），模拟半页落盘。"""
    out = bytearray(data)
    for i in range(torn_at_byte, len(out)):
        out[i] = 0
    return bytes(out)


# ---- META 页（pid 0）----
@dataclass
class MetaPage:
    root_pid: int = 0
    next_pid: int = 1
    page_size: int = 256
    page_lsn: int = 0
    flags: int = 0

    def serialize(self) -> bytes:
        buf = bytearray(self.page_size)
        hdr = _UNPACK_HDR.pack(MAGIC, PAGE_META, self.flags, 0, HEADER_SIZE,
                               self.page_size - CHECKSUM_SIZE, NO_SIBLING, self.page_lsn, 0)
        buf[0:HEADER_SIZE] = hdr
        struct.pack_into("<II", buf, HEADER_SIZE, self.root_pid, self.next_pid)
        cs = checksum(bytes(buf[: self.page_size - CHECKSUM_SIZE]))
        struct.pack_into("<I", buf, self.page_size - CHECKSUM_SIZE, cs)
        return bytes(buf)

    @staticmethod
    def read(data: bytes, page_size: int = 256) -> "MetaPage":
        magic, ptype, flags, n, gs, ge, sib, plsn, _rm = _UNPACK_HDR.unpack_from(data, 0)
        root_pid, next_pid = struct.unpack_from("<II", data, HEADER_SIZE)
        return MetaPage(root_pid=root_pid, next_pid=next_pid, page_size=page_size,
                        page_lsn=plsn, flags=flags)
