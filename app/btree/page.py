"""固定大小页的字节布局编码。

页内坐标（全部 0 基于页起始，单位 byte）::

    [0:20]                header（leaf/internal 共用）
    [20:22]               internal 专有子头：leftmost child pid（leaf 无）
    [22 .. free_start)     internal cell 区（leaf 从 20 开始）
    [free_start .. dstart) free space
    [dstart .. PAGE_SIZE)  slot directory，每项 2 byte，slot0 位于最末端

leaf cell:     uvarint(klen) | key | uvarint(vlen) | value
internal cell: uvarint(klen) | separator key | uint16(right child pid)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAGIC = 0x4254
HEADER_SIZE = 20
INTERNAL_SUBHEADER = 2
LEAF_CONTENT_START = HEADER_SIZE
INTERNAL_CONTENT_START = HEADER_SIZE + INTERNAL_SUBHEADER
SLOT_SIZE = 2
NO_PTR = 0xFFFF

KIND_LEAF = 1
KIND_INTERNAL = 2
FLAG_TORN = 0x01


class PageFormatError(Exception):
    pass


def encode_uvarint(n: int) -> bytes:
    if n < 0:
        raise PageFormatError("uvarint must be non-negative")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def decode_uvarint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise PageFormatError("truncated uvarint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 35:
            raise PageFormatError("uvarint too long")


def leaf_cell(key: bytes, value: bytes) -> bytes:
    return encode_uvarint(len(key)) + key + encode_uvarint(len(value)) + value


def internal_cell(sep: bytes, right_child: int) -> bytes:
    if not 0 <= right_child <= NO_PTR:
        raise PageFormatError(f"child pid out of range: {right_child}")
    return encode_uvarint(len(sep)) + sep + right_child.to_bytes(2, "big")


def decode_leaf_cell(payload: bytes) -> tuple[bytes, bytes]:
    klen, pos = decode_uvarint(payload, 0)
    key = payload[pos:pos + klen]
    pos += klen
    vlen, pos = decode_uvarint(payload, pos)
    value = payload[pos:pos + vlen]
    return key, value


def decode_internal_cell(payload: bytes) -> tuple[bytes, int]:
    klen, pos = decode_uvarint(payload, 0)
    sep = payload[pos:pos + klen]
    pos += klen
    child = int.from_bytes(payload[pos:pos + 2], "big")
    return sep, child


@dataclass
class Page:
    """内存中的一页；encode() 产出固定 PAGE_SIZE 的镜像。"""

    page_id: int
    kind: int = KIND_LEAF
    page_size: int = 128
    keys: list[bytes] = field(default_factory=list)
    values: list[bytes] = field(default_factory=list)       # leaf 用
    children: list[int] = field(default_factory=list)      # internal: [child0, child1..]
    sibling_prev: int = -1
    sibling_next: int = -1
    lsn: int = 0
    torn: bool = False

    # ---- 结构访问 ----
    @property
    def is_leaf(self) -> bool:
        return self.kind == KIND_LEAF

    @property
    def nslots(self) -> int:
        return len(self.keys)

    def content_start(self) -> int:
        return LEAF_CONTENT_START if self.is_leaf else INTERNAL_CONTENT_START

    def cell_payloads(self) -> list[bytes]:
        if self.is_leaf:
            return [leaf_cell(k, v) for k, v in zip(self.keys, self.values)]
        payloads: list[bytes] = []
        for i, key in enumerate(self.keys):
            payloads.append(internal_cell(key, self.children[i + 1]))
        return payloads

    def layout(self) -> dict:
        """页内字节布局，含每个 slot/cell 的坐标与字节尺寸。"""
        payloads = self.cell_payloads()
        sizes = np.array([len(p) for p in payloads], dtype=np.int64)
        start = self.content_start()
        offsets = (start + np.concatenate(([0], np.cumsum(sizes)))).astype(np.int64)
        cells = []
        for i, payload in enumerate(payloads):
            cells.append({
                "slot": i,
                "key_hex": self.keys[i].hex(),
                "key_repr": _key_repr(self.keys[i]),
                "offset": int(offsets[i]),
                "size_bytes": int(sizes[i]),
                "slot_dir_offset": self.page_size - SLOT_SIZE * (i + 1),
            })
        cell_area_end = int(offsets[-1]) if len(payloads) else start
        dir_start = self.page_size - SLOT_SIZE * len(payloads)
        return {
            "page_id": self.page_id,
            "page_size": self.page_size,
            "kind": "leaf" if self.is_leaf else "internal",
            "header": {"offset": 0, "size_bytes": HEADER_SIZE},
            "sub_header": None if self.is_leaf else {
                "offset": HEADER_SIZE, "size_bytes": INTERNAL_SUBHEADER,
                "leftmost_child": self.children[0] if self.children else -1,
            },
            "cells": cells,
            "cell_area_end": cell_area_end,
            "slot_directory_start": dir_start,
            "free_space_start": cell_area_end,
            "free_space_end": dir_start,
            "free_space_bytes": int(dir_start - cell_area_end),
            "torn": self.torn,
            "lsn": self.lsn,
            "sibling_prev": self.sibling_prev,
            "sibling_next": self.sibling_next,
            "coordinate_basis": "page-coordinate/0",
        }

    # ---- 编解码 ----
    def encode(self) -> bytes:
        buf = bytearray(self.page_size)
        flags = FLAG_TORN if self.torn else 0
        payloads = self.cell_payloads()
        start = self.content_start()
        offsets: list[int] = []
        pos = start
        for payload in payloads:
            offsets.append(pos)
            buf[pos:pos + len(payload)] = payload
            pos += len(payload)
        dir_start = self.page_size - SLOT_SIZE * len(payloads)
        if pos > dir_start:
            raise PageFormatError(
                f"page {self.page_id} overflow: cells end {pos} > dir start {dir_start}")
        for i, off in enumerate(offsets):
            where = self.page_size - SLOT_SIZE * (i + 1)
            buf[where:where + 2] = off.to_bytes(2, "big")
        child0 = self.children[0] if (not self.is_leaf and self.children) else -1
        header = b"".join([
            MAGIC.to_bytes(2, "big"),
            (self.page_id & 0xFFFF).to_bytes(2, "big"),
            bytes([self.kind, flags]),
            pos.to_bytes(2, "big"),
            dir_start.to_bytes(2, "big"),
            len(payloads).to_bytes(2, "big"),
            (self.sibling_prev & 0xFFFF).to_bytes(2, "big"),
            (self.sibling_next & 0xFFFF).to_bytes(2, "big"),
            (self.lsn & 0xFFFFFFFF).to_bytes(4, "big"),
        ])
        buf[0:HEADER_SIZE] = header
        if not self.is_leaf:
            buf[HEADER_SIZE:HEADER_SIZE + 2] = (child0 & 0xFFFF).to_bytes(2, "big")
        return bytes(buf)

    @staticmethod
    def decode(data: bytes, page_size: int | None = None) -> "Page":
        size = page_size or len(data)
        if len(data) != size:
            raise PageFormatError(f"page image has {len(data)} bytes, expected {size}")
        if int.from_bytes(data[0:2], "big") != MAGIC:
            raise PageFormatError("bad page magic")
        page_id = int.from_bytes(data[2:4], "big")
        kind = data[4]
        flags = data[5]
        free_start = int.from_bytes(data[6:8], "big")
        dir_start = int.from_bytes(data[8:10], "big")
        nslots = int.from_bytes(data[10:12], "big")
        prev = int.from_bytes(data[12:14], "big")
        nxt = int.from_bytes(data[14:16], "big")
        lsn = int.from_bytes(data[16:20], "big")
        if kind not in (KIND_LEAF, KIND_INTERNAL):
            raise PageFormatError(f"bad page kind {kind}")
        page = Page(
            page_id=page_id, kind=kind, page_size=size,
            sibling_prev=-1 if prev == NO_PTR else prev,
            sibling_next=-1 if nxt == NO_PTR else nxt,
            lsn=lsn, torn=bool(flags & FLAG_TORN),
        )
        expected_dir = size - SLOT_SIZE * nslots
        if dir_start != expected_dir:
            raise PageFormatError(
                f"slot directory start {dir_start} != expected {expected_dir}")
        for i in range(nslots):
            where = size - SLOT_SIZE * (i + 1)
            off = int.from_bytes(data[where:where + 2], "big")
            if off < page.content_start() or off + (0 if i + 1 < nslots else 0) > size:
                raise PageFormatError(f"slot {i} offset {off} out of bounds")
            end: int
            if i + 1 < nslots:
                next_where = size - SLOT_SIZE * (i + 2)
                end = int.from_bytes(data[next_where:next_where + 2], "big")
            else:
                end = free_start
            payload = data[off:end]
            if page.is_leaf:
                key, value = decode_leaf_cell(payload)
                page.keys.append(key)
                page.values.append(value)
            else:
                sep, child = decode_internal_cell(payload)
                page.keys.append(sep)
                page.children.append(child)
        if not page.is_leaf:
            child0 = int.from_bytes(data[HEADER_SIZE:HEADER_SIZE + 2], "big")
            page.children = [(-1 if child0 == NO_PTR else child0)] + page.children
        return page


def _key_repr(key: bytes) -> str:
    try:
        text = key.decode("utf-8")
        if all(ch.isprintable() for ch in text):
            return text
    except UnicodeDecodeError:
        pass
    return "0x" + key.hex()
