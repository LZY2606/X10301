"""固定大小页的二进制布局。

页内坐标基准：所有偏移量均为「相对本页起始的字节偏移」，单位 bytes。

::

    +------------------ page_size bytes ------------------+
    | page header (PAGE_HEADER_SIZE, 含 body 校验和)      |
    | free space                                           |
    | cells...（自右向左增长）                              |
    | slot directory（自左向右增长，每项 SLOT_SIZE）        |
    +------------------------------------------------------+

slot directory 的第 i 项描述键序第 i 个 cell：
    slot = struct(SLOT_FORMAT) = (offset:u16, klen:u16, vlen:u16, flags:u8)
"""
from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass, field

PAGE_MAGIC = b"BT"
PAGE_HEADER_FORMAT = ">2sHBBHIIHHI"
# magic(2)+page_id(2)+type(1)+flags(1)+lsn(2)+checksum(4)+free_start(4)
# +next(2)+cell_count(2)+body_len(4) = 24
PAGE_HEADER_SIZE = struct.calcs(PAGE_HEADER_FORMAT)
SLOT_FORMAT = ">HHHB"
SLOT_SIZE = struct.calcs(SLOT_FORMAT)

TYPE_LEAF = 1
TYPE_INTERNAL = 2

FLAG_TORN_GUARD = 0x01  # 该页在写入中断模拟中被写成半页（仅用于测试注入）
FLAG_EMPTY_KEPT = 0x02  # 删除后空页：保留、不回收、仍可达


@dataclass
class Entry:
    key: bytes
    value: bytes  # leaf: 原始 value bytes；internal: struct ">H" 右孩子 page id
    flags: int = 0

    def cell_size(self) -> int:
        return len(self.key) + len(self.value)


@dataclass
class Page:
    page_id: int
    page_type: int
    page_size: int
    entries: list[Entry] = field(default_factory=list)
    lsn: int = 0
    next_page: int | None = None  # leaf sibling link（右兄弟）
    leftmost_child: int | None = None  # internal 页左孩子
    flags: int = 0
    torn: bool = False  # 反序列化得到的半页标记
    origin: str = ""  # 来源标注，如 op seq

    # ---- 基本几何 ----
    def slot_dir_bytes(self) -> int:
        return len(self.entries) * SLOT_SIZE

    def cell_bytes(self) -> int:
        return sum(e.cell_size() for e in self.entries)

    def free_space(self) -> int:
        """头部与 slot directory 之外、且未被 cells 占用的字节。"""
        used_cells = self.cell_bytes()
        return self.page_size - PAGE_HEADER_SIZE - self.slot_dir_bytes() - used_cells

    def free_start(self) -> int:
        """cell 区左边界（第一个/最低地址 cell 的偏移）。"""
        return self.page_size - self.slot_dir_bytes() - self.cell_bytes()

    def fits(self, entry: Entry) -> bool:
        return entry.cell_size() + SLOT_SIZE <= self.free_space()

    # ---- 变更 ----
    def put_leaf(self, key: bytes, value: bytes, dup_policy: str) -> str:
        """插入/覆盖 leaf 键，返回 inserted | replaced | rejected。"""
        idx, existing = self._find(key)
        if existing is not None:
            if dup_policy == "reject":
                return "rejected"
            existing.value = value
            existing.flags = 0
            return "replaced"
        self.entries.insert(idx, Entry(key=key, value=value))
        return "inserted"

    def remove_leaf(self, key: bytes) -> bool:
        _, existing = self._find(key)
        if existing is None:
            return False
        self.entries.remove(existing)
        return True

    def _find(self, key: bytes) -> tuple[int, Entry | None]:
        lo, hi = 0, len(self.entries)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.entries[mid].key < key:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.entries) and self.entries[lo].key == key:
            return lo, self.entries[lo]
        return lo, None

    # ---- 序列化 ----
    def to_bytes(self, torn_guard: bool = False) -> bytes:
        """序列化为定长字节。

        torn_guard=True 时模拟写页中断：只写出前半页，剩余字节填 0，
        且头部 flags 带 FLAG_TORN_GUARD（半页判定亦由校验和失败兜底）。
        """
        ordered = sorted(self.entries, key=lambda e: e.key)
        body = bytearray(self.page_size - PAGE_HEADER_SIZE)
        cursor = len(body)  # cell 区从 body 末尾向左
        slots = bytearray()
        for entry in ordered:
            cursor -= entry.cell_size()
            body[cursor:cursor + len(entry.key)] = entry.key
            body[cursor + len(entry.key):cursor + entry.cell_size()] = entry.value
            slots += struct.pack(
                SLOT_FORMAT,
                PAGE_HEADER_SIZE + cursor,
                len(entry.key),
                len(entry.value),
                entry.flags,
            )
        body[0:len(slots)] = slots
        free_start = PAGE_HEADER_SIZE + (len(body) - self.cell_bytes())
        flags = self.flags | (FLAG_TORN_GUARD if torn_guard else 0)
        header = struct.pack(
            PAGE_HEADER_FORMAT,
            PAGE_MAGIC,
            self.page_id,
            self.page_type,
            flags,
            self.lsn & 0xFFFF,
            0,  # checksum 占位
            free_start,
            self.next_page if self.next_page is not None else 0,
            len(ordered),
            len(body),
        )
        checksum = binascii.crc32(header + bytes(body)) & 0xFFFFFFFF
        header = struct.pack(
            PAGE_HEADER_FORMAT,
            PAGE_MAGIC,
            self.page_id,
            self.page_type,
            flags,
            self.lsn & 0xFFFF,
            checksum,
            free_start,
            self.next_page if self.next_page is not None else 0,
            len(ordered),
            len(body),
        )
        raw = bytearray(header + bytes(body))
        if torn_guard:
            for pos in range(self.page_size // 2, self.page_size):
                raw[pos] = 0
        return bytes(raw)

    # ---- 布局视图 ----
    def layout_slots(self) -> list[dict]:
        """返回键序 slot 的坐标信息（单位 bytes，页相对偏移）。"""
        ordered = sorted(self.entries, key=lambda e: e.key)
        out = []
        offsets: dict[int, int] = {}
        cursor = self.page_size - self.slot_dir_bytes()
        for entry in reversed(ordered):
            cursor -= entry.cell_size()
            offsets[id(entry)] = cursor
        for index, entry in enumerate(ordered):
            cell_off = offsets[id(entry)]
            out.append({
                "slot_index": index,
                "slot_offset": PAGE_HEADER_SIZE + index * SLOT_SIZE,
                "cell_offset": cell_off,
                "key_offset": cell_off,
                "value_offset": cell_off + len(entry.key),
                "klen": len(entry.key),
                "vlen": len(entry.value),
                "key_hex": entry.key.hex(),
                "key_text": _safe_text(entry.key),
                "value_hex": entry.value.hex(),
                "flags": entry.flags,
            })
        return out


def _safe_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.hex()


@dataclass
class ParsedPage:
    page: Page | None
    torn: bool
    diagnostic: str | None


def parse_page(data: bytes, page_size: int, page_id: int) -> ParsedPage:
    """读回一页；校验和失败或 magic 不符 => 半页（torn）。"""
    if len(data) != page_size:

        return ParsedPage(None, True, f"page {page_id}: 短写 {len(data)}/{page_size} bytes")
    try:
        (magic, pid, ptype, flags, lsn_low, checksum, free_start,
         nxt, count, body_len) = struct.unpack(PAGE_HEADER_FORMAT, data[:PAGE_HEADER_SIZE])
    except struct.error as exc:
        return ParsedPage(None, True, f"page {page_id}: 头部损坏 ({exc})")
    header = bytearray(data[:PAGE_HEADER_SIZE])
    struct.pack_into(PAGE_HEADER_FORMAT, header, 0, magic, pid, ptype, flags,
                     lsn_low, 0, free_start, nxt, count, body_len)
    actual = binascii.crc32(bytes(header) + data[PAGE_HEADER_SIZE:]) & 0xFFFFFFFF
    if magic != PAGE_MAGIC:
        return ParsedPage(None, True, f"page {page_id}: magic 不符 {magic!r}")
    if actual != checksum:
        return ParsedPage(None, True, f"page {page_id}: 校验和失败，检测到写页中断（半页）")
    body = data[PAGE_HEADER_SIZE:]
    entries: list[Entry] = []
    for i in range(count):
        off, klen, vlen, eflags = struct.unpack_from(SLOT_FORMAT, body, i * SLOT_SIZE)
        rel = off - PAGE_HEADER_SIZE
        if rel < 0 or rel + klen + vlen > len(body):
            return ParsedPage(None, True, f"page {page_id}: slot {i} 越界")
        key = body[rel:rel + klen]
        value = body[rel + klen:rel + klen + vlen]
        entries.append(Entry(key=key, value=value, flags=eflags))
    entries.sort(key=lambda e: e.key)
    page = Page(page_id=pid, page_type=ptype, page_size=page_size,
                entries=entries, lsn=lsn_low,
                next_page=nxt or None, flags=flags)
    if ptype == TYPE_INTERNAL and entries:
        page.leftmost_child = struct.unpack(">H", entries[0].value)[0]
    return ParsedPage(page, False, None)
