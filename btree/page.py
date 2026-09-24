"""Fixed-size page layout with an explicit slot directory.

Leaf page body::

    [cell_n-1 ... cell_1 cell_0][slot_0][slot_1]...

Internal page body::

    [child0][sep1:cell1][child1]...  -> slots describe (key, child_after)

Every byte coordinate reported back carries a byte-offset basis so the review
UI can draw a faithful map of the page.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import BYTE_OFFSET

LEAF = "leaf"
INTERNAL = "internal"

MAGIC = b"PLBT"  # page layout B-tree
# magic(4) type(1) pid(2) nslots(2) free_start(2) free_end(2)
# sibling(2) flags(1) torn_marker(1) = 17
HEADER_SIZE = 17
SLOT_SIZE = 4  # offset(2) length(2)
PAGE_PAYLOAD_OFF = HEADER_SIZE

# flags
FLAG_TORN = 0x01
FLAG_FREED = 0x02


@dataclass
class Cell:
    key: bytes
    value: bytes = b""
    child: int = -1  # internal pages: child pointer to the right of the key

    def leaf_bytes(self) -> bytes:
        # length-prefixed key/value keeps every cell self describing
        return len(self.key).to_bytes(2, "big") + self.key + len(self.value).to_bytes(2, "big") + self.value

    def internal_bytes(self) -> bytes:
        return len(self.key).to_bytes(2, "big") + self.key + self.child.to_bytes(2, "big")


@dataclass
class SlotInfo:
    index: int
    offset: int
    length: int
    key: bytes
    value: bytes
    child: int


@dataclass
class Page:
    pid: int
    ptype: str = LEAF
    slots: list[Cell] = None
    sibling: int = -1
    leftmost_child: int = -1  # internal pages only
    flags: int = 0
    torn_lsn: int = -1  # when FLAG_TORN: lsn of the torn write
    page_lsn: int = 0
    page_size: int = 128

    def __post_init__(self):
        if self.slots is None:
            self.slots = []

    @property
    def is_torn(self) -> bool:
        return bool(self.flags & FLAG_TORN)

    @property
    def is_freed(self) -> bool:
        return bool(self.flags & FLAG_FREED)

    def dir_size(self) -> int:
        return len(self.slots) * SLOT_SIZE

    def cell_overhead(self) -> int:
        # 2-byte key len + 2-byte value len on leaves
        return len(self.slots) * 4

    def used_bytes(self) -> int:
        if self.ptype == LEAF:
            cells = sum(len(c.leaf_bytes()) for c in self.slots)
        else:
            cells = sum(len(c.internal_bytes()) for c in self.slots)
        return HEADER_SIZE + self.dir_size() + cells

    def free_bytes(self) -> int:
        return self.page_size - self.used_bytes()

    # --- serialisation -------------------------------------------------

    def serialize(self) -> bytes:
        buf = bytearray(self.page_size)
        buf[0:4] = MAGIC
        buf[4] = 1 if self.ptype == LEAF else 2
        buf[5:7] = self.pid.to_bytes(2, "big")
        buf[7:9] = len(self.slots).to_bytes(2, "big")
        buf[13:15] = (self.sibling & 0xFFFF).to_bytes(2, "big")
        buf[15] = self.flags
        buf[16] = (self.torn_lsn & 0xFF).to_bytes(1, "big") if self.torn_lsn >= 0 else b"\x00"

        if self.ptype == INTERNAL:
            # leftmost child pointer stored in the first two payload bytes; the
            # slot directory therefore starts after it.
            buf[HEADER_SIZE:HEADER_SIZE + 2] = (self.leftmost_child & 0xFFFF).to_bytes(2, "big")
            dir_off = HEADER_SIZE + 2
            free_end = dir_off
            free_start = self.page_size
        else:
            dir_off = HEADER_SIZE
            free_end = HEADER_SIZE
            free_start = self.page_size

        for i, cell in enumerate(self.slots):
            body = cell.leaf_bytes() if self.ptype == LEAF else cell.internal_bytes()
            free_start -= len(body)
            buf[free_start:free_start + len(body)] = body
            buf[dir_off + i * SLOT_SIZE:dir_off + i * SLOT_SIZE + 2] = free_start.to_bytes(2, "big")
            buf[dir_off + i * SLOT_SIZE + 2:dir_off + i * SLOT_SIZE + 4] = len(body).to_bytes(2, "big")
        buf[9:11] = free_start.to_bytes(2, "big")
        buf[11:13] = free_end.to_bytes(2, "big")
        return bytes(buf)

    @staticmethod
    def deserialize(data: bytes, page_size: int) -> "Page":
        if len(data) < HEADER_SIZE or data[0:4] != MAGIC:
            raise ValueError("bad page magic")
        ptype = LEAF if data[4] == 1 else INTERNAL
        pid = int.from_bytes(data[5:7], "big")
        nslots = int.from_bytes(data[7:9], "big")
        sibling = int.from_bytes(data[13:15], "big")
        flags = data[15]
        torn_lsn = data[16]
        page = Page(pid=pid, ptype=ptype, sibling=sibling, flags=flags,
                    torn_lsn=torn_lsn if flags & FLAG_TORN else -1, page_size=page_size)
        if ptype == INTERNAL:
            page.leftmost_child = int.from_bytes(data[HEADER_SIZE:HEADER_SIZE + 2], "big")
            dir_off = HEADER_SIZE + 2
        else:
            dir_off = HEADER_SIZE
        for i in range(nslots):
            e = dir_off + i * SLOT_SIZE
            off = int.from_bytes(data[e:e + 2], "big")
            length = int.from_bytes(data[e + 2:e + 4], "big")
            blob = data[off:off + length]
            if ptype == LEAF:
                klen = int.from_bytes(blob[0:2], "big")
                key = blob[2:2 + klen]
                voff = 2 + klen
                vlen = int.from_bytes(blob[voff:voff + 2], "big")
                value = blob[voff + 2:voff + 2 + vlen]
                page.slots.append(Cell(key=key, value=value))
            else:
                klen = int.from_bytes(blob[0:2], "big")
                key = blob[2:2 + klen]
                child = int.from_bytes(blob[2 + klen:4 + klen], "big")
                page.slots.append(Cell(key=key, child=child))
        return page

    # --- accessors -----------------------------------------------------

    def slot_infos(self) -> list[SlotInfo]:
        out = []
        if self.ptype == INTERNAL:
            dir_off = HEADER_SIZE + 2
        else:
            dir_off = HEADER_SIZE
        # offsets only known after packing; recompute deterministically
        packed = self.serialize()
        for i, cell in enumerate(self.slots):
            e = dir_off + i * SLOT_SIZE
            off = int.from_bytes(packed[e:e + 2], "big")
            length = int.from_bytes(packed[e + 2:e + 4], "big")
            out.append(SlotInfo(i, off, length, cell.key, cell.value, cell.child))
        return out

    def find_slot(self, key: bytes) -> int:
        lo, hi = 0, len(self.slots)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.slots[mid].key < key:
                lo = mid + 1
            else:
                hi = mid
        return lo


def freed_page(pid: int, page_size: int) -> Page:
    return Page(pid=pid, ptype=LEAF, flags=FLAG_FREED, page_size=page_size)
