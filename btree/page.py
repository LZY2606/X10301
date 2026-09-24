"""Fixed-size B-tree page layout.

Layout (all integers little-endian)::

    0   2   magic 'BP'
    2   4   page_id  (u32)
    6   1   flags    (bit0 = leaf)
    7   2   slot_count (u16)
    9   2   free_start (u16)  end of data region (grows upward from HEADER_SIZE)
    11  2   free_end   (u16)  start of slot directory (grows downward from page end)
    13  4   right_sibling page id (u32, 0 = none)
    17  4   leftmost_child page id (u32, internal pages only)
    21  8   lsn (u64)
    29  2   crc16 over every byte except the crc field itself
    31  1   reserved

Leaf record:     klen u16 | vlen u16 | key bytes | value bytes
Internal record: klen u16 | key bytes | right-child page id u32
Slot:            offset u16 | length u16   (slots are kept sorted by key)
"""
import struct

MAGIC = b"BP"
HEADER_SIZE = 32
LEAF_FLAG = 0x1
SLOT_SIZE = 4
CRC_OFFSET = 29


class PageOverflow(Exception):
    pass


class PageCorrupt(Exception):
    pass


def crc16(buf: bytes) -> int:
    crc = 0xFFFF
    for i, b in enumerate(buf):
        if CRC_OFFSET <= i < CRC_OFFSET + 2:
            continue
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc & 0xFFFF


def leaf_rec(key: bytes, value: bytes) -> bytes:
    return struct.pack("<HH", len(key), len(value)) + key + value


def leaf_value(rec: bytes) -> bytes:
    klen, vlen = struct.unpack_from("<HH", rec, 0)
    return rec[4 + klen:4 + klen + vlen]


def internal_rec(key: bytes, child: int) -> bytes:
    return struct.pack("<H", len(key)) + key + struct.pack("<I", child)


def internal_child(rec: bytes) -> int:
    klen = struct.unpack_from("<H", rec, 0)[0]
    return struct.unpack_from("<I", rec, 2 + klen)[0]


def rec_key(rec: bytes, leaf: bool) -> bytes:
    if leaf:
        klen = struct.unpack_from("<H", rec, 0)[0]
        return rec[4:4 + klen]
    klen = struct.unpack_from("<H", rec, 0)[0]
    return rec[2:2 + klen]


class Page:
    def __init__(self, page_size: int, page_id: int = 0, flags: int = LEAF_FLAG, buf=None):
        self.page_size = page_size
        if buf is None:
            self.buf = bytearray(page_size)
            self.buf[0:2] = MAGIC
            struct.pack_into("<I", self.buf, 2, page_id)
            self.buf[6] = flags
            struct.pack_into("<H", self.buf, 7, 0)
            struct.pack_into("<H", self.buf, 9, HEADER_SIZE)
            struct.pack_into("<H", self.buf, 11, page_size)
        else:
            if len(buf) != page_size:
                raise PageCorrupt(f"page buffer is {len(buf)} bytes, expected {page_size}")
            self.buf = bytearray(buf)

    # -- header fields -------------------------------------------------
    @property
    def page_id(self) -> int:
        return struct.unpack_from("<I", self.buf, 2)[0]

    @property
    def flags(self) -> int:
        return self.buf[6]

    @property
    def is_leaf(self) -> bool:
        return bool(self.buf[6] & LEAF_FLAG)

    @property
    def slot_count(self) -> int:
        return struct.unpack_from("<H", self.buf, 7)[0]

    @property
    def free_start(self) -> int:
        return struct.unpack_from("<H", self.buf, 9)[0]

    @property
    def free_end(self) -> int:
        return struct.unpack_from("<H", self.buf, 11)[0]

    @property
    def free_space(self) -> int:
        return self.free_end - self.free_start

    @property
    def right_sibling(self) -> int:
        return struct.unpack_from("<I", self.buf, 13)[0]

    @right_sibling.setter
    def right_sibling(self, v: int):
        struct.pack_into("<I", self.buf, 13, v)

    @property
    def leftmost_child(self) -> int:
        return struct.unpack_from("<I", self.buf, 17)[0]

    @leftmost_child.setter
    def leftmost_child(self, v: int):
        struct.pack_into("<I", self.buf, 17, v)

    @property
    def lsn(self) -> int:
        return struct.unpack_from("<Q", self.buf, 21)[0]

    @lsn.setter
    def lsn(self, v: int):
        struct.pack_into("<Q", self.buf, 21, v)

    # -- integrity -----------------------------------------------------
    def set_crc(self):
        struct.pack_into("<H", self.buf, CRC_OFFSET, crc16(bytes(self.buf)))

    def crc_ok(self) -> bool:
        return crc16(bytes(self.buf)) == struct.unpack_from("<H", self.buf, CRC_OFFSET)[0]

    def magic_ok(self) -> bool:
        return bytes(self.buf[0:2]) == MAGIC

    # -- records ---------------------------------------------------------
    def slot(self, i: int):
        return struct.unpack_from("<HH", self.buf, self.page_size - SLOT_SIZE * (i + 1))

    def records(self):
        """Return list of (key, raw_record) sorted by key."""
        n = self.slot_count
        out = []
        for i in range(n):
            off, ln = self.slot(i)
            if off < HEADER_SIZE or off + ln > self.page_size - SLOT_SIZE * n:
                raise PageCorrupt(f"slot {i} points outside data region")
            rec = bytes(self.buf[off:off + ln])
            out.append((rec_key(rec, self.is_leaf), rec))
        return out

    def needed_bytes(self, records) -> int:
        return sum(len(r) for _, r in records) + SLOT_SIZE * len(records)

    def fits(self, records) -> bool:
        return HEADER_SIZE + self.needed_bytes(records) <= self.page_size

    def rebuild(self, records, lsn: int):
        """Rewrite the whole page from a sorted record list (compacts free space)."""
        if not self.fits(records):
            raise PageOverflow(
                f"page {self.page_id}: need {HEADER_SIZE + self.needed_bytes(records)} > {self.page_size}")
        data = bytearray()
        slots = []
        for _, rec in records:
            slots.append((HEADER_SIZE + len(data), len(rec)))
            data += rec
        free_start = HEADER_SIZE + len(data)
        free_end = self.page_size - SLOT_SIZE * len(records)
        self.buf[HEADER_SIZE:free_start] = data
        self.buf[free_start:] = b"\x00" * (self.page_size - free_start)
        for i, (off, ln) in enumerate(slots):
            struct.pack_into("<HH", self.buf, self.page_size - SLOT_SIZE * (i + 1), off, ln)
        struct.pack_into("<H", self.buf, 7, len(records))
        struct.pack_into("<H", self.buf, 9, free_start)
        struct.pack_into("<H", self.buf, 11, free_end)
        self.lsn = lsn

    def slot_directory(self):
        return [{"slot": i, "offset": off, "length": ln}
                for i, (off, ln) in enumerate(self.slot(i) for i in range(self.slot_count))]

    def validate(self):
        problems = []
        if not self.magic_ok():
            problems.append("bad_magic")
            return problems
        if not self.crc_ok():
            problems.append("bad_crc")
            return problems
        n = self.slot_count
        if self.free_start < HEADER_SIZE or self.free_end > self.page_size \
                or self.free_start > self.free_end:
            problems.append("free_space_pointers_inconsistent")
        try:
            recs = self.records()
        except PageCorrupt as exc:
            problems.append(str(exc))
            return problems
        keys = [k for k, _ in recs]
        if keys != sorted(keys):
            problems.append("keys_not_sorted")
        if self.free_start != HEADER_SIZE + sum(len(r) for _, r in recs):
            problems.append("free_start_mismatch")
        if self.free_end != self.page_size - SLOT_SIZE * n:
            problems.append("free_end_mismatch")
        return problems
