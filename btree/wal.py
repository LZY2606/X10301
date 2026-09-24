"""预写日志（WAL）。

日志序号 ``lsn`` 即记录在 WAL 中的字节偏移；每条页写入都带页前镜像，
因此回滚（未提交事务）与重做（已提交事务）都是确定性的，恢复可重复重放。
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

FRAME_FORMAT = ">IIIIiI"  # txn, lsn, rec_type, payload_len, crc_padding?
# 设计为：txn_id(I) lsn(I) rec_type(I) plen(I) checksum(I) -> 20 字节
FRAME_FORMAT = ">IIII I".replace(" ", "")
FRAME_SIZE = struct.calcsize(FRAME_FORMAT)

BEGIN = 1
PAGE_ALLOC = 2
PAGE_WRITE = 3
ROOT_SET = 4
COMMIT = 5
ABORT = 6


@dataclass
class Frame:
    txn_id: int
    lsn: int
    rec_type: int
    payload: bytes


def _frame(txn: int, lsn: int, rec_type: int, payload: bytes) -> bytes:
    head = struct.pack(FRAME_FORMAT, txn, lsn, rec_type, len(payload), 0)
    checksum = zlib.crc32(head + payload) & 0xFFFFFFFF
    return (
        struct.pack(FRAME_FORMAT, txn, lsn, rec_type, len(payload), checksum)
        + payload
    )


@dataclass
class WAL:
    buf: bytearray = field(default_factory=bytearray)

    def tell(self) -> int:
        return len(self.buf)

    def append(self, txn: int, rec_type: int, payload: bytes) -> int:
        lsn = len(self.buf)
        self.buf += _frame(txn, lsn, rec_type, payload)
        return lsn

    def begin(self, txn: int) -> int:
        return self.append(txn, BEGIN, b"")

    def alloc(self, txn: int, pid: int) -> int:
        return self.append(txn, PAGE_ALLOC, struct.pack(">i", pid))

    def write_page(
        self,
        txn: int,
        pid: int,
        before: bytes | None,
        after: bytes,
    ) -> int:
        payload = (
            struct.pack(">i", pid)
            + struct.pack(">I", 0 if before is None else 1)
            + (b"" if before is None else before)
            + after
        )
        return self.append(txn, PAGE_WRITE, payload)

    def root_set(self, txn: int, pid: int) -> int:
        return self.append(txn, ROOT_SET, struct.pack(">i", pid))

    def commit(self, txn: int) -> int:
        return self.append(txn, COMMIT, b"")

    def abort(self, txn: int) -> int:
        return self.append(txn, ABORT, b"")

    def frames(self) -> list[Frame]:
        out: list[Frame] = []
        pos = 0
        while pos < len(self.buf):
            if pos + FRAME_SIZE > len(self.buf):
                break  # 截断的尾帧：WAL 写中断
            txn, lsn, rec_type, plen, checksum = struct.unpack(
                FRAME_FORMAT, self.buf[pos : pos + FRAME_SIZE]
            )
            end = pos + FRAME_SIZE + plen
            if end > len(self.buf):
                break  # 截断的负载
            payload = bytes(self.buf[pos + FRAME_SIZE : end])
            head = struct.pack(
                FRAME_FORMAT, txn, lsn, rec_type, plen, 0
            )
            if (zlib.crc32(head + payload) & 0xFFFFFFFF) != checksum:
                break  # WAL 帧损坏：停在此处之前
            if lsn != pos:
                break
            out.append(Frame(txn, lsn, rec_type, payload))
            pos = end
        return out


def parse_page_write(payload: bytes) -> dict:
    pid = struct.unpack(">i", payload[:4])[0]
    had_before = struct.unpack(">I", payload[4:8])[0]
    idx = 8
    before = None
    if had_before:
        raise ValueError("页大小未知，应使用 parse_page_write_sized")
    after = payload[idx:]
    return {"pid": pid, "before": before, "after": after}


def parse_page_write_sized(payload: bytes, page_size: int) -> dict:
    pid = struct.unpack(">i", payload[:4])[0]
    had_before = struct.unpack(">I", payload[4:8])[0]
    idx = 8
    before = None
    if had_before:
        before = bytes(payload[idx : idx + page_size])
        idx += page_size
    after = bytes(payload[idx : idx + page_size])
    return {"pid": pid, "before": before, "after": after}
