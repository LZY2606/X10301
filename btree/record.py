"""Write-ahead log records.

Record kinds::

    BEGIN txn
    COMMIT txn
    ALLOC_PAGE pid, kind
    UPDATE_PAGE pid, lsn_from, before_image, after_image
    FREE_PAGE pid, before_image
    CLR pid, undo_lsn, image   (compensation, idempotent redo)

Every page-mutating record carries the full before/after image, so a replay can
reconstruct both pre-images and torn-write detection without secondary stores.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

MAGIC = b"WALR"

BEGIN = "BEGIN"
COMMIT = "COMMIT"
ALLOC_PAGE = "ALLOC_PAGE"
UPDATE_PAGE = "UPDATE_PAGE"
FREE_PAGE = "FREE_PAGE"
CLR = "CLR"


@dataclass
class LogRecord:
    lsn: int
    kind: str
    txn: int
    pid: int = -1
    prev_lsn: int = -1
    undo_lsn: int = -1
    page_lsn_from: int = -1
    before: Optional[bytes] = None
    after: Optional[bytes] = None
    page_kind: int = 0
    payload: dict = field(default_factory=dict)

    def encode(self) -> bytes:
        head = bytearray(MAGIC)
        body = _encode_body(self)
        head += struct.pack(">Q", self.lsn)
        head += self.kind.encode().ljust(11, b"\0")[:11]
        head += struct.pack(">iiqii", self.txn, self.pid, self.prev_lsn,
                            self.undo_lsn, self.page_lsn_from)
        head += bytes([self.page_kind])
        head += _enc_json(self.payload)
        head += struct.pack(">I", len(body))
        out = bytes(head) + body
        return out

    @property
    def is_page_record(self) -> bool:
        return self.kind in (UPDATE_PAGE, FREE_PAGE, CLR, ALLOC_PAGE)


def decode_record(buf: bytes, offset: int) -> tuple[LogRecord, int]:
    if buf[offset:offset + 4] != MAGIC:
        raise ValueError("bad WAL magic at %d" % offset)
    pos = offset + 4
    (lsn,) = struct.unpack_from(">Q", buf, pos)
    pos += 8
    kind = bytes(buf[pos:pos + 11]).rstrip(b"\0").decode()
    pos += 11
    txn, pid, prev_lsn, undo_lsn, page_lsn_from = struct.unpack_from(">iiqii", buf, pos)
    pos += 4 * 4 + 8
    page_kind = buf[pos]
    pos += 1
    payload, used = _dec_json(buf, pos)
    pos += used
    (body_len,) = struct.unpack_from(">I", buf, pos)
    pos += 4
    body = buf[pos:pos + body_len]
    pos += body_len
    before = after = None
    if body:
        if kind == UPDATE_PAGE:
            a, used_a = _take_bytes(body, 0)
            b, _ = _take_bytes(body, used_a)
            before, after = a, b
        elif kind in (FREE_PAGE, CLR):
            before, _ = _take_bytes(body, 0)
        elif kind == ALLOC_PAGE:
            after, _ = _take_bytes(body, 0)
    return LogRecord(lsn=lsn, kind=kind, txn=txn, pid=pid, prev_lsn=prev_lsn,
                     undo_lsn=undo_lsn, page_lsn_from=page_lsn_from,
                     before=before, after=after, page_kind=page_kind,
                     payload=payload), pos


def _take_bytes(buf: bytes, pos: int) -> tuple[bytes, int]:
    (n,) = struct.unpack_from(">I", buf, pos)
    pos += 4
    return buf[pos:pos + n], pos + n


def _put_bytes(b: bytes) -> bytes:
    return struct.pack(">I", len(b)) + b


def _encode_body(rec: LogRecord) -> bytes:
    if rec.kind == UPDATE_PAGE:
        assert rec.before is not None and rec.after is not None
        return _put_bytes(rec.before) + _put_bytes(rec.after)
    if rec.kind in (FREE_PAGE, CLR):
        assert rec.before is not None
        return _put_bytes(rec.before)
    if rec.kind == ALLOC_PAGE:
        assert rec.after is not None
        return _put_bytes(rec.after)
    return b""


def _enc_json(payload: dict) -> bytes:
    import json
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return struct.pack(">I", len(raw)) + raw


def _dec_json(buf: bytes, pos: int) -> tuple[dict, int]:
    import json
    (n,) = struct.unpack_from(">I", buf, pos)
    pos += 4
    return json.loads(buf[pos:pos + n].decode()), pos + 4 + n


def encode_all(records: list[LogRecord]) -> bytes:
    return b"".join(r.encode() for r in records)


def decode_all(buf: bytes) -> list[LogRecord]:
    out: list[LogRecord] = []
    pos = 0
    while pos < len(buf):
        rec, pos = decode_record(buf, pos)
        out.append(rec)
    return out
