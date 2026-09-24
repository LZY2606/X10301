"""Write-ahead log records.

Every record carries a log sequence number (lsn). Page writes are physical
full-page after-images, which makes redo idempotent: replaying recovery any
number of times converges to the same bytes.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field, asdict


@dataclass
class WalRecord:
    lsn: int
    kind: str                 # page_write | alloc | dealloc | op_begin | op_commit
    page_id: int = 0
    image: bytes = b""        # page_write: full after-image
    op_index: int = -1        # op_begin/op_commit group tag
    data: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {"lsn": self.lsn, "kind": self.kind, "page_id": self.page_id,
             "op_index": self.op_index, "data": self.data,
             "image_b64": base64.b64encode(self.image).decode("ascii") if self.image else ""}
        return d

    @staticmethod
    def from_json(d: dict) -> "WalRecord":
        img = base64.b64decode(d["image_b64"]) if d.get("image_b64") else b""
        return WalRecord(lsn=d["lsn"], kind=d["kind"], page_id=d.get("page_id", 0),
                         image=img, op_index=d.get("op_index", -1), data=d.get("data", {}))


class Wal:
    def __init__(self):
        self.records: list[WalRecord] = []
        self.next_lsn = 1

    def append(self, kind: str, page_id: int = 0, image: bytes = b"",
               op_index: int = -1, data: dict | None = None) -> WalRecord:
        rec = WalRecord(lsn=self.next_lsn, kind=kind, page_id=page_id,
                        image=image, op_index=op_index, data=data or {})
        self.next_lsn += 1
        self.records.append(rec)
        return rec

    def committed_op_indexes(self) -> set[int]:
        return {r.op_index for r in self.records if r.kind == "op_commit"}

    def to_json(self) -> list[dict]:
        return [r.to_json() for r in self.records]
