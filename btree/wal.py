"""Write-ahead log + paged store with crash/replay simulation.

Durability order models ARIES-style WAL:
1. append BEGIN and PAGE before-images/after-images to the WAL (flush);
2. write pages (a crash may be injected mid-write -> torn/half page);
3. append COMMIT (or ABORT) and publish the new root/next_page_id.

Recovery is redo/undo based on page LSNs and is idempotent: running it a
second time changes no reachable page.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .encoding import (
    FLAG_HALF_PAGE,
    NO_PAGE,
    Page,
    PageFormatError,
)

REC_BEGIN = "BEGIN"
REC_PAGE = "PAGE"
REC_COMMIT = "COMMIT"
REC_ABORT = "ABORT"


class CrashInjected(Exception):
    """Raised in-process when a crash point fires; the store is 'dead'."""


@dataclass
class WalRecord:
    lsn: int
    tx: int
    kind: str
    payload: dict = field(default_factory=dict)

    def encode(self) -> bytes:
        body = json.dumps(
            {"lsn": self.lsn, "tx": self.tx, "kind": self.kind,
             "payload": self.payload}, separators=(",", ":")).encode()
        return struct.pack("<Q", len(body)) + body

    @classmethod
    def decode(cls, buf: bytes, pos: int):
        (length,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        data = json.loads(buf[pos:pos + length])
        return cls(lsn=data["lsn"], tx=data["tx"], kind=data["kind"],
                   payload=data["payload"]), pos + length


@dataclass
class CrashPoint:
    """Where the next committed write crashes.

    at: "before_commit" after pages written but before COMMIT (clean pages,
        transaction never commits -> undo on recovery)
        "during_write" the first page write is interrupted (torn half page)
    """

    at: str = "during_write"
    fired: bool = False


class Store:
    def __init__(self, page_size: int = 256):
        self.page_size = page_size
        self.disk: Dict[int, bytes] = {}
        self.wal: List[WalRecord] = []
        self.lsn = 0
        self.next_page_id = 0
        self.root_page_id = NO_PAGE
        self.meta_lsn = 0  # lsn of last committed meta (root/id allocator)
        self._tx = 0
        self.alive = True
        # bookkeeping for tests / archive
        self.recoveries = 0

    # ------------------------------------------------------------- allocator
    def allocate_page_id(self) -> int:
        pid = self.next_page_id
        self.next_page_id += 1
        return pid

    def _new_lsn(self) -> int:
        self.lsn += 1
        return self.lsn

    def _append(self, tx: int, kind: str, payload: dict) -> WalRecord:
        rec = WalRecord(self._new_lsn(), tx, kind, payload)
        self.wal.append(rec)
        return rec

    # ------------------------------------------------------------- write tx
    def begin(self) -> int:
        self._tx += 1
        self._append(self._tx, REC_BEGIN, {})
        return self._tx

    def write_page(self, tx: int, before: Optional[Page], after: Page) -> None:
        """Log before/after image then (conceptually) flush the page."""
        payload = {
            "page_id": after.page_id,
            "after_lsn": after.lsn,
            "after_b64": _b64(after.encode()),
        }
        if before is not None:
            payload["before_lsn"] = before.lsn
            payload["before_b64"] = _b64(before.encode())
        else:
            payload["before_lsn"] = None
            payload["before_b64"] = None
        self._append(tx, REC_PAGE, payload)

    def write_page_torn(self, tx: int, before: Optional[Page],
                        after: Page) -> None:
        """Log the change but persist only a prefix (CRC fails): half page."""
        self.write_page(tx, before, after)
        raw = after.encode()
        cut = max(40, len(raw) // 2)
        self.disk[after.page_id] = raw[:cut] + b"\x00" * (len(raw) - cut)

    def materialize(self, page: Page) -> None:
        self.disk[page.page_id] = page.encode()

    def commit(self, tx: int, root_page_id: int, next_page_id: int,
               crash: Optional[CrashPoint] = None) -> None:
        if crash is not None and crash.at == "during_write":
            crash.fired = True
            self.alive = False
            raise CrashInjected("write-page interruption before COMMIT")
        self._append(tx, REC_COMMIT,
                     {"root_page_id": root_page_id,
                      "next_page_id": next_page_id})
        if crash is not None and crash.at == "before_commit":
            crash.fired = True
            self.alive = False
            raise CrashInjected("crash immediately before COMMIT")
        self.root_page_id = root_page_id
        self.meta_lsn = self.lsn
        self.next_page_id = next_page_id

    def abort(self, tx: int) -> None:
        self._append(tx, REC_ABORT, {})

    # ----------------------------------------------------------------- reads
    def read_page(self, page_id: int) -> Page:
        raw = self.disk.get(page_id)
        if raw is None:
            raise PageFormatError(f"page {page_id} not on disk")
        return Page.decode(raw)

    def page_is_torn(self, page_id: int) -> bool:
        try:
            self.read_page(page_id)
            return False
        except PageFormatError:
            return True

    def snapshot(self) -> dict:
        return {
            "disk": {str(k): _b64(v) for k, v in self.disk.items()},
            "wal": [r.encode().hex() for r in self.wal],
            "lsn": self.lsn, "next_page_id": self.next_page_id,
            "root_page_id": self.root_page_id, "meta_lsn": self.meta_lsn,
            "_tx": self._tx,
        }

    # ------------------------------------------------------------- recovery
    def recover(self) -> dict:
        """Analysis + redo committed winners + undo losers. Idempotent."""
        self.recoveries += 1
        records = list(self.wal)
        winners: Dict[int, WalRecord] = {}
        aborted: set = set()
        began: set = set()
        for rec in records:
            if rec.kind == REC_BEGIN:
                began.add(rec.tx)
            elif rec.kind == REC_COMMIT:
                winners[rec.tx] = rec
            elif rec.kind == REC_ABORT:
                aborted.add(rec.tx)

        # Persist undo markers for losers that never got an ABORT so a second
        # recovery run sees the same outcome (recovery idempotency).
        losers = began - winners.keys() - aborted
        for tx in sorted(losers):
            self._append(tx, REC_ABORT, {"recovery": self.recoveries})
        aborted |= losers

        redone = []
        # Redo pass: every PAGE record of a winner is reapplied if the on-disk
        # page LSN is behind (torn pages simply get the full after-image).
        for rec in records:
            if rec.kind != REC_PAGE or rec.tx not in winners:
                continue
            pid = rec.payload["page_id"]
            after_lsn = rec.payload["after_lsn"]
            current = self.disk.get(pid)
            current_lsn = None
            if current is not None:
                try:
                    current_lsn = Page.decode(current).lsn
                except PageFormatError:
                    current_lsn = None
            if current_lsn is None or current_lsn < after_lsn:
                self.disk[pid] = _unb64(rec.payload["after_b64"])
                redone.append((pid, after_lsn))

        # Undo pass: losers' pages return to their before-images (in reverse).
        undone = []
        for rec in reversed(records):
            if rec.kind != REC_PAGE or rec.tx not in aborted:
                continue
            pid = rec.payload["page_id"]
            if rec.payload["before_b64"] is None:
                self.disk.pop(pid, None)
            else:
                self.disk[pid] = _unb64(rec.payload["before_b64"])
            undone.append(pid)

        # Publish meta of the last committed transaction.
        if winners:
            last = winners[max(winners, key=lambda t: winners[t].lsn)]
            self.root_page_id = last.payload["root_page_id"]
            self.next_page_id = last.payload["next_page_id"]
            self.meta_lsn = last.lsn

        self.alive = True
        return {"redone": redone, "undone": undone,
                "losers": sorted(losers), "winners": sorted(winners)}


def _b64(raw: bytes) -> str:
    import base64
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    import base64
    return base64.b64decode(text.encode("ascii"))
