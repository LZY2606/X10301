"""Append-only write-ahead log plus crash/replay model.

Durability order on every logical operation:

1. append BEGIN, UPDATE/ALLOC records (each UPDATE carries the full
   *before* and *after* page image, so reviewers can inspect the page
   pre-image and recovery can undo);
2. flush the WAL;
3. apply the cached page images to the page store;
4. append COMMIT and flush the WAL again.

A "write-page interruption" fault is injected between steps 2 and 3 for one
specific page: the page store receives a *torn* (zero-tail) image, which is
exactly the reachable half-page the crash invariants forbid.  Recovery
analyses the log, undoes transactions with no COMMIT, redos committed page
writes whose page LSN is older than the record LSN, and is safe to run
repeatedly (replay is idempotent).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .pages import Page, FRESH, DEFAULT_PAGE_SIZE, NONE

BEGIN = "BEGIN"
UPDATE = "UPDATE_PAGE"
ALLOC = "ALLOC_PAGE"
ROOT = "SET_ROOT"
COMMIT = "COMMIT"
CKPT = "CHECKPOINT"


@dataclass
class WalRecord:
    lsn: int
    kind: str
    txn: int
    payload: dict[str, Any]
    prev_lsn: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"lsn": self.lsn, "kind": self.kind, "txn": self.txn,
                "prev_lsn": self.prev_lsn, "payload": self.payload}


class PageStore:
    """Persisted page images.  A torn write truncates bytes to zeroes."""

    def __init__(self, page_size: int = DEFAULT_PAGE_SIZE):
        self.page_size = page_size
        self.pages: dict[int, bytes] = {}

    def write(self, pid: int, raw: bytes, torn: bool = False,
              torn_at: Optional[int] = None) -> None:
        assert len(raw) == self.page_size
        if torn:
            cut = self.page_size // 2 if torn_at is None else torn_at
            image = raw[:cut] + b"\x00" * (self.page_size - cut)
        else:
            image = raw
        self.pages[pid] = image

    def read(self, pid: int) -> Optional[bytes]:
        return self.pages.get(pid)

    def snapshot(self) -> dict[int, bytes]:
        return dict(self.pages)

    def load_snapshot(self, snap: dict[int, bytes]) -> None:
        self.pages = dict(snap)


class Wal:
    def __init__(self):
        self.records: list[WalRecord] = []

    def append(self, kind: str, txn: int, payload: dict[str, Any]) -> WalRecord:
        lsn = len(self.records) + 1
        prev = self.records[-1].lsn if self.records else 0
        rec = WalRecord(lsn=lsn, kind=kind, txn=txn,
                        payload=payload, prev_lsn=prev)
        self.records.append(rec)
        return rec

    def to_json(self) -> str:
        return json.dumps(
            {"records": [r.to_dict() for r in self.records]},
            separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Wal":
        wal = cls()
        for r in json.loads(text)["records"]:
            wal.records.append(WalRecord(**r))
        return wal

    def as_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]


def _img(page: Page) -> dict[str, Any]:
    return {"bytes": page.serialize().hex()}


def _page_from_payload(payload: dict[str, Any],
                       page_size: int) -> Page:
    return Page.parse(bytes.fromhex(payload["image"]["bytes"]), page_size)


def recover(records: list[WalRecord], store: PageStore,
            root_pid: int, page_size: int = DEFAULT_PAGE_SIZE) -> dict[str, Any]:
    """Analyse + undo + redo.  Returns a structured recovery report.

    Safe to run repeatedly: re-applying the same log to an already
    recovered store produces an identical snapshot.
    """
    committed: set[int] = set()
    for rec in records:
        if rec.kind == COMMIT:
            committed.add(rec.txn)

    # ---- undo phase: drop effects of transactions that never committed,
    #      restoring their before-images (FRESH allocation -> removal).
    undone_writes: list[WalRecord] = []
    for rec in records:
        if rec.kind in (UPDATE, ALLOC) and rec.txn not in committed:
            undone_writes.append(rec)
    for rec in reversed(undone_writes):
        payload = rec.payload
        if rec.kind == ALLOC:
            store.pages.pop(payload["pid"], None)
        else:
            before = payload.get("before")
            if before is None:
                store.pages.pop(payload["pid"], None)
            else:
                store.write(payload["pid"], bytes.fromhex(before["bytes"]))

    # ---- redo phase: replay committed writes whose on-disk LSN is stale.
    redone = 0
    for rec in records:
        if rec.txn not in committed:
            continue
        if rec.kind == ALLOC:
            if rec.payload["pid"] not in store.pages:
                store.write(rec.payload["pid"],
                            bytes.fromhex(rec.payload["image"]["bytes"]))
                redone += 1
        elif rec.kind == UPDATE:
            raw = store.read(rec.payload["pid"])
            on_disk_lsn = -1
            if raw is not None:
                try:
                    on_disk_lsn = Page.parse(raw, page_size).page_lsn
                except ValueError:
                    on_disk_lsn = -1  # torn / unreadable image
            if on_disk_lsn < rec.lsn:
                store.write(rec.payload["pid"],
                            bytes.fromhex(rec.payload["after"]["bytes"]))
                redone += 1

    # ---- resolve root from the latest committed SET_ROOT
    new_root = root_pid
    for rec in records:
        if rec.kind == ROOT and rec.txn in committed:
            new_root = rec.payload["root"]

    digest = {pid: raw.hex() for pid, raw in sorted(store.pages.items())}
    return {
        "committed_txns": sorted(committed),
        "undone_records": [r.lsn for r in undone_writes],
        "redone_records": redone,
        "root": new_root,
        "digest": digest,
    }
