"""Write-ahead log, crash points and idempotent recovery.

Every logical operation appends log records with a monotone LSN
(unit/basis: "lsn"/"log_sequence").  Page images in the log are the source
of truth: recovery replays page_write records onto a fresh disk image.
Replay is idempotent -- a record is applied only when its LSN is newer
than the LSN already materialized for that page, so replaying the same
prefix twice yields byte-identical state (R4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pages import Page, image_checksum, validate_image


@dataclass
class LogRecord:
    lsn: int
    kind: str            # op_begin | page_write | op_commit | page_free
    page_id: int = 0
    image: bytes = b""
    checksum: int = 0
    note: str = ""

    def to_json(self) -> dict:
        return {
            "lsn": {"value": self.lsn, "unit": "lsn", "basis": "log_sequence"},
            "kind": self.kind,
            "page_id": self.page_id,
            "image": self.image.hex(),
            "checksum": self.checksum,
            "note": self.note,
        }


class WAL:
    def __init__(self):
        self.records: list[LogRecord] = []
        self._next_lsn = 1

    def append(self, kind: str, page_id: int = 0, image: bytes = b"", note: str = "") -> LogRecord:
        rec = LogRecord(
            lsn=self._next_lsn, kind=kind, page_id=page_id,
            image=image, checksum=image_checksum(image) if image else 0, note=note,
        )
        self._next_lsn += 1
        self.records.append(rec)
        return rec

    @property
    def next_lsn(self) -> int:
        return self._next_lsn


class DiskImage:
    """Materialized page images keyed by page id, each with the applied LSN."""

    def __init__(self, page_size: int):
        self.page_size = page_size
        self.pages = {}   # page_id -> bytes
        self.page_lsn = {}  # page_id -> lsn last applied

    def write(self, page_id: int, image: bytes, lsn: int) -> None:
        self.pages[page_id] = image
        self.page_lsn[page_id] = lsn

    def torn_write(self, page_id: int, image: bytes, lsn: int) -> None:
        """Simulate an interrupted page write: only the first half lands."""
        half = len(image) // 2
        self.pages[page_id] = image[:half] + bytes(len(image) - half)
        self.page_lsn[page_id] = lsn

    def digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for pid in sorted(self.pages):
            h.update(pid.to_bytes(4, "big"))
            h.update(self.pages[pid])
        return h.hexdigest()


def replay(wal: WAL, page_size: int, upto_lsn: int | None = None) -> DiskImage:
    """Rebuild the disk image from the log; idempotent by LSN."""
    disk = DiskImage(page_size)
    for rec in wal.records:
        if upto_lsn is not None and rec.lsn > upto_lsn:
            break
        if rec.kind == "page_write":
            if rec.lsn > disk.page_lsn.get(rec.page_id, 0):
                disk.write(rec.page_id, rec.image, rec.lsn)
        elif rec.kind == "page_free":
            if rec.lsn > disk.page_lsn.get(rec.page_id, 0):
                disk.pages.pop(rec.page_id, None)
                disk.page_lsn[rec.page_id] = rec.lsn
    return disk


def reachable_pages(disk: DiskImage, root_page_id: int) -> dict:
    """Walk from the root over the *recovered* images.

    Returns {"reachable": [...], "corrupt": {page_id: [diagnostics]}}.
    A crash point must never leave a reachable half page (R3).
    """
    reachable = []
    corrupt = {}
    stack = [root_page_id]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen or pid not in disk.pages:
            continue
        seen.add(pid)
        reachable.append(pid)
        diags = validate_image(disk.pages[pid], disk.page_size)
        if diags:
            corrupt[pid] = diags
            continue
        page = Page.from_bytes(disk.pages[pid], disk.page_size)
        if page.ptype == 2:
            if page.left_child:
                stack.append(page.left_child)
            stack.extend(c.child for c in page.cells)
        else:
            if page.right_sibling:
                stack.append(page.right_sibling)
    return {"reachable": sorted(reachable), "corrupt": corrupt}
