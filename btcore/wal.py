"""Write-ahead log, crash points and crash recovery replay.

The execution model is intentionally small but explicit:

* every logical operation is a serial transaction:
  ``BEGIN -> WRITE* (ordered: split sibling, split page, parent/root)
  -> COMMIT``;
* every WRITE keeps the before-image of the page (or marks the pid as newly
  allocated) so a loser transaction can be physically undone;
* a crash interrupts the write sequence; ``page_write_torn`` additionally
  persists a partial (checksum-invalid) image;
* recovery runs analysis / redo / undo. The accepted strategy
  (``undo_loser``) restores before-images and drops freshly allocated
  pages, so no reachable half-page survives. Repeating recovery must be a
  no-op with an identical state hash (idempotence).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .pages import (
    LEAF,
    INTERNAL,
    TORN_FLAG,
    encode_page,
    decode_page,
    page_checksum,
    LeafPage,
    InternalPage,
    BasePage,
)
from .tree import BTree, SplitRecord, DuplicateKey

WAL_RECORD_TYPES = ("BEGIN", "WRITE", "COMMIT", "CLR", "ABORT", "CHECKPOINT")

CRASH_POINTS = (
    "split_new_page_written",
    "parent_not_updated",
    "commit_missing",
    "page_write_torn",
)


class Crash(Exception):
    """Raised in-memory when the scheduled crash fires."""

    def __init__(self, point: str, after_lsn: int, partial_pid: Optional[int] = None):
        super().__init__(point)
        self.point = point
        self.after_lsn = after_lsn
        self.partial_pid = partial_pid


@dataclass
class WriteIntent:
    pid: int
    image: bytes
    new_page: bool
    before_image: Optional[bytes]
    role: str  # 'sibling' | 'split_page' | 'parent' | 'new_root' | 'leaf'


@dataclass
class WalRecord:
    seq: int
    type: str
    lsn: int
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        p = dict(self.payload)
        return {"seq": self.seq, "type": self.type, "lsn": self.lsn, "payload": p}


class WalStore:
    """Durable page images plus an append-only log (simulated storage)."""

    def __init__(self, page_size: int):
        self.page_size = page_size
        self.pages: Dict[int, bytes] = {}
        self.log: List[WalRecord] = []
        self._seq = 0

    def _append(self, rtype: str, lsn: int, **payload) -> WalRecord:
        rec = WalRecord(self._seq, rtype, lsn, payload)
        self._seq += 1
        self.log.append(rec)
        return rec

    def state_hash(self) -> str:
        h = hashlib.sha256()
        for pid in sorted(self.pages):
            h.update(str(pid).encode())
            h.update(b":")
            h.update(self.pages[pid])
            h.update(b";")
        h.update(b"loglen=" + str(len(self.log)).encode())
        return h.hexdigest()

    def torn_reachable(self, root_pid: int) -> List[int]:
        bad: List[int] = []

        def walk(pid: int) -> None:
            if pid not in self.pages or pid in seen:
                return
            seen.add(pid)
            image = self.pages[pid]
            stored, computed = page_checksum(image)
            flag = image[1] & TORN_FLAG
            if stored != computed or flag:
                bad.append(pid)
            kind = image[0]
            if kind == INTERNAL:
                import struct
                n = struct.unpack_from("<H", image, 2)[0]
                (hi,) = struct.unpack_from("<I", image, 16)
                for i in range(n):
                    (off,) = struct.unpack_from("<H", image, 32 + 2 * i)
                    (child,) = struct.unpack_from("<I", image, off)
                    walk(child)
                walk(hi)

        seen: set[int] = set()
        walk(root_pid)
        return bad

    def clone(self) -> "WalStore":
        other = WalStore(self.page_size)
        other.pages = {pid: bytes(img) for pid, img in self.pages.items()}
        other.log = [
            WalRecord(r.seq, r.type, r.lsn, dict(r.payload)) for r in self.log
        ]
        other._seq = self._seq
        return other


def load_tree(store: WalStore, root_pid: int, dup_policy: str) -> BTree:
    """Reconstruct a BTree from durable images (used after recovery)."""
    tree = BTree(page_size=store.page_size, dup_policy=dup_policy)
    tree.pages = {}
    max_pid = -1
    for pid, image in store.pages.items():
        page = decode_page(image, verify=False)
        tree.pages[pid] = page
        max_pid = max(max_pid, pid)
    tree._next_pid = max_pid + 1
    tree.root = root_pid
    return tree


@dataclass
class RecoveryReport:
    ran: bool
    loser_lsn: Optional[int]
    analysis: List[dict] = field(default_factory=list)
    redo: List[dict] = field(default_factory=list)
    undo: List[dict] = field(default_factory=list)
    torn_reachable_after: List[int] = field(default_factory=list)
    state_hash_before: str = ""
    state_hash_after: str = ""
    state_hash_second: str = ""
    idempotent: Optional[bool] = None
    strategy: str = "undo_loser"


def _loser_range(log: List[WalRecord]) -> Optional[tuple[int, int, int]]:
    """Return (begin_index, lsn, end_index) for the trailing uncommitted txn."""
    last_commit = -1
    for i, rec in enumerate(log):
        if rec.type in ("COMMIT", "ABORT"):
            last_commit = i
    begin = -1
    for i in range(last_commit + 1, len(log)):
        if log[i].type == "BEGIN":
            begin = i
            break
    if begin < 0:
        return None
    return begin, log[begin].lsn, len(log) - 1


def recover(
    store: WalStore,
    root_pid_before: int,
    root_pid_after: Optional[int],
    strategy: str = "undo_loser",
) -> RecoveryReport:
    """Analysis / redo / undo. ``root_pid_after`` is the committed root if any.

    With ``leave_writes`` the loser writes are intentionally kept; that
    variant is only used to materialize the rejected counter-example.
    """
    h_before = store.state_hash()
    rng = _loser_range(store.log)
    report = RecoveryReport(ran=False, loser_lsn=None,
                            state_hash_before=h_before, strategy=strategy)
    report.analysis.append({"phase": "scan", "records": len(store.log)})
    if rng is None:
        report.analysis.append({"phase": "analysis", "loser": None})
        report.state_hash_after = store.state_hash()
        report.state_hash_second = store.state_hash()
        report.idempotent = True
        report.torn_reachable_after = store.torn_reachable(root_pid_before)
        return report

    report.ran = True
    begin_idx, loser_lsn, end_idx = rng
    report.loser_lsn = loser_lsn
    report.analysis.append({"phase": "analysis", "loser_lsn": loser_lsn,
                            "begin_seq": store.log[begin_idx].seq})

    # Redo: committed writes are already reflected; verify byte equality.
    for rec in store.log[:begin_idx]:
        if rec.type == "WRITE":
            pid = rec.payload["pid"]
            want = rec.payload["image"]
            now = store.pages.get(pid)
            report.redo.append({
                "seq": rec.seq, "pid": pid,
                "already_applied": now == want,
            })

    if strategy == "undo_loser":
        for rec in store.log[begin_idx : end_idx + 1]:
            if rec.type != "WRITE":
                continue
            pid = rec.payload["pid"]
            if rec.payload.get("new_page"):
                store.pages.pop(pid, None)
                report.undo.append({"seq": rec.seq, "pid": pid,
                                    "action": "drop_new_page"})
            else:
                store.pages[pid] = rec.payload["before_image"]
                report.undo.append({"seq": rec.seq, "pid": pid,
                                    "action": "restore_before_image"})
        for rec in reversed(store.log[begin_idx : end_idx + 1]):
            if rec.type == "WRITE":
                store._append("CLR", loser_lsn, pid=rec.payload["pid"],
                              undo_seq=rec.seq)
        store._append("ABORT", loser_lsn)
        final_root = root_pid_before
    else:  # leave_writes: intentionally unsound, used as rejected candidate
        final_root = root_pid_after if root_pid_after is not None else root_pid_before

    report.state_hash_after = store.state_hash()
    report.torn_reachable_after = store.torn_reachable(final_root)

    # Second replay: must be a no-op.
    second = recover(store, final_root, None, strategy="undo_loser")
    report.state_hash_second = second.state_hash_after
    report.idempotent = report.state_hash_after == report.state_hash_second
    return report


class TransactionalExecutor:
    """Applies logical ops to a BTree and a WalStore with crash scheduling."""

    def __init__(self, tree: BTree, store: WalStore):
        self.tree = tree
        self.store = store
        self.crashed: Optional[Crash] = None
        self._publish_initial()

    def _publish_initial(self) -> None:
        for pid, page in self.tree.pages.items():
            self.store.pages[pid] = encode_page(page, self.tree.page_size)
        self.store._append("CHECKPOINT", 0, root=self.tree.root,
                           pids=self.tree.allocated_pids())

    def _write(self, intent: WriteIntent, torn: bool = False) -> None:
        image = intent.image
        if torn:
            cut = self.store.page_size // 2
            partial = bytearray(self.store.page_size)
            partial[:cut] = image[:cut]  # trailing zero bytes -> bad checksum
            image = bytes(partial)
        self.store.pages[intent.pid] = image
        self.store._append(
            "WRITE", self.tree.lsn, pid=intent.pid, new_page=intent.new_page,
            before_image=intent.before_image, image=image, role=intent.role,
            torn=torn,
        )

    def apply(self, kind: str, key: bytes, value: Optional[bytes],
              crash: Optional[str] = None) -> dict:
        """Execute one logical op, optionally crashing mid-commit."""
        lsn = self.tree.lsn + 1
        pre_pages = self.tree.allocated_pids()
        pre_root = self.tree.root
        before_images = {pid: self.store.pages[pid]
                         for pid in list(self.tree.pages)}
        self.store._append("BEGIN", lsn, op=kind,
                           key=key.hex(),
                           value=None if value is None else value.hex(),
                           pids=list(pre_pages), root=pre_root)

        rejected = None
        splits: List[SplitRecord] = []
        try:
            if kind == "set":
                splits = self.tree.insert(key, value or b"")
            elif kind == "del":
                self.tree.delete(key)
            else:
                raise ValueError(kind)
        except DuplicateKey as exc:
            rejected = str(exc)
            self.store._append("ABORT", lsn, reason=rejected)
            return {"lsn": lsn, "splits": [], "rejected": rejected,
                    "crashed": False}

        # Build the physically ordered write plan from split propagation.
        intents = self._write_plan(splits, set(pre_pages), before_images)

        torn_done = False
        for idx, intent in enumerate(intents):
            torn_here = crash == "page_write_torn" and not torn_done
            if torn_here:
                torn_done = True
            self._write(intent, torn=torn_here)
            if crash == "split_new_page_written" and intent.role == "sibling":
                self.crashed = Crash(crash, lsn)
                raise self.crashed
            if (crash == "parent_not_updated" and intent.role == "split_page"
                    and any(s.level in ("leaf", "root") for s in splits)):
                # sibling + split page durable, parent linkage not yet written
                self.crashed = Crash(crash, lsn)
                raise self.crashed
            if crash == "page_write_torn" and torn_here:
                self.crashed = Crash(crash, lsn, partial_pid=intent.pid)
                raise self.crashed

        if crash == "commit_missing":
            self.crashed = Crash(crash, lsn)
            raise self.crashed

        self.store._append("COMMIT", lsn, root=self.tree.root,
                           pids=self.tree.allocated_pids())
        return {"lsn": lsn, "splits": splits, "rejected": None,
                "crashed": False, "root_after": self.tree.root}

    def _write_plan(self, splits: List[SplitRecord], pre_pages: set,
                    before_images: Dict[int, bytes]) -> List[WriteIntent]:
        plan: List[WriteIntent] = []
        split_pids = {s.split_pid for s in splits}
        new_pids = {s.new_pid for s in splits}
        # bottom-up: leaf sibling/leaf page first ...
        for rec in reversed(splits) if False else splits:
            plan.append(WriteIntent(
                pid=rec.new_pid, image=rec.after_right_image,
                new_page=rec.new_pid not in pre_pages,
                before_image=None, role="sibling"))
            plan.append(WriteIntent(
                pid=rec.split_pid, image=rec.after_left_image,
                new_page=False, before_image=before_images.get(rec.split_pid),
                role="split_page"))
        root_pid = self.tree.root
        for pid in sorted(self.tree._touched):
            if pid in split_pids or pid in new_pids:
                continue
            plan.append(WriteIntent(
                pid=pid,
                image=encode_page(self.tree.pages[pid], self.tree.page_size),
                new_page=pid not in pre_pages,
                before_image=before_images.get(pid),
                role="new_root" if pid == root_pid and pid not in pre_pages
                else "parent"))
        # non-splitting leaf/root write
        for pid in sorted(self.tree._touched):
            if pid in split_pids or pid in new_pids or pid in {i.pid for i in plan}:
                continue
            plan.append(WriteIntent(
                pid=pid,
                image=encode_page(self.tree.pages[pid], self.tree.page_size),
                new_page=pid not in pre_pages,
                before_image=before_images.get(pid), role="leaf"))
        return plan
