"""ARIES-style recovery: torn-page detection, redo, undo, idempotent replay."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from . import page as P
from .pager import Pager, META_PID, decode_meta
from .record import (LogRecord, BEGIN, COMMIT, ALLOC_PAGE, UPDATE_PAGE,
                     FREE_PAGE, CLR)


def state_hash(disk: dict[int, bytes]) -> str:
    h = hashlib.sha256()
    for pid in sorted(disk):
        h.update(pid.to_bytes(4, "big"))
        h.update(disk[pid])
    return h.hexdigest()[:16]


def reachable_pages(pager: Pager) -> dict:
    """Traverse from the on-disk meta root. Half pages fail checksum."""
    meta_frame = pager.disk.get(META_PID)
    result = {"meta_valid": False, "root_pid": None, "pages": [],
              "torn": [], "missing": [], "traversal_ok": True, "errors": []}
    if meta_frame is None:
        result["errors"].append("meta page absent")
        result["traversal_ok"] = False
        return result
    meta = decode_meta(meta_frame)
    result["meta_valid"] = meta["valid"]
    if not meta["valid"]:
        result["errors"].append("meta checksum invalid")
        result["traversal_ok"] = False
        return result
    result["root_pid"] = meta["root_pid"]
    stack = [meta["root_pid"]]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        frame = pager.disk.get(pid)
        if frame is None:
            result["missing"].append(pid)
            result["errors"].append(f"reachable page {pid} missing")
            result["traversal_ok"] = False
            continue
        if not P.valid_checksum(frame):
            result["torn"].append(pid)
            result["errors"].append(f"reachable page {pid} checksum invalid")
            result["traversal_ok"] = False
            continue
        result["pages"].append(pid)
        h = P.header(frame)
        if h.kind == P.KIND_INTERNAL:
            for _, cell in P.cell_bytes(frame):
                d = P.decode_internal_cell(cell)
                stack.append(d["child"])
    result["pages"].sort()
    result["torn"].sort()
    result["missing"].sort()
    return result


@dataclass
class RecoveryReport:
    redo_applied: list[int] = field(default_factory=list)
    clr_applied: list[int] = field(default_factory=list)
    new_clr_lsns: list[int] = field(default_factory=list)
    pre_reachable_torn: list[int] = field(default_factory=list)
    pre_reachable_missing: list[int] = field(default_factory=list)
    post_reachable: list[int] = field(default_factory=list)
    post_torn: list[int] = field(default_factory=list)
    traversal_ok: bool = True
    hash_before: str = ""
    hash_after: str = ""
    committed_txns: list[int] = field(default_factory=list)
    aborted_txns: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _committed_txns(records: list[LogRecord]) -> set[int]:
    return {r.txn for r in records if r.kind == COMMIT}


def recover(pager: Pager, append_clr: bool = True) -> RecoveryReport:
    report = RecoveryReport()
    report.hash_before = state_hash(pager.disk)
    pre = reachable_pages(pager)
    report.pre_reachable_torn = list(pre["torn"])
    report.pre_reachable_missing = list(pre["missing"])

    records = pager.records
    committed = _committed_txns(records)
    report.committed_txns = sorted(committed)

    # ---- analysis + redo: LSN order, install/redo with LSN gate ----
    for rec in records:
        if rec.kind in (BEGIN, COMMIT):
            continue
        if rec.kind == CLR:
            frame = pager.disk.get(rec.pid)
            if frame is not None and P.valid_checksum(frame) \
                    and P.header(frame).page_lsn >= rec.lsn:
                continue
            if rec.before is not None:
                pager.disk[rec.pid] = rec.before
            else:
                pager.disk.pop(rec.pid, None)
            report.clr_applied.append(rec.lsn)
            continue
        if rec.kind == ALLOC_PAGE:
            frame = pager.disk.get(rec.pid)
            if frame is not None and P.valid_checksum(frame) \
                    and P.header(frame).page_lsn >= rec.lsn:
                continue
            pager.disk[rec.pid] = rec.after
            report.redo_applied.append(rec.lsn)
        elif rec.kind == UPDATE_PAGE:
            frame = pager.disk.get(rec.pid)
            if frame is not None and P.valid_checksum(frame) \
                    and P.header(frame).page_lsn >= rec.lsn:
                continue
            pager.disk[rec.pid] = rec.after
            report.redo_applied.append(rec.lsn)
        elif rec.kind == FREE_PAGE:
            if rec.pid in pager.disk:
                pager.disk.pop(rec.pid, None)
                report.redo_applied.append(rec.lsn)

    # ---- undo uncommitted transactions in reverse prev_lsn order ----
    by_txn: dict[int, list[LogRecord]] = {}
    for rec in records:
        if rec.txn not in committed and rec.kind not in (BEGIN, COMMIT):
            by_txn.setdefault(rec.txn, []).append(rec)
    last_lsn_per_txn = {}
    for rec in records:
        last_lsn_per_txn[rec.txn] = rec.lsn
    for txn_id, recs in by_txn.items():
        report.aborted_txns.append(txn_id)
        cur = last_lsn_per_txn.get(txn_id)
        by_lsn = {r.lsn: r for r in recs}
        seen = set()
        while cur and cur > 0 and cur not in seen:
            seen.add(cur)
            rec = by_lsn.get(cur)
            if rec is None:
                break
            if rec.kind == UPDATE_PAGE:
                pager.disk[rec.pid] = rec.before
                _clr(pager, rec, rec.before, append_clr, report)
            elif rec.kind == ALLOC_PAGE:
                pager.disk.pop(rec.pid, None)
                _clr(pager, rec, None, append_clr, report)
            elif rec.kind == FREE_PAGE:
                pager.disk[rec.pid] = rec.before
                _clr(pager, rec, rec.before, append_clr, report)
            cur = rec.prev_lsn

    post = reachable_pages(pager)
    report.post_reachable = post["pages"]
    report.post_torn = post["torn"]
    report.traversal_ok = post["traversal_ok"]
    report.hash_after = state_hash(pager.disk)
    return report


def _clr(pager: Pager, rec: LogRecord, image: bytes | None,
         append_clr: bool, report: RecoveryReport) -> None:
    if not append_clr:
        return
    lsn = pager.next_lsn()
    clr = LogRecord(lsn=0, kind=CLR, txn=rec.txn, pid=rec.pid,
                    prev_lsn=pager._lsn, undo_lsn=rec.lsn, before=image)
    pager.append_log(clr)
    report.new_clr_lsns.append(lsn)
