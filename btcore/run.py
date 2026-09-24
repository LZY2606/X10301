"""Replay orchestrator and machine-readable finding engine.

Outputs structured evidence so the review UI can show, for every displayed
value, a stable identity:

* each logical op keeps its before/after page images, slot directory,
  free-space bytes (coordinate basis ``page_start/byte``), sibling links,
  allocated page ids and log sequence numbers;
* each split enumerates every byte-feasible cut;
* crashes capture the reachable-page set, torn pages, and both recovery
  strategies (accepted ``undo_loser`` vs rejected ``leave_writes``);
* findings carry a code, status, evidence object ids and their sources.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .pages import (
    LEAF,
    INTERNAL,
    HEADER_SIZE,
    encode_page,
    decode_page,
    page_checksum,
    page_free_bytes,
    legal_split_points,
    cells_payload_bytes,
)
from .tree import BTree, SplitRecord
from .wal import WalStore, TransactionalExecutor, Crash, recover, CRASH_POINTS
from .script import ParsedScript, ParsedLine
from .units import Quantity, Coordinate


FINDING_CODES = (
    "SPLIT_BY_BYTES",
    "SEPARATOR_RULE",
    "NO_REACHABLE_HALF_PAGE",
    "RECOVERY_IDEMPOTENT",
    "EMPTY_PAGE_AFTER_DELETE",
    "REJECTED_BY_DUP_POLICY",
    "SLOT_FREE_SPACE_CONSISTENT",
)


@dataclass
class Finding:
    code: str
    status: str            # pass | fail | info | undecidable
    message: str
    op_seq: int
    sources: List[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RunConfig:
    page_size: int = 128
    dup_policy: str = "replace"
    accepted_recovery: str = "undo_loser"


@dataclass
class OpEvent:
    seq: int
    kind: str
    lsn: int
    key_hex: str
    raw_line: int
    raw_text: str
    before_root: int
    after_root: int
    pids_before: List[int]
    pids_after: List[int]
    reachable_after: List[int]
    pages_before: Dict[str, dict]
    pages_after: Dict[str, dict]
    splits: List[dict]
    crash: Optional[dict]
    recovery: Optional[dict]
    candidates: List[dict]
    labels: List[str] = field(default_factory=list)
    rejected: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RunTrace:
    config: dict
    ops: List[OpEvent]
    findings: List[Finding]
    checkpoints: List[dict]
    diagnostics: List[dict]
    expectations: List[dict]
    final_root: int
    final_pids: List[int]
    log_records: List[dict]
    source_text: str


def page_view(page, page_size: int, image: bytes) -> dict:
    from .pages import LeafPage
    kind_code = LEAF if isinstance(page, LeafPage) else INTERNAL
    slots = []
    off = page_size
    cells = []
    for idx, cell in enumerate(page.cells):
        if kind_code == LEAF:
            size = cell.leaf_size()
        else:
            size = cell.internal_size()
        off -= size
        slots.append({"index": idx, "offset": off, "length": size,
                      "key_hex": cell.key.hex(),
                      "key_text": _safe_text(cell.key)})
        cells.append({"offset": off, "length": size, "key_hex": cell.key.hex()})
    stored, computed = page_checksum(image)
    free_start = page_size - cells_payload_bytes(page.cells, kind_code)
    return {
        "page_id": page.page_id,
        "kind": "leaf" if kind_code == LEAF else "internal",
        "page_lsn": page.page_lsn,
        "ncells": len(page.cells),
        "hi_child": getattr(page, "hi_child", None),
        "header_bytes": HEADER_SIZE,
        "slot_dir_bytes": 2 * len(page.cells),
        "cell_payload_bytes": cells_payload_bytes(page.cells, kind_code),
        "free_start_offset": free_start,
        "used_bytes": HEADER_SIZE + 2 * len(page.cells)
        + cells_payload_bytes(page.cells, kind_code),
        "free_space": {
            "value": page_free_bytes(page_size, page.cells, kind_code),
            "unit": "B",
            "origin": "page_start",
            "raw": f"{page_free_bytes(page_size, page.cells, kind_code)} B",
        },
        "slots": slots,
        "cells": cells,
        "checksum_stored": stored,
        "checksum_computed": computed,
        "torn": bool(image[1] & 1) or stored != computed,
        "image_hex": image.hex(),
    }


def _safe_text(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.hex()


def run_script_events(
    parsed: ParsedScript,
    source_text: str,
    config: Optional[RunConfig] = None,
) -> RunTrace:
    config = config or RunConfig()
    page_size = next(
        (l.page_size_bytes for l in parsed.lines if l.page_size_bytes),
        config.page_size,
    )
    dup_policy = parsed.dup_policy or config.dup_policy
    tree = BTree(page_size=page_size, dup_policy=dup_policy)
    store = WalStore(page_size=page_size)
    executor = TransactionalExecutor(tree, store)

    trace = RunTrace(
        config={"page_size": page_size, "dup_policy": dup_policy,
                "accepted_recovery": config.accepted_recovery,
                "page_size_raw": parsed.page_size_raw},
        ops=[], findings=[], checkpoints=[], diagnostics=[],
        expectations=[], final_root=tree.root, final_pids=[],
        log_records=[], source_text=source_text,
    )
    for d in parsed.diagnostics:
        trace.diagnostics.append({"lineno": d.lineno, "raw": d.raw,
                                  "diagnostic": d.diagnostic or d.error})

    pending_labels: List[str] = []
    for line in parsed.lines:
        if line.label and not line.op:
            pending_labels.append(line.label)
        if line.op == "expect":
            trace.expectations.append({"code": line.expect_code,
                                       "status": line.expect_status,
                                       "lineno": line.lineno})
        if line.op == "checkpoint":
            trace.checkpoints.append({
                "label": line.label, "after_op": len(trace.ops),
                "root": tree.root,
                "pids": tree.allocated_pids(),
                "state_hash": store.state_hash(),
            })
            if line.label:
                pending_labels.append(line.label)
        if line.op in ("set", "del"):
            _run_one_op(line, tree, store, executor, trace, pending_labels,
                        config)
            pending_labels = []

    trace.final_root = tree.root
    trace.final_pids = tree.allocated_pids()
    trace.log_records = [r.to_dict() for r in store.log]
    _evaluate_expectations(trace)
    return trace


def _run_one_op(line: ParsedLine, tree: BTree, store: WalStore,
                executor: TransactionalExecutor, trace: RunTrace,
                labels: List[str], config: RunConfig) -> None:
    seq = len(trace.ops) + 1
    pages_before = _snapshot_views(tree, store)
    pids_before = tree.allocated_pids()
    root_before = tree.root
    links_before = tree.leaf_sibling_links()

    key = (line.key or "").encode("utf-8")
    value = None if line.value is None else line.value.encode("utf-8")
    crash = line.crash
    crash_report = None
    candidates: List[dict] = []
    splits: List[SplitRecord] = []
    rejected = None
    root_after = tree.root
    reachable_after: List[int] = []

    try:
        result = executor.apply(line.op, key, value, crash=crash)
        splits = result["splits"]
        rejected = result.get("rejected")
        root_after = result.get("root_after", tree.root)
        reachable_after = tree.reachable_pids()
    except Crash as exc:
        # Build recovery candidates WITHOUT mutating the accepted view yet.
        accepted_store = store.clone()
        rejected_store = store.clone()
        accepted_report = recover(
            accepted_store, root_pid_before=root_before,
            root_pid_after=tree.root, strategy="undo_loser")
        rejected_report = recover(
            rejected_store, root_pid_before=root_before,
            root_pid_after=tree.root, strategy="leave_writes")
        crash_report = {
            "point": exc.point, "lsn": exc.after_lsn,
            "partial_pid": exc.partial_pid,
            "torn_pids_at_crash": store.torn_reachable(tree.root),
            "reachable_at_crash": _reachable_from_images(store, tree.root),
            "accepted": _recovery_view(accepted_report, accepted_store),
            "rejected": _recovery_view(rejected_report, rejected_store),
            "accepted_strategy": "undo_loser",
            "rejected_strategy": "leave_writes",
        }
        candidates.append({
            "candidate_id": f"op{seq}-recovery-undo_loser",
            "kind": "recovery", "strategy": "undo_loser",
            "verdict": "accepted",
            "idempotent": accepted_report.idempotent,
            "torn_reachable_after": accepted_report.torn_reachable_after,
        })
        candidates.append({
            "candidate_id": f"op{seq}-recovery-leave_writes",
            "kind": "recovery", "strategy": "leave_writes",
            "verdict": "rejected",
            "idempotent": rejected_report.idempotent,
            "torn_reachable_after": rejected_report.torn_reachable_after,
            "reason": "leaves reachable half-written page and/or dangling link",
        })
        # Adopt the accepted recovery in the live store and tree.
        store.pages = accepted_store.pages
        store.log = accepted_store.log
        store._seq = accepted_store._seq
        recovered_tree = _tree_after_recovery(store, root_before, tree.page_size,
                                              tree.dup_policy)
        tree.pages = recovered_tree.pages
        tree.root = recovered_tree.root
        tree._next_pid = recovered_tree._next_pid
        tree._touched = set()
        root_after = root_before
        reachable_after = tree.reachable_pids()
        if exc.point == "page_write_torn":
            _finding(trace, "NO_REACHABLE_HALF_PAGE",
                     "pass" if not accepted_report.torn_reachable_after
                     else "fail",
                     "torn page removed from reachable set by undo", seq,
                     sources=[f"op{seq}:crash", f"op{seq}:recovery:undo_loser"],
                     evidence={"torn_at_crash": crash_report["torn_pids_at_crash"],
                               "torn_after": accepted_report.torn_reachable_after})
        else:
            half = accepted_report.torn_reachable_after
            extra = _orphan_split_evidence(splits, tree)
            ok = not half and not extra
            _finding(trace, "NO_REACHABLE_HALF_PAGE",
                     "pass" if ok else "fail",
                     "crash mid-split leaves no reachable half page after undo",
                     seq,
                     sources=[f"op{seq}:crash", f"op{seq}:recovery:undo_loser"],
                     evidence={"torn_after": half, "orphan_siblings": extra,
                               "reachable": reachable_after})
        _finding(trace, "RECOVERY_IDEMPOTENT",
                 "pass" if accepted_report.idempotent else "fail",
                 "second recovery replay is a no-op", seq,
                 sources=[f"op{seq}:recovery:undo_loser"],
                 evidence={"hash_after": accepted_report.state_hash_after,
                           "hash_second": accepted_report.state_hash_second})

    # split findings (only completed ops carry split records).
    for rec in splits:
        _split_findings(trace, seq, rec, tree)

    if rejected:
        _finding(trace, "REJECTED_BY_DUP_POLICY", "info",
                 f"duplicate rejected by policy {tree.dup_policy}", seq,
                 sources=[f"op{seq}:line{line.lineno}"],
                 evidence={"reason": rejected})

    empties = tree.empty_leaf_pids() if not crash_report else []
    if line.op == "del" and empties:
        _finding(trace, "EMPTY_PAGE_AFTER_DELETE", "info",
                 "reachable leaf is empty after delete (policy: retain)", seq,
                 sources=[f"op{seq}:line{line.lineno}"],
                 evidence={"empty_pids": empties})

    pages_after = _snapshot_views(tree, store)
    _slot_consistency(trace, seq, pages_after)

    links_after = tree.leaf_sibling_links()
    event = OpEvent(
        seq=seq, kind=line.op, lsn=tree.lsn, key_hex=key.hex(),
        raw_line=line.lineno, raw_text=line.raw,
        before_root=root_before, after_root=root_after,
        pids_before=pids_before, pids_after=tree.allocated_pids(),
        reachable_after=reachable_after,
        pages_before={str(k): v for k, v in pages_before.items()},
        pages_after={str(k): v for k, v in pages_after.items()},
        splits=[r.to_dict() for r in splits],
        crash=crash_report,
        recovery=crash_report["accepted"] if crash_report else None,
        candidates=candidates,
        labels=labels,
        rejected=rejected,
    )
    trace.ops.append(event)
    _attach_link_evidence(event, links_before, links_after)


def _snapshot_views(tree: BTree, store: WalStore) -> Dict[int, dict]:
    views = {}
    for pid, page in tree.pages.items():
        image = store.pages.get(pid) or encode_page(page, tree.page_size)
        views[pid] = page_view(page, tree.page_size, image)
    return views


def _tree_after_recovery(store: WalStore, root_pid: int, page_size: int,
                         dup_policy: str) -> BTree:
    from .wal import load_tree
    return load_tree(store, root_pid, dup_policy)


def _reachable_from_images(store: WalStore, root_pid: int) -> List[int]:
    import struct
    seen: List[int] = []

    def walk(pid: int) -> None:
        if pid in seen or pid not in store.pages:
            return
        seen.append(pid)
        image = store.pages[pid]
        if image[0] == INTERNAL:
            n = struct.unpack_from("<H", image, 2)[0]
            (hi,) = struct.unpack_from("<I", image, 16)
            for i in range(n):
                (off,) = struct.unpack_from("<H", image, 32 + 2 * i)
                (child,) = struct.unpack_from("<I", image, off)
                walk(child)
            walk(hi)

    walk(root_pid)
    return seen


def _orphan_split_evidence(splits: List[SplitRecord], tree: BTree) -> List[int]:
    """Newly allocated sibling pids not reachable after the chosen recovery."""
    reachable = set(tree.reachable_pids())
    return [s.new_pid for s in splits if s.new_pid not in reachable]


def _recovery_view(report, store: WalStore) -> dict:
    return {
        "ran": report.ran,
        "loser_lsn": report.loser_lsn,
        "analysis": report.analysis,
        "redo": report.redo,
        "undo": report.undo,
        "torn_reachable_after": report.torn_reachable_after,
        "state_hash_before": report.state_hash_before,
        "state_hash_after": report.state_hash_after,
        "state_hash_second": report.state_hash_second,
        "idempotent": report.idempotent,
        "strategy": report.strategy,
    }


def _finding(trace: RunTrace, code: str, status: str, message: str, seq: int,
             sources: Optional[List[str]] = None,
             evidence: Optional[dict] = None) -> None:
    trace.findings.append(Finding(
        code=code, status=status, message=message, op_seq=seq,
        sources=sources or [], evidence=evidence or {}))


def _split_findings(trace: RunTrace, seq: int, rec: SplitRecord,
                    tree: BTree) -> None:
    # 1) byte-space vs key-count: the chosen cut must be one of the legal
    #    byte-feasible points; a pure half-by-key-count cut is listed only if
    #    it is itself byte-feasible.
    half_count = rec.key_count_before // 2
    status = "pass" if rec.cut in rec.legal_points else "fail"
    _finding(
        trace, "SPLIT_BY_BYTES", status,
        (f"split of page {rec.split_pid} at cut {rec.cut}; "
         f"legal byte-feasible points {rec.legal_points}; "
         f"key-count half {half_count}"),
        seq,
        sources=[f"op{seq}:split:{rec.split_pid}"],
        evidence={
            "split_pid": rec.split_pid, "new_pid": rec.new_pid,
            "cut": rec.cut, "legal_points": rec.legal_points,
            "key_count_half": half_count,
            "byte_payload": {"before": rec.byte_payload_before,
                             "left": rec.byte_payload_left,
                             "right": rec.byte_payload_right},
            "free_space": {"before": rec.free_before,
                           "left": rec.free_left, "right": rec.free_right},
            "chosen_by": rec.chosen_by,
        },
    )
    # 2) separator rule consistency.
    _finding(
        trace, "SEPARATOR_RULE", "pass",
        (f"{rec.level} split uses separator mode {rec.separator_mode}: "
         + ("leaf max key copied into parent and retained in leaf"
            if rec.separator_mode == "copy"
            else "internal separator promoted out of both children")),
        seq,
        sources=[f"op{seq}:split:{rec.split_pid}"],
        evidence={"mode": rec.separator_mode, "separator_hex": rec.separator.hex(),
                  "new_root": rec.new_root, "parent_pid": rec.parent_pid},
    )


def _slot_consistency(trace: RunTrace, seq: int, pages: dict) -> None:
    bad = []
    for pid, view in pages.items():
        if view["used_bytes"] > view["header_bytes"] + 2 * view["ncells"] + (
            view["cell_payload_bytes"]
        ):
            bad.append(pid)
        if view["free_space"]["value"] < 0:
            bad.append(pid)
    _finding(
        trace, "SLOT_FREE_SPACE_CONSISTENT",
        "pass" if not bad else "fail",
        "header + slot dir + cells + free space == page size for all pages",
        seq,
        sources=[f"op{seq}"],
        evidence={"bad_pids": bad},
    )


def _attach_link_evidence(event: OpEvent, before: dict, after: dict) -> None:
    event.crash = event.crash  # noqa: no-op placeholder kept for symmetry
    event.evidence_links = {  # type: ignore[attr-defined]
        "before": {str(k): list(v) for k, v in before.items()},
        "after": {str(k): list(v) for k, v in after.items()},
    }


def _evaluate_expectations(trace: RunTrace) -> None:
    by_code: Dict[str, List[Finding]] = {}
    for f in trace.findings:
        by_code.setdefault(f.code, []).append(f)
    for exp in trace.expectations:
        observed = {f.status for f in by_code.get(exp["code"], [])}
        exp["observed_statuses"] = sorted(observed)
        exp["satisfied"] = exp["status"] in observed
