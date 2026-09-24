"""Machine-checkable derived judgments over a run of B-tree operations."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..model import BTree, replay, replay_twice_idempotent, run_operations

PARAMS_VERSION = "1.0.0"  # bump when rule definitions change


@dataclass
class MetricResult:
    metric_id: str
    status: str                 # pass | fail | ambiguous
    value: Any
    unit: str                   # every domain quantity carries a unit
    depends_on_ops: List[int]   # logical op indices this metric read
    depends_on_params: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)  # legal candidates when ambiguous
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "status": self.status,
            "value": self.value,
            "unit": self.unit,
            "depends_on_ops": self.depends_on_ops,
            "depends_on_params": self.depends_on_params,
            "evidence": self.evidence,
            "candidates": self.candidates,
            "diagnostics": self.diagnostics,
            "params_version": PARAMS_VERSION,
        }


def _op_range(n: int) -> List[int]:
    return list(range(n))


def metric_byte_space_split(run: Dict[str, Any]) -> MetricResult:
    """Splits must be decided by byte space, not key count.

    Evidence: for every split event, the pre-split page's used_bytes exceeded
    page_size, and pages with more keys but fitting bytes did NOT split.
    """
    tree: BTree = run["tree"]
    page_size = tree.page_size
    splits = [w for w in tree.wal if w["kind"].startswith("split")]
    violations = []
    for w in splits:
        left = tree.pages.get(w["left"])
        # post-split both sides must fit
        for pid in (w["left"], w["right"]):
            p = tree.pages.get(pid)
            if p and p.used_bytes() > page_size:
                violations.append({"page_id": pid, "used_bytes": p.used_bytes(),
                                   "page_size": page_size})
    status = "pass" if not violations else "fail"
    return MetricResult(
        metric_id="byte_space_split",
        status=status,
        value={"split_count": len(splits), "violations": violations},
        unit="bytes",
        depends_on_ops=_op_range(len(run["steps"])),
        depends_on_params=["page_size", "duplicate_policy"],
        evidence={"splits": splits},
    )


def metric_separator_consistency(run: Dict[str, Any]) -> MetricResult:
    """Separator rule must be uniform: leaf splits copy the separator up,
    internal splits push it up (removed from the lower level)."""
    tree: BTree = run["tree"]
    violations = []
    for w in tree.wal:
        if w["kind"] == "split_leaf":
            sep, right = w["separator"], tree.pages.get(w["right"])
            # copy-up: separator must still be the first key of the right leaf
            if right is None or not right.keys or right.keys[0] != sep:
                violations.append({"lsn": w["lsn"], "rule": "copy_up",
                                   "separator": sep})
        elif w["kind"] == "split_internal":
            sep, left, right = w["separator"], tree.pages.get(w["left"]), tree.pages.get(w["right"])
            # push-up: separator must NOT remain in either child
            for p in (left, right):
                if p is not None and sep in p.keys:
                    violations.append({"lsn": w["lsn"], "rule": "push_up",
                                       "separator": sep, "page_id": p.page_id})
    return MetricResult(
        metric_id="separator_consistency",
        status="pass" if not violations else "fail",
        value={"violations": violations},
        unit="rule",
        depends_on_ops=_op_range(len(run["steps"])),
        depends_on_params=["duplicate_policy"],
        evidence={},
    )


def metric_no_reachable_half_page(run: Dict[str, Any]) -> MetricResult:
    """After a crash point, no reachable page may be a torn/half page."""
    bad = run["half_page_violations"]
    return MetricResult(
        metric_id="no_reachable_half_page",
        status="pass" if not bad else "fail",
        value={"violations": bad},
        unit="bytes",
        depends_on_ops=_op_range(len(run["steps"])),
        depends_on_params=["page_size"],
        evidence={"reachable_pages": run["reachable"]},
    )


def metric_replay_idempotent(run: Dict[str, Any]) -> MetricResult:
    """Replaying recovery twice must be idempotent."""
    tree: BTree = run["tree"]
    res = replay_twice_idempotent(tree.wal, tree.page_size, tree.duplicate_policy)
    return MetricResult(
        metric_id="replay_idempotent",
        status="pass" if res["idempotent"] else "fail",
        value={"idempotent": res["idempotent"]},
        unit="boolean",
        depends_on_ops=_op_range(len(run["steps"])),
        depends_on_params=["page_size", "duplicate_policy"],
        evidence={"reachable_after_first": sorted(res["first"].keys()),
                  "reachable_after_second": sorted(res["second"].keys())},
    )


def metric_duplicate_key_policy(run: Dict[str, Any]) -> MetricResult:
    """Duplicate-key policy: 'update' keeps one entry per key; 'allow' keeps all."""
    tree: BTree = run["tree"]
    ops = [s.op for s in run["steps"] if s.op.get("op") == "insert"]
    dup_keys = {o["key"] for o in ops if [x["key"] for x in ops].count(o["key"]) > 1}
    violations = []
    for key in dup_keys:
        leaf = tree._find_leaf(key) if tree.root_id is not None else None
        count = sum(1 for k in leaf.keys if k == key) if leaf else 0
        if tree.duplicate_policy == "update" and count != 1:
            violations.append({"key": key, "count": count, "policy": "update"})
        if tree.duplicate_policy == "allow" and count < 2:
            violations.append({"key": key, "count": count, "policy": "allow"})
    return MetricResult(
        metric_id="duplicate_key_policy",
        status="pass" if not violations else "fail",
        value={"violations": violations, "duplicate_keys": sorted(dup_keys)},
        unit="count",
        depends_on_ops=_op_range(len(run["steps"])),
        depends_on_params=["duplicate_policy"],
        evidence={},
    )


def metric_ambiguous_candidates(run: Dict[str, Any]) -> MetricResult:
    """When a separator position cannot be uniquely decided (equal byte fit on
    both sides), keep ALL legal candidates instead of silently picking one."""
    tree: BTree = run["tree"]
    candidates = []
    for w in tree.wal:
        if w["kind"] != "split_leaf":
            continue
        left, right = tree.pages.get(w["left"]), tree.pages.get(w["right"])
        if left is None or right is None:
            continue
        # a split is ambiguous if moving one more record either way still fits
        if left.keys and right.keys:
            alt_left = left.used_bytes() + len(right.keys[0].encode()) + 2
            alt_right = right.used_bytes() + len(left.keys[-1].encode()) + 2
            if alt_left <= tree.page_size and alt_right <= tree.page_size:
                candidates.append({
                    "lsn": w["lsn"],
                    "chosen_separator": w["separator"],
                    "alternatives": [left.keys[-1], right.keys[0]],
                })
    status = "ambiguous" if candidates else "pass"
    return MetricResult(
        metric_id="separator_candidates",
        status=status,
        value={"candidate_count": len(candidates)},
        unit="candidates",
        depends_on_ops=_op_range(len(run["steps"])),
        depends_on_params=["page_size"],
        evidence={},
        candidates=candidates,
    )


METRICS = {
    "byte_space_split": metric_byte_space_split,
    "separator_consistency": metric_separator_consistency,
    "no_reachable_half_page": metric_no_reachable_half_page,
    "replay_idempotent": metric_replay_idempotent,
    "duplicate_key_policy": metric_duplicate_key_policy,
    "separator_candidates": metric_ambiguous_candidates,
}


def compute_metric(metric_id: str, ops: List[Dict[str, Any]], page_size: int,
                   duplicate_policy: str, crash_at: Optional[int] = None) -> MetricResult:
    run = run_operations(ops, page_size, duplicate_policy, crash_at)
    return METRICS[metric_id](run)


def compute_all(ops: List[Dict[str, Any]], page_size: int,
                duplicate_policy: str, crash_at: Optional[int] = None) -> Dict[str, MetricResult]:
    run = run_operations(ops, page_size, duplicate_policy, crash_at)
    return {mid: fn(run) for mid, fn in METRICS.items()}


def invalidate_scope(changed_op_indices: Set[int],
                     metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    """Given which logical ops changed, return metric ids whose dependency
    sets intersect — these and ONLY these must be recomputed."""
    out = []
    for mid, m in metrics.items():
        if changed_op_indices & set(m.get("depends_on_ops", [])):
            out.append(mid)
    return sorted(out)


def fingerprint(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
