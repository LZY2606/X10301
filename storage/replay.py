"""Replay the offline case pack through the B+tree engine.

Produces a baseline replay plus one-fact-at-a-time counterexample replays:
variable keys, root split, cascade split, duplicate policy, empty page,
write-page interruption, and double recovery replay.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from btree.engine import (
    DUP_MULTISET,
    DUP_REJECT,
    DUP_REPLACE,
    EMPTY_DETACH,
    EMPTY_KEEP,
    Engine,
    Params,
)


def load_case(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expand_operations(case: Dict[str, Any]) -> Tuple[List[Dict[str, Any]],
                                                     List[Dict[str, Any]]]:
    ops: List[Dict[str, Any]] = []
    op_provenance: List[Dict[str, Any]] = []
    for stage in case["stages"]:
        for j, op0 in enumerate(stage.get("operations", [])):
            op = copy.deepcopy(op0)
            gen = stage.get("operations_gen")
            ops.append(op)
            op_provenance.append({
                "stage": stage["stage"],
                "index_in_stage": j,
                "source_ref": f"{case['case_id']}#{stage['stage']}:{j}",
                "raw_representation": copy.deepcopy(op),
            })
        gen = stage.get("operations_gen")
        if gen:
            base_index = len(ops)
            for k in range(gen["count"]):
                key = f"{gen['prefix']}{k:04d}"
                op = {"op": "put", "key": key,
                      "value": "v" + ("x" * (gen["value_bytes"] - 1))}
                ops.append(op)
                op_provenance.append({
                    "stage": stage["stage"],
                    "index_in_stage": k,
                    "source_ref":
                        f"{case['case_id']}#{stage['stage']}:gen{k}",
                    "raw_representation": copy.deepcopy(op),
                    "generated_by": gen["kind"],
                })
    return ops, op_provenance


def _op_key(op: Dict[str, Any]) -> Any:
    return op.get("key")


def run_replay(case: Dict[str, Any],
               ops: List[Dict[str, Any]],
               params: Optional[Params] = None,
               *,
               label: str = "baseline",
               source_patch: Optional[Dict[str, Any]] = None,
               variant: Optional[str] = None) -> Dict[str, Any]:
    params = params or Params(page_size=case["page_size_bytes"])
    engine = Engine(params)
    diagnostics: List[Dict[str, Any]] = []
    effective_ops = copy.deepcopy(ops)
    if source_patch:
        i = source_patch["op_index"]
        field_name = source_patch["field"]
        effective_ops[i][field_name] = source_patch["value"]

    records: List[Dict[str, Any]] = []
    for i, op in enumerate(effective_ops):
        if not _op_is_encodable(op, diagnostics, i):
            continue
        try:
            events = engine.execute(op)
        except Exception as exc:  # noqa: BLE001
            diagnostics.append({"op_index": i, "op": op,
                                "diagnostic": f"operation failed: {exc}",
                                "severity": "error",
                                "raw_retained": True})
            continue
        for ev in events:
            records.append({
                "event_id": f"{label}/e{ev.seq:05d}",
                "seq": ev.seq,
                "kind": ev.kind,
                "op_index": i + 1,
                "payload": ev.payload,
            })

    tree = None
    readback = []
    invariant_pass = None
    try:
        tree = engine.tree_view()
        readback = engine.read_all()
        inv_events = [r for r in records if r["kind"] == "invariants"]
        invariant_pass = (all(r["payload"]["passed"] for r in inv_events)
                          if inv_events else None)
    except Exception as exc:  # noqa: BLE001
        diagnostics.append({"diagnostic": f"readback failed: {exc}",
                            "severity": "error"})

    verdicts = _machine_verdicts(records, variant)
    return {
        "label": label,
        "variant": variant,
        "params_version": params.version_id(),
        "params": params.__dict__,
        "events": records,
        "tree": tree,
        "readback": readback,
        "diagnostics": diagnostics,
        "invariants_passed": invariant_pass,
        "verdicts": verdicts,
        "final": {
            "root_page_id": engine.store.root_page_id,
            "next_page_id": engine.store.next_page_id,
            "last_lsn": engine.store.lsn,
            "recovery_runs": engine.store.recoveries,
        },
    }


def _op_is_encodable(op: Dict[str, Any], diagnostics: list,
                     index: int) -> bool:
    for field_name in ("key", "value"):
        val = op.get(field_name)
        if val is None:
            continue
        if isinstance(val, (str, bytes, bytearray)):
            continue
        diagnostics.append({
            "op_index": index, "field": field_name, "value": val,
            "diagnostic": f"{field_name} has no byte representation; "
                          "kept at original position, not normalized away",
            "severity": "unconvertible",
            "raw_retained": True,
        })
        return False
    return True


def _machine_verdicts(records: List[Dict[str, Any]],
                      variant: Optional[str]) -> List[Dict[str, Any]]:
    verdicts: List[Dict[str, Any]] = []
    for r in records:
        p = r["payload"]
        if r["kind"] == "split":
            verdicts.append({
                "rule_id": "split-by-byte-space",
                "event_id": r["event_id"],
                "judgement": "decision_byte_space" in p
                and p["decision_byte_space"] is True,
                "basis": {"left_bytes": p["left_bytes"],
                          "right_bytes": p["right_bytes"],
                          "unit": p.get("byte_unit", "byte"),
                          "legal_candidates": p["legal_candidates"],
                          "ambiguous": p["ambiguous"]},
                "scope": ["split:" + p["group_id"]],
            })
            verdicts.append({
                "rule_id": "separator-copy-or-promote-consistent",
                "event_id": r["event_id"],
                "judgement": p["separator_policy"] in ("copy", "promote")
                and "promoted" in p
                and "separator_key_hex" in p,
                "basis": {"policy": p["separator_policy"],
                          "promoted": p["promoted"],
                          "separator_key_hex": p["separator_key_hex"]},
                "scope": ["split:" + p["group_id"]],
            })
        if r["kind"] == "recovery_verdict":
            verdicts.append({
                "rule_id": "no-reachable-half-page",
                "event_id": r["event_id"],
                "judgement": p["passed"],
                "basis": {"reachable_half_pages":
                          p["reachable_half_pages"]},
                "scope": ["recovery:run" + str(
                    next((x["payload"].get("run")
                          for x in records
                          if x["kind"] == "recovery_replay"
                          and x["event_id"] < r["event_id"]), 0))],
            })
    # recovery idempotency judgement
    replays = [r for r in records if r["kind"] == "recovery_replay"]
    if len(replays) >= 2:
        first = replays[-2]["payload"]
        second = replays[-1]["payload"]
        verdicts.append({
            "rule_id": "recovery-replay-idempotent",
            "event_id": replays[-1]["event_id"],
            "judgement": second["redone"] == [] and second["undone"] == [],
            "basis": {"first_run": {"redone": first["redone"],
                                    "undone": first["undone"]},
                      "second_run": {"redone": second["redone"],
                                     "undone": second["undone"]}},
            "scope": ["recovery:replay-twice"],
        })
    return verdicts


VARIANTS = (
    ("variable_keys", "变长键：按字节空间决定分裂"),
    ("root_split", "根分裂"),
    ("cascade_split", "级联分裂"),
    ("duplicate_policy", "重复键策略"),
    ("empty_page", "删除后空页"),
    ("write_interrupt", "写页中断"),
    ("recover_twice", "恢复重放两次"),
)


def build_archive(case_path: str | Path) -> Dict[str, Any]:
    case = load_case(case_path)
    ops, provenance = expand_operations(case)
    baseline = run_replay(case, ops, label="baseline")

    variants: Dict[str, Any] = {}
    # Each counterexample changes exactly one direct source from the baseline.
    variants["variable_keys"] = run_replay(
        case, ops,
        Params(page_size=case["page_size_bytes"], split_bias="left_heavy"),
        label="variant-variable-keys", variant="variable_keys")
    variants["root_split"] = _subreplay(
        case, ops, label="variant-root-split", variant="root_split",
        stop_after_first_kind="root_change")
    variants["cascade_split"] = run_replay(
        case, ops, Params(page_size=case["page_size_bytes"]),
        label="variant-cascade-split", variant="cascade_split")
    variants["duplicate_policy"] = run_replay(
        case, ops,
        Params(page_size=case["page_size_bytes"],
               duplicate_policy=DUP_MULTISET),
        label="variant-duplicate-multiset", variant="duplicate_policy")
    variants["duplicate_policy_reject"] = run_replay(
        case, ops,
        Params(page_size=case["page_size_bytes"],
               duplicate_policy=DUP_REJECT),
        label="variant-duplicate-reject", variant="duplicate_policy")
    variants["empty_page"] = run_replay(
        case, ops,
        Params(page_size=case["page_size_bytes"],
               empty_leaf_policy=EMPTY_DETACH),
        label="variant-empty-detach", variant="empty_page")
    variants["write_interrupt"] = run_replay(
        case, ops, Params(page_size=case["page_size_bytes"]),
        label="variant-write-interrupt", variant="write_interrupt")
    variants["recover_twice"] = run_replay(
        case, ops, Params(page_size=case["page_size_bytes"]),
        label="variant-recover-twice", variant="recover_twice")

    # parameter revision (v2): only split/separator dependent results change
    rev = case["parameter_revision"]
    params_v2 = Params(page_size=case["page_size_bytes"],
                       split_bias=rev["to"]["split_bias"],
                       separator_policy=rev["to"]["separator_policy"])
    revised = run_replay(case, ops, params_v2, label="params-v2")

    # local source change: one operation's value -> local invalidation only
    target_stage = next(s for s in case["stages"]
                        if s["stage"] == case["source_change"]["stage"])
    stage_op_indices = []
    count = 0
    for s in case["stages"]:
        n = len(s.get("operations", []))
        if s["stage"] == target_stage["stage"]:
            stage_op_indices = list(range(count,
                                          count + n))
            break
        count += n
    patch_index = stage_op_indices[case["source_change"]["target_op"]]
    local_change = run_replay(
        case, ops, Params(page_size=case["page_size_bytes"]),
        label="source-local-change",
        source_patch={"op_index": patch_index, "field": "value",
                      "value": case["source_change"]["new_value"]})

    return {
        "case": case,
        "operations": ops,
        "operation_provenance": provenance,
        "baseline": baseline,
        "variants": variants,
        "parameter_revision": {
            "from": rev["from"], "to": rev["to"],
            "replay": revised,
            "invalidation_scope":
                ["split", "root_change", "parent_insert", "invariants"],
        },
        "source_change": {
            "spec": case["source_change"],
            "op_index": patch_index,
            "replay": local_change,
            "invalidation_scope": [
                f"logical_operation:op{patch_index + 1}",
                f"page_pre_image:op{patch_index + 1}",
                f"invariants:op{patch_index + 1}",
            ],
        },
    }


def _subreplay(case, ops, *, label, variant, stop_after_first_kind):
    replay = run_replay(case, ops, Params(page_size=case["page_size_bytes"]),
                        label=label, variant=variant)
    first = next((i for i, e in enumerate(replay["events"])
                  if e["kind"] == stop_after_first_kind), None)
    replay["first_event_of_interest"] = first
    return replay
