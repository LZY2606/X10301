"""Offline case package: deterministic replay + archive building.

The raw material (full trace with pre-images, slot directories, free space,
sibling links, page-id allocation and LSNs) is stored verbatim as a source.
Interpretations (judgments, candidates, rulings, invalidations) live in the
event layer, so re-interpretation always creates a new branch.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from btree import (Engine, LEAF, INTERNAL, HEADER_SIZE, free_bytes,
                   page_layout_segments, slot_offsets, read_cells, _u16,
                   _O_LEFTMOST, _O_PARENT, _O_SIBLING, _O_FLAGS, FLAG_TORN)
import invariants

CASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cases", "page_split_archive")
KNOWN_UNITS = {"byte", "page_offset"}


def load_case(path=None):
    path = path or os.path.join(CASE_DIR, "case.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def replay(ops, params):
    engine = Engine(
        page_size=params["page_size"],
        duplicate_policy=params.get("duplicate_policy", "allow_right"),
        empty_leaf_policy=params.get("empty_leaf_policy", "unlink"),
    )
    engine.run_script(ops)
    torn_after = [pid for pid, page in engine.disk.items()
                  if int(page[_O_FLAGS]) & FLAG_TORN]
    state = {
        "reachable_after_recovery": engine.reachable(),
        "torn_after_recovery": torn_after,
        "root": engine.root,
        "state_hash": engine.state_hash(),
    }
    return engine, state


def _normalized_annotations(store, source_id, annotations):
    """Derived normalization; unconvertible quantities stay raw + diagnostic."""
    results = []
    for ann in annotations or []:
        unit = ann.get("unit")
        if unit not in KNOWN_UNITS:
            store.add_diagnostic(
                source_id, ann["field"],
                "unknown unit %r; raw value retained without normalization"
                % unit)
            results.append({**ann, "normalized": None,
                            "diagnostic": "unconvertible_unit"})
        else:
            results.append({**ann, "normalized": ann["value"]})
    return results


def _judgment_events(store, branch, trace, log, params, rules_id, source_id,
                     state, id_prefix="jdg"):
    judgments = invariants.evaluate(trace, log, params, state, source_id)
    events = {}
    for i, judgment in enumerate(judgments, start=1):
        judgment_id = "%s-%s-%02d" % (id_prefix, branch, i)
        payload = copy.deepcopy(judgment)
        payload["judgment_id"] = judgment_id
        event = store.add_event(branch, "judgment", payload,
                                rule_version=rules_id)
        events[judgment["rule_id"]] = event
    return judgments, events


def disk_at(trace, page_size, at_step=None):
    """Reconstruct page bytes from post-images up to a trace step."""
    disk = {}
    for step in trace:
        if at_step is not None and step["seq"] > at_step:
            break
        for write in step.get("writes", []):
            image = write["post_image"]
            disk[write["page_id"]] = np.frombuffer(
                bytes.fromhex(image), dtype=np.uint8).copy()
    # A crash step holds no redo records; emulate the torn page on disk.
    for step in trace:
        if at_step is not None and step["seq"] > at_step:
            break
        if step["kind"] == "crash":
            lsn = step["torn_lsn"]
            record = next(w for s in trace for w in s.get("writes", [])
                          if w["lsn"] == lsn)
            page = disk[record["page_id"]]
            post = bytes.fromhex(record["post_image"])
            pre = bytes.fromhex(record["pre_image"])
            half = len(post) // 2
            page[:half] = np.frombuffer(post[:half], dtype=np.uint8)
            page[half:] = np.frombuffer(pre[half:], dtype=np.uint8)
            page[_O_FLAGS] |= FLAG_TORN
    return disk


def step_view(trace, at_step=None):
    if at_step is None:
        step = next((s for s in reversed(trace)
                     if s["kind"] != "init"), trace[-1])
    else:
        step = trace[min(at_step, len(trace) - 1)]
    return step


def tree_view(trace, page_size, at_step=None):
    step = step_view(trace, at_step)
    disk = disk_at(trace, page_size, at_step)
    root = step["root"]
    freed = set(step["freed"])
    nodes, edges = [], []

    def walk(pid, depth):
        if pid not in disk:
            return
        page = disk[pid]
        page_type = int(page[0])
        node = {
            "id": "pg:%d@step:%d" % (pid, step["seq"]),
            "stable_id": "pg:%d" % pid,
            "page_id": pid, "depth": depth,
            "kind": "leaf" if page_type == LEAF else "internal",
            "sibling": _u16(page, _O_SIBLING),
            "lsn": _u16(page, 7),
            "free_space": {"value": free_bytes(page), "unit": "byte"},
            "slot_count": _u16(page, 9),
            "torn": bool(int(page[_O_FLAGS]) & FLAG_TORN),
            "freed": pid in freed,
            "keys": [k.decode("latin-1") for k, _ in read_cells(page)],
        }
        nodes.append(node)
        if page_type == INTERNAL:
            children = [_u16(page, _O_LEFTMOST)] + \
                [c for _, c in read_cells(page)]
            for child in children:
                edges.append({"from": "pg:%d" % pid, "to": "pg:%d" % child})
                walk(child, depth + 1)
        elif _u16(page, _O_SIBLING):
            edges.append({"from": "pg:%d" % pid,
                          "to": "pg:%d" % _u16(page, _O_SIBLING),
                          "kind": "sibling"})

    walk(root, 0)
    return {"root": "pg:%d" % root, "step_seq": step["seq"],
            "step_kind": step["kind"], "nodes": nodes, "edges": edges}


def page_view(trace, page_size, page_id, at_step=None):
    disk = disk_at(trace, page_size, at_step)
    step = step_view(trace, at_step)
    page = disk[page_id]
    return {
        "id": "pg:%d@step:%d" % (page_id, step["seq"]),
        "stable_id": "pg:%d" % page_id,
        "page_size": {"value": page_size, "unit": "byte"},
        "page_id": page_id,
        "step_seq": step["seq"],
        "bytes": bytes(page).hex(),
        "segments": page_layout_segments(bytes(page).hex()),
        "slot_directory": [
            {"slot": i, "offset": {"value": off, "basis": "page_offset"}}
            for i, off in enumerate(slot_offsets(page))
        ],
        "free_space": {"value": free_bytes(page), "unit": "byte"},
    }


def _trace_payload(trace):
    return trace


def build_archive(store, case=None):
    case = case or load_case()
    params = dict(case["params"])
    store.add_rule_version("rules-v1", params)

    engine, state = replay(case["ops"], params)
    expected = set(case["expect_scenarios"])
    actual = {t for s in engine.trace for t in s.get("scenarios", [])}
    missing = expected - actual
    if missing:
        raise ValueError("case replay missing scenarios: %s" % sorted(missing))

    store.add_source("src-baseline", "page_split_trace", json.dumps({
        "case_id": case["case_id"], "title": case["title"],
        "raw_ops": case["ops"], "params": params,
        "page_size": case["page_size"],
        "annotations": case["annotations"],
        "trace": _trace_payload(engine.trace),
        "log": engine.log,
        "state": state,
    }, ensure_ascii=False))

    store.create_branch("main", "主叙事：基线-参数修订-局部源修订-冲突裁定")
    import_event = store.add_event("main", "import", {
        "source": "src-baseline", "case_id": case["case_id"],
        "title": case["title"],
        "page_size": case["page_size"],
        "scenarios_present": sorted(actual),
    }, rule_version="rules-v1")

    for i, step in enumerate([s for s in engine.trace if s["kind"] != "init"]):
        payload = {
            "source": "src-baseline", "step_seq": step["seq"],
            "op_index": i,
            "kind": step["kind"], "scenarios": step.get("scenarios", []),
            "root": step["root"], "freed": step["freed"],
            "lsn_range": step["lsn_range"],
            "logical_op": step.get("logical_op"),
            "key": step.get("key"),
            "splits": step.get("splits", []),
            "idempotent": step.get("idempotent"),
        }
        store.add_event("main", "run_step", payload,
                        rule_version="rules-v1")

    judgments, judgment_events = _judgment_events(
        store, "main", engine.trace, engine.log, params,
        "rules-v1", "src-baseline")

    # decision point: all legal split-point candidates when not unique
    tie_step = next(s for s in engine.trace
                    if any(sp.get("undecided") for sp in s.get("splits", [])))
    tie_split = next(sp for sp in tie_step["splits"] if sp.get("undecided"))
    tie_step = tie_step["seq"]
    candidates = [str(c["index"]) for c in tie_split["candidates"]]
    chosen = str(tie_split["chosen"])
    store.add_event("main", "decision_point", {
        "decision_point": "dp-tie-step%d" % tie_step,
        "source": "src-baseline", "step_seq": tie_step,
        "question": "变长键叶分裂在多个字节合法切点中如何选择",
        "candidates": candidates, "chosen": chosen,
    }, rule_version="rules-v1")

    # window A saves; window B saves against the stale head -> conflict kept
    base_after_import = store.head("main")
    ruling_event = store.add_event("main", "ruling", {
        "decision_point": "dp-tie-step%d" % tie_step,
        "candidate": chosen, "action": "accept",
        "window": "A", "note": "采纳最均衡切点",
    }, rule_version="rules-v1")
    other = next(c for c in candidates if c != chosen)
    try:
        store.save_guarded("main", "ruling", {
            "decision_point": "dp-tie-step%d" % tie_step,
            "candidate": other, "action": "accept",
            "window": "B", "note": "另一审阅者的候选",
        }, base_after_import, rule_version="rules-v1")
    except Exception as conflict:
        store.add_event("main", "conflict", {
            "decision_point": "dp-tie-step%d" % tie_step,
            "attempted": conflict.attempted,
            "base_head": base_after_import,
            "current_head": conflict.current_head,
            "events_since_base": [
                {"id": e["id"], "kind": e["kind"],
                 "payload": e["payload"]}
                for e in conflict.events_since_base],
        }, rule_version="rules-v1")

    # parameter revision -> dependency-scoped invalidation
    revised = dict(params)
    revised.update(case["param_revision"]["changes"])
    store.add_rule_version(case["param_revision"]["version_id"], revised,
                           parent="rules-v1")
    store.add_event("main", "param_revision", {
        "new_rule_version": case["param_revision"]["version_id"],
        "changes": case["param_revision"]["changes"],
        "note": case["param_revision"]["note"],
    }, rule_version="rules-v1")
    store.invalidate_for_params(
        "main", "rules-v1",
        list(case["param_revision"]["changes"].keys()),
        "duplicate policy revision")

    # localized source revision: correct one op's bytes
    first_split_seq = next(
        s["seq"] for s in engine.trace
        if "leaf_split" in s.get("scenarios", []))
    # steps before delete_leaf_keys map 1:1 onto ops (init occupies seq 0)
    op_idx = first_split_seq - 1
    revised_ops = copy.deepcopy(case["ops"])
    revised_ops[op_idx]["value"] = case["source_revision"]["to"]
    engine_v2, state_v2 = replay(revised_ops, params)
    store.add_source(case["source_revision"]["id"], "page_split_trace",
                     json.dumps({
                         "case_id": case["case_id"],
                         "raw_ops": revised_ops, "params": params,
                         "trace": engine_v2.trace, "log": engine_v2.log,
                         "state": state_v2,
                     }, ensure_ascii=False),
                     supersedes="src-baseline")
    store.add_event("main", "source_revision", {
        "source": case["source_revision"]["id"],
        "supersedes": "src-baseline", "op_index": op_idx,
        "key": revised_ops[op_idx].get("key"),
        "field": case["source_revision"]["field"],
        "note": case["source_revision"]["note"],
    }, rule_version=case["param_revision"]["version_id"])
    store.invalidate_for_sources(
        "main", "src-baseline", [engine.trace[1 + op_idx]["seq"]],
        "localized byte correction")

    # unconvertible quantities: kept raw, diagnostics emitted
    normalized = _normalized_annotations(
        store, "src-baseline", case["annotations"])
    store.add_event("main", "diagnostic", {
        "source": "src-baseline", "annotations": normalized,
    }, rule_version="rules-v1")

    _build_perturbations(store, case, params, import_event["id"])
    return {"import_event": import_event["id"],
            "tie_step": tie_step, "op_index": op_idx,
            "scenarios": sorted(actual)}


def _build_perturbations(store, case, params, base_event_id):
    """Each perturbation changes exactly one direct source from the baseline."""
    for perturb in case["perturbations"]:
        branch = "pert-" + perturb["id"]
        store.create_branch(branch, "扰动：" + perturb["desc"],
                            parent="main", base_event=base_event_id)
        pert_params = dict(params)
        pert_params.update(perturb.get("param_patch", {}))
        rules_id = perturb.get("rules", "rules-v1")
        if rules_id != "rules-v1":
            store.add_rule_version(rules_id, pert_params, parent="rules-v1")
        ops = copy.deepcopy(case["ops"])
        patch = perturb.get("patch")
        if patch:
            if "index" in patch:
                ops[patch["index"]].update(
                    {k: v for k, v in patch.items() if k != "index"})
            else:
                for spec in ops:
                    if spec["op"] == patch["op"]:
                        spec.update({k: v for k, v in patch.items()
                                     if k != "op"})
                        break
        engine, state = replay(ops, pert_params)
        source_id = "src-pert-" + perturb["id"]
        store.add_source(source_id, "page_split_trace", json.dumps({
            "case_id": case["case_id"], "perturbation": perturb["id"],
            "desc": perturb["desc"], "raw_ops": ops, "params": pert_params,
            "trace": engine.trace, "log": engine.log, "state": state,
        }, ensure_ascii=False), supersedes="src-baseline")
        store.add_event(branch, "import", {
            "source": source_id, "perturbation": perturb["id"],
            "desc": perturb["desc"], "changed_source": "src-baseline",
        }, rule_version=rules_id)
        _judgment_events(store, branch, engine.trace, engine.log,
                         pert_params, rules_id, source_id, state,
                         id_prefix="jdg-pert")
        outcome = case["perturbation_rulings"][perturb["id"]]
        store.add_event(branch, "ruling", {
            "decision_point": "dp-pert-" + perturb["id"],
            "candidate": "perturbed-result",
            "action": "accept" if outcome == "accepted" else "reject",
            "note": "扰动结果%s" % ("被采纳" if outcome == "accepted" else "未采纳"),
        }, rule_version=rules_id)
