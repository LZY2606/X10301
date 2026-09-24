"""Offline case pack runner for "页裂档案".

Raw representations are preserved (quantities keep value/unit/coordinate
basis); only the derived layer normalizes.  Values that cannot be converted
stay in place with diagnostics instead of being dropped for rendering.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from btree import BTree, check_invariants, reachable_in_cache

CASE_PATH = Path(__file__).parent / "cases" / "pagesplit_case.json"


def load_case(path: str | Path | None = None) -> dict:
    with open(path or CASE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def normalize_params(case: dict, overrides: dict | None = None) -> dict:
    """Derived-layer normalization; raw quantity text is retained alongside."""
    raw = case["params"]
    params = {
        "page_size": raw["page_size"]["value"],
        "duplicate_policy": raw["duplicate_policy"],
        "separator_rule": raw["separator_rule"],
        "units": {
            "page_size": raw["page_size"]["unit"],
            "header_size": raw["header_size"]["unit"],
            "slot_size": raw["slot_size"]["unit"],
            "coordinate_basis": case["coordinate_basis"]["page_origin"],
        },
        "raw": copy.deepcopy(raw),
    }
    if overrides:
        params.update(overrides)
    return params


def bulk_keys(op: dict) -> list[tuple[bytes, bytes]]:
    prefix = op.get("prefix", "k-")
    count = int(op["count"])
    pad = int(op.get("key_pad_bytes", 0))
    value_len = int(op.get("value_bytes", 4))
    return [
        (prefix.encode() + f"{i:04d}".encode() + b"x" * pad, b"v" * value_len)
        for i in range(1, count + 1)
    ]


def _resolve_torn_target(tree: BTree, target: str) -> int:
    if target == "root":
        return tree.root_id
    if target == "leftmost-leaf":
        return tree.leaf_chain()[0]
    return int(target)


class Session:
    """Replays an ordered narrative; never drops pre-images or diagnostics."""

    def __init__(self, params: dict):
        self.params = params
        self.tree = BTree(
            page_size=params["page_size"],
            duplicate_policy=params["duplicate_policy"],
            separator_rule=params["separator_rule"],
        )
        self.records: list[dict] = []
        self.log_snapshots: list[dict] = []

    def _page_dump(self) -> dict:
        return {
            str(pid): self.tree.pages[pid].serialize(self.params["page_size"]).hex()
            for pid in reachable_in_cache(self.tree)
        }

    def run(self, operations: list[dict]) -> list[dict]:
        for index, op in enumerate(operations):
            record = {"index": index, "raw_op": copy.deepcopy(op),
                      "input_fragment_id": f"op-{index:03d}"}
            kind = op["op"]
            if kind == "insert":
                record["trace"] = self.tree.insert(
                    op["key"].encode(), op.get("value", "").encode()
                )
            elif kind == "insert_bulk":
                traces = []
                for key, value in bulk_keys(op):
                    traces.append(self.tree.insert(key, value))
                record["trace"] = {"op": "insert_bulk", "count": len(traces),
                                   "log_lsns": [t["log_lsns"] for t in traces],
                                   "allocations": [a for t in traces
                                                   for a in t["allocations"]],
                                   "splits": [s for t in traces
                                              for s in t["splits"]],
                                   "pre_images": {},
                                   "root_changed": any(t["root_changed"] for t in traces)}
            elif kind == "delete":
                record["trace"] = self.tree.delete(op["key"].encode())
            elif kind == "checkpoint":
                self.tree.commit()
                record["trace"] = {"op": "checkpoint",
                                   "checkpoint_lsn": self.tree.lsn}
            elif kind == "crash":
                amount = op["written_bytes"]
                written = amount["value"]
                target = _resolve_torn_target(self.tree, op.get("torn_target", "root"))
                record["trace"] = self.tree.crash(target, written)
                record["reach_after_crash"] = self.tree.reachable_pages()
            elif kind == "recover":
                record["trace"] = self.tree.recover()
                record["reach_after_recover"] = self.tree.reachable_pages()
            elif kind == "inspect":
                reach = self.tree.reachable_pages()
                record["trace"] = {"op": "inspect", "reachable": reach}
                record["expect_torn"] = bool(op.get("expect_torn"))
                record["torn_observed"] = bool(reach["torn"])
                record["inspect_ok"] = record["torn_observed"] == record["expect_torn"]
            else:
                raise ValueError(f"unknown op {kind}")
            record["post_pages"] = self._page_dump()
            record["superblock"] = dict(self.tree.superblock)
            record["log_tail"] = [
                {"lsn": r["lsn"], "type": r["type"], "page_id": r["page_id"]}
                for r in self.tree.log[-6:]
            ]
            self.records.append(record)
        return self.records

    def verdicts(self) -> list[dict]:
        return check_invariants(self.tree)


def run_baseline(case: dict, overrides: dict | None = None) -> Session:
    params = normalize_params(case, overrides)
    session = Session(params)
    session.run(case["baseline_operations"])
    return session


def run_counterexample(case: dict, counterexample: dict) -> dict:
    """Runs one counterexample; candidate parameter sets each get full results.

    Accepted/rejected status is never decided here — humans adjudicate; every
    legal candidate is retained as undetermined.
    """
    candidates = counterexample.get("candidates")
    variants = candidates if candidates else [{"code": counterexample["code"]}]
    results = []
    for candidate in variants:
        overrides = {}
        if "params" in candidate:
            overrides.update(candidate["params"])
        if "separator_rule" in candidate:
            overrides["separator_rule"] = candidate["separator_rule"]
        params = normalize_params(case, overrides)
        session = Session(params)
        session.run(counterexample["operations"])
        verdicts = session.verdicts()
        expected = counterexample["expect"]
        matched = next((v for v in verdicts if v["rule_id"] == expected["rule_id"]), None)
        incompatible = candidate.get("incompatible_rules", [])
        blocked = next((v["rule_id"] for v in verdicts
                        if v["rule_id"] in incompatible and v["status"] == "pass"), None)
        results.append({
            "candidate_code": candidate["code"],
            "params_used": {
                "page_size": {"value": params["page_size"],
                              "unit": params["units"]["page_size"]},
                "duplicate_policy": params["duplicate_policy"],
                "separator_rule": params["separator_rule"],
            },
            "records": session.records,
            "verdicts": verdicts,
            "expect": expected,
            "status": "undetermined",
            "matches_expect": bool(
                matched and matched["status"] == expected["status"] and blocked is None
            ),
            "blocked_by_rule": blocked,
            "final_pages": session._page_dump(),
            "root_id": session.tree.root_id,
        })
    return {
        "code": counterexample["code"],
        "title": counterexample["title"],
        "varies": counterexample["varies"],
        "expect": counterexample["expect"],
        "results": results,
    }


def raw_fragments(case: dict) -> list[dict]:
    """Imported raw, never normalized away; unconvertible -> diagnostic."""
    fragments = []
    for fragment in case.get("raw_fragments", []):
        kept = copy.deepcopy(fragment)
        if fragment.get("encoding") not in ("utf-8", "hex", None):
            kept.setdefault("diagnostics", []).append(
                {"code": fragment["diagnostic"],
                 "value_kept_at": "raw_hex",
                 "normalized": False})
        fragments.append(kept)
    return fragments
