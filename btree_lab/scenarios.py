"""Counterexample scenarios for the page-split archive.

Every scenario keeps a *baseline* run and changes exactly one direct source
(an input fragment or one rule parameter).  Both the adopted and the
rejected outcome are retained, and the judgement is emitted as a
machine-readable event (rule / status / evidence / affected scope).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from .engine import (
    Engine, EngineConfig, CrashInjected,
    DUP_ERROR, DUP_LAST, DUP_MULTI,
    EMPTY_KEEP, EMPTY_FREE, EMPTY_UNLINK,
)
from .pages import Page
from .wal import recover
from . import invariants


def run_clean(config: EngineConfig, ops: list[dict[str, Any]],
              crash_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    engine = Engine(config)
    crashed: dict[str, Any] | None = None
    traces: list = []
    try:
        traces = engine.run_ops(ops, crash_plan)
    except CrashInjected as exc:
        crashed = {"where": exc.where, "op_seq": exc.trace.seq}
    return {"engine": engine, "traces": traces, "crashed": crashed}


def _page_digest(engine: Engine) -> dict[int, str]:
    return {pid: page.serialize().hex()
            for pid, page in sorted(engine.pages.items())}


def _store_digest(engine: Engine) -> dict[int, str]:
    return {pid: raw.hex() for pid, raw in sorted(engine.store.pages.items())}


def _trace_json(traces: list) -> list[dict[str, Any]]:
    out = []
    for t in traces:
        out.append({
            "seq": t.seq, "op": t.op,
            "before_snapshot": t.before_snapshot,
            "changes": t.changes,
            "root_before": t.root_before, "root_after": t.root_after,
            "lsn_range": t.lsn_range,
            "pages_allocated": t.pages_allocated,
            "split_events": t.split_events, "note": t.note,
        })
    return out


def _checks(engine: Engine, config: EngineConfig) -> list[dict[str, Any]]:
    return invariants.run_tree_checks(
        engine.pages, [engine.root],
        dup_policy=config.dup_policy, sep_mode=config.separator_rule)


def _event(rule: str, status: str, scenario: str, message: str,
           affected: list[dict[str, Any]], evidence: list | None = None,
           candidates: list | None = None) -> dict[str, Any]:
    return {
        "kind": "verdict", "scenario": scenario, "rule": rule,
        "status": status, "message": message,
        "affected_scope": affected, "evidence": evidence or [],
        "candidates": candidates or [],
    }


# --------------------------------------------------------------- scenario 1
def variable_length_bytes(config: EngineConfig, ops: list) -> dict[str, Any]:
    """Byte- vs key-count split decision on variable-length keys."""
    good = EngineConfig(**{**config.__dict__, "split_metric": "bytes"})
    bad = EngineConfig(**{**config.__dict__, "split_metric": "keys"})
    a = run_clean(good, ops)["engine"]
    b = run_clean(bad, ops)["engine"]
    affected: list[dict[str, Any]] = []
    status = "pass"
    details = []
    for pid in sorted(set(a.pages) & set(b.pages)):
        pa, pb = a.pages[pid], b.pages[pid]
        if (pa.free_bytes, pa.slot_count) != (pb.free_bytes, pb.slot_count):
            affected.append({"kind": "page", "id": pid})
            details.append({"pid": pid,
                            "bytes_rule_free": pa.free_bytes,
                            "count_rule_free": pb.free_bytes,
                            "bytes_rule_slots": pa.slot_count,
                            "count_rule_slots": pb.slot_count})
    overflow = [pid for pid, p in b.pages.items() if p.free_bytes < 0]
    if overflow:
        status = "fail"
        msg = (f"key-count split overflows pages {overflow} with "
               "variable-length keys; byte-space split required")
    else:
        msg = ("byte-space and key-count splits diverge on pages "
               f"{[x['id'] for x in affected]}; byte rule is the accepted one")
    split_checks = []
    for t in run_clean(good, ops)["traces"]:
        for ev in t.split_events:
            if ev["kind"] == "leaf_split":
                split_checks.append(ev)
    return {
        "id": "variable_length_bytes",
        "title": "变长键按字节空间而非键数决定分裂",
        "baseline": {"split_metric": "bytes",
                     "checks": [c for c in _checks(a, good)
                                if c["rule"] == "page.fits_in_page_size"],
                     "tree": a.tree_shape()},
        "mutant": {"changed_source": "parameter:split_metric",
                   "old": "bytes", "new": "keys",
                   "checks": [c for c in _checks(b, bad)
                              if c["rule"] == "page.fits_in_page_size"],
                   "overflow_pages": overflow, "tree": b.tree_shape()},
        "events": [_event(
            "split.by_byte_space_not_key_count", status,
            "variable_length_bytes", msg,
            affected=affected + [{"kind": "parameter", "id": "split_metric"}],
            evidence=[{"metric": "page_occupancy", "detail": details,
                       "unit": "byte"},
                      {"metric": "split_events", "detail": split_checks}])],
        "accepted": "bytes", "rejected": "keys",
    }
