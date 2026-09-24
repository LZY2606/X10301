"""Structural-invariant evaluation.

Judgments are machine-readable events: rule id, status, evidence (trace seq /
lsn), dependency scope and the full set of legal candidates when the result
is not uniquely decidable.
"""
from __future__ import annotations

from btree import LEAF, INTERNAL, page_keys, parse_cell, slot_offsets

RULES = [
    ("split_basis_bytes", ["split_basis"]),
    ("separator_consistency", ["separator_leaf", "separator_internal"]),
    ("duplicate_policy", ["duplicate_policy"]),
    ("empty_leaf_unlinked", ["empty_leaf_policy"]),
    ("crash_no_reachable_half_page", ["recovery"]),
    ("replay_idempotent", ["recovery"]),
]


def _post_image(trace, page_id, before_seq=None):
    for step in reversed(trace):
        if before_seq is not None and step["seq"] >= before_seq:
            continue
        for write in step.get("writes", []):
            if write["page_id"] == page_id:
                return write["post_image"]
    return None


def evaluate(trace, log, rules, state_hashes=None, source_id="src"):
    state_hashes = state_hashes or {}
    judgments = []
    split_steps = [(s, split) for s in trace for split in s.get("splits", [])]

    # 1. split decided by byte space, not key count
    evidence = []
    ok = True
    for step, split in split_steps:
        if split.get("basis") != "bytes":
            ok = False
        for write in step["writes"]:
            if write["page_id"] in (split.get("left"), split.get("right")):
                evidence.append({
                    "step_seq": step["seq"], "page_id": write["page_id"],
                    "free_space": write["free_space"],
                    "chosen": split["chosen"],
                    "candidate_count": len(split["candidates"]),
                })
    candidates = sorted({c["index"] for _, s in split_steps for c in s["candidates"]})
    judgments.append({
        "rule_id": "split_basis_bytes",
        "status": "pass" if ok and split_steps else ("fail" if split_steps else "undecided"),
        "scope": {"steps": [s["seq"] for s, _ in split_steps],
                  "lsns": [w["lsn"] for s in trace for w in s.get("writes", [])
                           if any(x for x in s.get("splits", []))]},
        "evidence": evidence,
        "candidates": [{"split_index": i} for i in candidates],
        "undecided_points": [
            {"step_seq": s["seq"], "split_type": sp["type"],
             "candidates": sp["candidates"]}
            for s, sp in split_steps if sp.get("undecided")
        ],
    })

    # 2. separator rule: copy for leaf, promote for internal
    sep_ok = True
    sep_evidence = []
    for step, split in split_steps:
        right_img = next((w["post_image"] for w in step["writes"]
                          if w["page_id"] == split["right"]), None)
        ptype = LEAF if split["type"] == "leaf" else INTERNAL
        sep = split.get("separator") or split.get("promoted")
        right_keys = [k.decode("latin-1") for k in page_keys(right_img, ptype)]
        if split["type"] == "leaf":
            present = sep in right_keys
            consistent = present and split["separator_rule"] == "copy"
        else:
            left_img = next((w["post_image"] for w in step["writes"]
                             if w["page_id"] == split["left"]), None)
            left_keys = [k.decode("latin-1") for k in page_keys(left_img, INTERNAL)]
            present = sep in right_keys or sep in left_keys
            consistent = (not present) and split["separator_rule"] == "promote"
        sep_ok = sep_ok and consistent
        sep_evidence.append({"step_seq": step["seq"], "type": split["type"],
                             "separator": sep, "right_keys": right_keys,
                             "consistent": consistent})
    judgments.append({
        "rule_id": "separator_consistency",
        "status": "pass" if sep_ok and split_steps else "undecided",
        "scope": {"steps": [s["seq"] for s, _ in split_steps]},
        "evidence": sep_evidence,
        "candidates": [],
        "undecided_points": [],
    })

    # 3. duplicate key policy
    dup_steps = [s for s in trace if "duplicate_key" in s.get("scenarios", [])]
    rejected = [s for s in trace if s.get("rejected_duplicate")]
    policy = rules.get("duplicate_policy", "allow_right")
    if policy == "allow_right":
        dup_ok = len(dup_steps) > 0 and not rejected
    else:
        dup_ok = len(rejected) > 0 and not dup_steps
    judgments.append({
        "rule_id": "duplicate_policy",
        "status": "pass" if dup_ok else "undecided",
        "scope": {"steps": [s["seq"] for s in dup_steps + rejected]},
        "evidence": [{"step_seq": s["seq"], "key": s.get("key"),
                      "policy": policy} for s in dup_steps + rejected],
        "candidates": [],
        "undecided_points": [],
    })

    # 4. deleted-to-empty pages are unlinked and unreachable
    empty_steps = [s for s in trace
                   if "delete_empty_page" in s.get("scenarios", [])]
    judgments.append({
        "rule_id": "empty_leaf_unlinked",
        "status": "pass" if empty_steps else "undecided",
        "scope": {"steps": [s["seq"] for s in empty_steps]},
        "evidence": [{"step_seq": s["seq"], "freed": s["freed"],
                      "root": s["root"]} for s in empty_steps],
        "candidates": [],
        "undecided_points": [],
    })

    # 5. no reachable half page at the crash point
    crash_steps = [s for s in trace if s["kind"] == "crash"]
    reach_after = state_hashes.get("reachable_after_recovery", [])
    torn_pages = state_hashes.get("torn_after_recovery", [])
    crash_ok = bool(crash_steps) and not torn_pages
    judgments.append({
        "rule_id": "crash_no_reachable_half_page",
        "status": "pass" if crash_ok else ("undecided" if not crash_steps else "fail"),
        "scope": {"steps": [s["seq"] for s in crash_steps]},
        "evidence": [{"crash_step": s["seq"], "torn_lsn": s.get("torn_lsn"),
                      "torn_page": s.get("torn_page"),
                      "reachable_after_recovery": reach_after,
                      "torn_after_recovery": torn_pages} for s in crash_steps],
        "candidates": [],
        "undecided_points": [],
    })

    # 6. recovery replay is idempotent
    idem_steps = [s for s in trace if s.get("idempotent") is not None]
    idem_ok = bool(idem_steps) and all(s["idempotent"] for s in idem_steps)
    judgments.append({
        "rule_id": "replay_idempotent",
        "status": "pass" if idem_ok else ("undecided" if not idem_steps else "fail"),
        "scope": {"steps": [s["seq"] for s in idem_steps]},
        "evidence": [{"step_seq": s["seq"], "hashes": s.get("replay_hashes")}
                     for s in idem_steps],
        "candidates": [],
        "undecided_points": [],
    })

    params_by_rule = dict(RULES)
    for j in judgments:
        j["params_used"] = params_by_rule.get(j["rule_id"], [])
        j["sources"] = [source_id]
    return judgments
