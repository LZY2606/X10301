"""Counterexample archive: a baseline run plus scenarios that each vary exactly
one direct factor. Adopted and rejected candidates are both preserved, with
machine-readable judgment events explaining why."""
from .engine import Engine, DEFAULT_RULES, SplitFailure
from .invariants import check_all


def _with_committed_hash(page_size, script, rules):
    """Recompute the expected committed hash at each recover op for THIS rule
    set: run the script up to the recover point with the crash removed."""
    if not any(op["op"] == "recover" for op in script):
        return script
    pre = [op for op in script if op["op"] != "crash"]
    cut = next((i for i, op in enumerate(pre) if op["op"] == "recover"), len(pre))
    probe = Engine(page_size, dict(rules))
    digest = None
    try:
        probe.run_script(pre[:cut])
        digest = probe.state_hash()
    except Exception:
        digest = None
    out = []
    for op in script:
        if op["op"] == "recover":
            op = dict(op, committed_hash=digest)
        out.append(op)
    return out


def execute(page_size, script, rules):
    """Run a script under a rule set; collect every emitted event + judgments."""
    script = _with_committed_hash(page_size, script, rules)
    events = []
    eng = Engine(page_size, rules, emit=lambda k, p: events.append({"kind": k, "payload": p}))
    error = None
    try:
        eng.run_script(script)
    except SplitFailure as exc:
        error = str(exc)
    if eng.crashed:
        reachable, problems = eng.reachable_pages()
        events.append({"kind": "judgment", "payload": {
            "rule": "crash_no_half_page",
            "status": "pass" if not problems else "fail",
            "details": {"root_id": eng.root_id, "reachable_pages": reachable,
                        "problems": problems, "recovered": False}}})
    judgments = [e["payload"] for e in events if e["kind"] == "judgment"]
    judgments += check_all(eng, events)
    for j in check_all(eng, events):
        events.append({"kind": "judgment", "payload": j})
    if error:
        events.append({"kind": "judgment", "payload": {
            "rule": "split_basis_byte", "status": "fail",
            "details": {"split_failure": error}}})
    return {"events": events, "judgments": judgments, "error": error,
            "root_id": eng.root_id}


def _rules(**over):
    return dict(DEFAULT_RULES, **over)


def _without_recover(script):
    return [op for op in script if op["op"] != "recover"]


def _naive_recover(script):
    return [dict(op, naive=True) if op["op"] == "recover" else op for op in script]


def _judgment_summary(run):
    return {j["rule"]: j["status"] for j in run["judgments"]}


SCENARIOS = [
    {
        "id": "varlen-keys",
        "title": "变长键：按字节空间而非键数决定分裂",
        "factor": "split_basis",
        "candidates": [
            {"label": "byte-basis", "rules": _rules(), "adopted": True},
            {"label": "count-basis", "rules": _rules(split_basis="count"), "adopted": False},
        ],
    },
    {
        "id": "root-split",
        "title": "根分裂：separator 复制规则必须一致",
        "factor": "separator_leaf",
        "candidates": [
            {"label": "leaf-copy-up", "rules": _rules(), "adopted": True},
            {"label": "leaf-promote", "rules": _rules(separator_leaf="promote"),
             "adopted": False},
        ],
    },
    {
        "id": "cascading-split",
        "title": "级联分裂：internal separator 提升规则必须一致",
        "factor": "separator_internal",
        "candidates": [
            {"label": "internal-promote", "rules": _rules(), "adopted": True},
            {"label": "internal-copy-up", "rules": _rules(separator_internal="copy-up"),
             "adopted": False},
        ],
    },
    {
        "id": "duplicate-policy",
        "title": "重复键策略：reject / overwrite / append 均为合法候选",
        "factor": "duplicate_policy",
        "candidates": [
            {"label": "reject", "rules": _rules(), "adopted": True},
            {"label": "overwrite", "rules": _rules(duplicate_policy="overwrite"),
             "adopted": False},
            {"label": "append", "rules": _rules(duplicate_policy="append"),
             "adopted": False},
        ],
    },
    {
        "id": "empty-page",
        "title": "删除后空页：必须从 sibling 链摘除",
        "factor": "empty_leaf",
        "candidates": [
            {"label": "unlink-empty", "rules": _rules(), "adopted": True},
            {"label": "keep-empty", "rules": _rules(empty_leaf="keep"), "adopted": False},
        ],
    },
    {
        "id": "interrupted-write",
        "title": "写页中断：崩溃点不能留下可达半页",
        "factor": "recovery",
        "candidates": [
            {"label": "with-recovery", "rules": _rules(), "adopted": True},
            {"label": "no-recovery", "rules": _rules(), "script_mod": "no_recover",
             "adopted": False},
        ],
    },
    {
        "id": "replay-twice",
        "title": "恢复重放两次：重放必须幂等",
        "factor": "replay",
        "candidates": [
            {"label": "lsn-guarded-replay", "rules": _rules(), "adopted": True},
            {"label": "naive-replay", "rules": _rules(), "script_mod": "naive_recover",
             "adopted": False},
        ],
    },
]

_SCRIPT_MODS = {"no_recover": _without_recover, "naive_recover": _naive_recover}


def run_scenario(scenario, page_size, script):
    """Run every candidate of one scenario; return candidates with judgments."""
    out = {"id": scenario["id"], "title": scenario["title"],
           "factor": scenario["factor"], "candidates": []}
    for cand in scenario["candidates"]:
        mod = _SCRIPT_MODS.get(cand.get("script_mod"))
        cand_script = mod(script) if mod else script
        run = execute(page_size, cand_script, cand["rules"])
        out["candidates"].append({
            "label": cand["label"],
            "adopted": cand["adopted"],
            "rules": cand["rules"],
            "judgments": _judgment_summary(run),
            "error": run["error"],
            "events": run["events"],
        })
    return out
