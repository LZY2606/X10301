"""页裂档案 — B-tree 页面布局、分裂传播与恢复检查 review tool.

来源(sources)与解释(events/candidates/rulings)分开保存；裁定不改写原始数据；
参数或源变化只按依赖关系计算失效范围。
"""
import json
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from btree.engine import DEFAULT_RULES
from btree.scenarios import SCENARIOS, run_scenario, execute
from btree.units import parse_quantity
from store import Store, Conflict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_PATH = os.path.join(BASE_DIR, "casepack", "split_archive_case.json")
DB_PATH = os.environ.get("ARCHIVE_DB", os.path.join(BASE_DIR, "archive.db"))

app = FastAPI(title="页裂档案")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def get_store():
    store = Store(DB_PATH)
    try:
        yield store
    finally:
        store.conn.close()


from fastapi import Depends  # noqa: E402


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


# ----------------------------------------------------------------------
# case loading / derivation
# ----------------------------------------------------------------------
@app.post("/api/case/load")
def load_case(store: Store = Depends(get_store)):
    try:
        with open(CASE_PATH, encoding="utf-8") as fh:
            case = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"case pack unreadable: {exc}")
    conn = store.conn
    try:
        # ---- import raw sources (no irreversible cleaning) ----
        store.add_source("src-page-size", "quantity", case["quantities"]["page_size"], 0)
        store.add_source("src-cache-hint", "quantity", case["quantities"]["cache_hint"], 1)
        for i, op in enumerate(case["ops"]):
            store.add_source(op["id"], "logical-op", op, i + 2)
        store.add_source("src-chapters", "chapters", case["chapters"], len(case["ops"]) + 2)
        store.add_source("src-meta", "case-meta",
                         {"case_id": case["case_id"], "title": case["title"],
                          "narrative": case["narrative"]}, len(case["ops"]) + 3)

        # ---- derived layer: normalize quantities, keep raw in place ----
        page_size_q = parse_quantity(case["quantities"]["page_size"]["raw"])
        hint_q = parse_quantity(case["quantities"]["cache_hint"]["raw"])
        for did, sid, q in (("diag-page-size", "src-page-size", page_size_q),
                            ("diag-cache-hint", "src-cache-hint", hint_q)):
            if q.get("diagnostic"):
                store.add_diagnostic(did, sid, q["diagnostic"])
        if page_size_q["value"] is None:
            raise ValueError("page_size is not convertible; refusing to derive")
        page_size = int(page_size_q["value"])

        rule_v = store.add_rules(DEFAULT_RULES)
        store.create_branch("main", "基线", None, rule_v)
        rule_ref = f"rules-v{rule_v}"

        # ---- baseline derivation: every fact preserved with provenance ----
        run = execute(page_size, case["ops"], dict(DEFAULT_RULES))
        deps = {rule_ref, "src-page-size"}
        for ev in run["events"]:
            payload = ev["payload"]
            if ev["kind"] == "logical_op" and payload.get("source_id"):
                deps.add(payload["source_id"])
            store.add_event("main", ev["kind"], payload, deps)

        # ---- counterexample scenarios: adopted + rejected candidates ----
        for sc in SCENARIOS:
            out = run_scenario(sc, page_size, case["ops"])
            for cand in out["candidates"]:
                cid = f"cand-{sc['id']}-{cand['label']}"
                store.add_candidate(cid, "main", sc["id"], cand["label"],
                                    cand["adopted"],
                                    {"title": out["title"], "factor": out["factor"],
                                     "rules": cand["rules"],
                                     "judgments": cand["judgments"],
                                     "error": cand["error"]})
                store.add_event("main", "candidate_judgment", {
                    "candidate_id": cid, "scenario": sc["id"],
                    "label": cand["label"], "adopted_by_default": cand["adopted"],
                    "judgments": cand["judgments"], "error": cand["error"]},
                    {rule_ref, "src-page-size"})
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise HTTPException(400, f"import failed, rolled back: {exc}")
    return {"status": "loaded", "case_id": case["case_id"], "title": case["title"]}


# ----------------------------------------------------------------------
# state views
# ----------------------------------------------------------------------
def assemble_state(store: Store, branch_id: str):
    events = store.events(branch_id)
    pages, root_id, log, crashes, splits = {}, None, [], [], []
    judgments, stale_ids = [], set()
    for ev in events:
        k, p = ev["kind"], ev["payload"]
        if ev["status"] == "stale":
            stale_ids.add(ev["id"])
            if k != "invalidation":
                continue
        if k == "page_image":
            pages[p["page_id"]] = {**p, "event_id": ev["id"]}
        elif k == "page_freed":
            pages.pop(p["page_id"], None)
        elif k == "root_change":
            root_id = p["new_root"]
        elif k == "log_record":
            log.append({**p, "event_id": ev["id"]})
        elif k == "crash":
            crashes.append({**p, "event_id": ev["id"]})
        elif k == "page_split":
            splits.append({**p, "event_id": ev["id"]})
        elif k == "judgment":
            judgments.append({**p, "event_id": ev["id"]})
    candidates = store.candidates(branch_id)
    states = store.candidate_states(branch_id)
    for c in candidates:
        c["adopted"] = states[c["id"]]
    return {
        "branch_id": branch_id,
        "head_version": store.head_version(branch_id),
        "root_id": root_id,
        "pages": [pages[k] for k in sorted(pages)],
        "splits": splits,
        "log": log,
        "crashes": crashes,
        "judgments": judgments,
        "candidates": candidates,
        "rulings": store.rulings(branch_id),
        "branches": store.branches(),
        "rules": store.rules(),
        "diagnostics": store.diagnostics(),
        "sources": store.sources(),
        "stale_event_ids": sorted(stale_ids),
        "scenarios": [{"id": s["id"], "title": s["title"]} for s in SCENARIOS],
    }


@app.get("/api/state")
def state(branch_id: str = "main", store: Store = Depends(get_store)):
    return assemble_state(store, branch_id)


@app.get("/api/page/{page_id}/layout")
def page_layout(page_id: int, branch_id: str = "main",
                store: Store = Depends(get_store)):
    image = None
    for ev in store.events(branch_id):
        if ev["kind"] == "page_image" and ev["payload"]["page_id"] == page_id \
                and ev["status"] == "current":
            image = ev
    if image is None:
        raise HTTPException(404, f"no current image for page {page_id}")
    p = image["payload"]
    raw = np.frombuffer(bytes.fromhex(p["hex"]), dtype=np.uint8)
    regions = np.full(raw.shape, "free", dtype=object)
    regions[:32] = "header"
    for slot in p["slot_directory"]:
        regions[slot["offset"]:slot["offset"] + slot["length"]] = "record"
        pos = len(raw) - 4 * (slot["slot"] + 1)
        regions[pos:pos + 4] = "slot"
    rows = []
    for base in range(0, len(raw), 16):
        rows.append({
            "offset": base,
            "bytes": [int(b) for b in raw[base:base + 16]],
            "regions": list(regions[base:base + 16]),
        })
    return {"page_id": page_id, "event_id": image["id"], "lsn": p["lsn"],
            "is_leaf": p["is_leaf"], "free_space": p["free_space"],
            "right_sibling": p["right_sibling"],
            "slot_directory": p["slot_directory"], "keys": p["keys"],
            "rows": rows}


@app.get("/api/events/{event_id}/context")
def event_context(event_id: str, store: Store = Depends(get_store)):
    ev = store.get_event(event_id)
    if not ev:
        raise HTTPException(404, event_id)
    branch_events = store.events(ev["branch_id"])
    invalidated = set()
    for other in branch_events:
        if other["kind"] == "invalidation" and other["seq"] <= ev["seq"]:
            invalidated.update(other["payload"]["scope"])
    visible = [e for e in branch_events
               if e["seq"] <= ev["seq"] and e["id"] not in invalidated
               and e["kind"] == "judgment"]
    sources = {s["id"]: s for s in store.sources()}
    rules = {f"rules-v{r['version']}": json.loads(r["params"]) for r in store.rules()}
    deps = []
    for dep in ev["depends_on"]:
        if dep in sources:
            deps.append({"id": dep, "kind": "source", "raw": json.loads(sources[dep]["raw"])})
        elif dep in rules:
            deps.append({"id": dep, "kind": "rules", "params": rules[dep]})
        else:
            deps.append({"id": dep, "kind": "event"})
    return {"event": ev, "dependencies": deps,
            "visible_judgments_at_event": visible}


# ----------------------------------------------------------------------
# rulings (never rewrite source data; undo restores the previous branch state)
# ----------------------------------------------------------------------
class RulingIn(BaseModel):
    branch_id: str = "main"
    candidate_id: str
    decision: str           # accept | reject
    rationale: str = ""
    base_version: int


@app.post("/api/rulings")
def make_ruling(body: RulingIn, store: Store = Depends(get_store)):
    if body.decision not in ("accept", "reject"):
        raise HTTPException(400, "decision must be accept|reject")
    try:
        rid = store.add_ruling(body.branch_id, body.candidate_id, body.decision,
                               body.rationale, body.base_version)
    except Conflict as exc:
        raise HTTPException(409, detail=exc.diff)
    return {"ruling_id": rid, "head_version": store.head_version(body.branch_id)}


@app.post("/api/rulings/{ruling_id}/undo")
def undo_ruling(ruling_id: str, store: Store = Depends(get_store)):
    try:
        rid = store.undo_ruling(ruling_id)
    except KeyError:
        raise HTTPException(404, ruling_id)
    return {"ruling_id": rid}


# ----------------------------------------------------------------------
# revisions: dependency-scoped invalidation, never a global silent replace
# ----------------------------------------------------------------------
class RulesReviseIn(BaseModel):
    branch_id: str = "main"
    params: dict
    affected_source_ids: list[str]


@app.post("/api/rules/revise")
def revise_rules(body: RulesReviseIn, store: Store = Depends(get_store)):
    new_version = store.add_rules({**DEFAULT_RULES, **body.params})
    old_ref = f"rules-v{new_version - 1}"
    affected = set(body.affected_source_ids)
    seed = [e["id"] for e in store.events(body.branch_id)
            if old_ref in e["depends_on"] and affected & set(e["depends_on"])]
    scope = store.invalidate(body.branch_id, seed,
                             changed=[old_ref, f"rules-v{new_version}"] + sorted(affected))
    return {"new_rule_version": new_version, "invalidated": scope}


class SourceReviseIn(BaseModel):
    branch_id: str = "main"
    source_id: str
    new_raw: dict


@app.post("/api/sources/revise")
def revise_source(body: SourceReviseIn, store: Store = Depends(get_store)):
    try:
        new_id = store.revise_source(body.source_id, body.new_raw)
    except KeyError:
        raise HTTPException(404, body.source_id)
    seed = [e["id"] for e in store.events(body.branch_id)
            if body.source_id in e["depends_on"]]
    scope = store.invalidate(body.branch_id, seed, changed=[body.source_id, new_id])
    return {"new_source_id": new_id, "invalidated": scope}


# ----------------------------------------------------------------------
# export / import / reset: identity, order and provenance survive a round trip
# ----------------------------------------------------------------------
@app.get("/api/export")
def export(store: Store = Depends(get_store)):
    return store.export_all()


@app.post("/api/import")
def import_archive(data: dict, store: Store = Depends(get_store)):
    try:
        store.import_all(data)
    except Exception as exc:
        store.conn.rollback()
        raise HTTPException(400, f"import failed, rolled back: {exc}")
    return {"status": "imported"}


@app.post("/api/reset")
def reset(store: Store = Depends(get_store)):
    store.reset()
    return {"status": "reset"}
