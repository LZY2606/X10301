"""Review server: tree / byte layout / split propagation / log / crash views.

Reads never mutate; every ruling, undo, parameter change or import is an
append-only event. Concurrent saves are checked with optimistic base heads.
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import casepack
from store import Store, VersionConflict

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DEFAULT_DB = os.environ.get("ARCHIVE_DB", "archive.db")


def create_app(db_path=DEFAULT_DB):
    store = Store(db_path)
    app = FastAPI(title="页裂档案：B-tree 页面布局、分裂传播与恢复检查")
    app.state.store = store

    def _branch_or_404(branch_id):
        if not store.get_branch(branch_id):
            raise HTTPException(404, "unknown branch %s" % branch_id)

    def _trace_for(branch_id):
        events = store.events(branch_id)
        source_id = None
        for event in events:
            if event["kind"] == "import":
                source_id = event["payload"]["source"]
        if source_id is None:
            raise HTTPException(404, "branch has no imported source")
        source = store.get_source(source_id)
        content = json.loads(source["content"])
        return content, source

    def _step_seq_for(branch_id, at_event):
        if at_event is None:
            return None
        event = store.get_event(at_event)
        if event is None:
            raise HTTPException(404, "unknown event %s" % at_event)
        if event["kind"] == "run_step":
            return event["payload"]["step_seq"]
        seqs = [e["payload"]["step_seq"] for e in store.events(branch_id)
                if e["kind"] == "run_step" and e["seq"] <= event["seq"]]
        return seqs[-1] if seqs else None

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(os.path.join(STATIC_DIR, "index.html"),
                  encoding="utf-8") as handle:
            return handle.read()

    @app.get("/api/state")
    def state():
        branches = store.list_branches()
        return {
            "loaded": store.get_source("src-baseline") is not None,
            "branches": [{**b, "head": store.head(b["id"])} for b in branches],
            "counts": {
                "sources": len(store.export_all()["sources"]),
                "events": len(store.export_all()["events"]),
            },
            "diagnostics": store.list_diagnostics(),
        }

    @app.post("/api/case/load")
    def load_case():
        if store.get_source("src-baseline") is not None:
            return {"loaded": True, "already": True}
        return {"loaded": True, **casepack.build_archive(store)}

    @app.get("/api/branches")
    def branches():
        return [{**b, "head": store.head(b["id"])}
                for b in store.list_branches()]

    class BranchIn(BaseModel):
        name: str
        parent: str
        base_event: str | None = None

    @app.post("/api/branches")
    def create_branch(body: BranchIn):
        existing = {b["id"] for b in store.list_branches()}
        branch_id = "branch-%02d" % (len(existing) + 1)
        store.create_branch(branch_id, body.name, parent=body.parent,
                            base_event=body.base_event)
        return store.get_branch(branch_id)

    @app.get("/api/branches/{branch_id}/events")
    def branch_events(branch_id: str):
        _branch_or_404(branch_id)
        return store.events(branch_id)

    @app.get("/api/events/{event_id}")
    def event_detail(event_id: str):
        event = store.get_event(event_id)
        if event is None:
            raise HTTPException(404, "unknown event")
        visible = store.visible_state(event["branch"], at_event_id=event_id)
        return {"event": event, "visible_at_event": visible}

    @app.get("/api/branches/{branch_id}/visible")
    def visible(branch_id: str, at_event: str | None = None):
        _branch_or_404(branch_id)
        return store.visible_state(branch_id, at_event_id=at_event)

    @app.get("/api/run/tree")
    def run_tree(branch: str = "main", at_event: str | None = None):
        _branch_or_404(branch)
        content, _ = _trace_for(branch)
        step_seq = _step_seq_for(branch, at_event)
        return casepack.tree_view(content["trace"],
                                  content["params"]["page_size"], step_seq)

    @app.get("/api/run/page/{page_id}")
    def run_page(page_id: int, branch: str = "main",
                 at_event: str | None = None):
        _branch_or_404(branch)
        content, _ = _trace_for(branch)
        step_seq = _step_seq_for(branch, at_event)
        try:
            return casepack.page_view(content["trace"],
                                      content["params"]["page_size"],
                                      page_id, step_seq)
        except KeyError:
            raise HTTPException(404, "page %s not present at that step"
                                % page_id)

    @app.get("/api/run/splits")
    def run_splits(branch: str = "main", at_event: str | None = None):
        _branch_or_404(branch)
        content, _ = _trace_for(branch)
        step_seq = _step_seq_for(branch, at_event)
        out = []
        for step in content["trace"]:
            if step_seq is not None and step["seq"] > step_seq:
                break
            for split in step.get("splits", []):
                out.append({"step_seq": step["seq"], "id": "split:%d" % step["seq"],
                            **split})
        return out

    @app.get("/api/run/log")
    def run_log(branch: str = "main", at_event: str | None = None):
        _branch_or_404(branch)
        content, _ = _trace_for(branch)
        step_seq = _step_seq_for(branch, at_event)
        max_lsn = None
        if step_seq is not None:
            step = content["trace"][step_seq]
            max_lsn = step["lsn_range"][1]
        records = []
        for record in content["log"]:
            if max_lsn is not None and record["lsn"] > max_lsn:
                continue
            records.append({k: v for k, v in record.items()
                            if k not in ("pre_image", "post_image")})
        return records

    @app.get("/api/run/reachable")
    def run_reachable(branch: str = "main", at_event: str | None = None):
        _branch_or_404(branch)
        content, _ = _trace_for(branch)
        step_seq = _step_seq_for(branch, at_event)
        tree = casepack.tree_view(content["trace"],
                                  content["params"]["page_size"], step_seq)
        crash = next((s for s in reversed(content["trace"])
                      if s["kind"] == "crash" and
                      (step_seq is None or s["seq"] <= step_seq)), None)
        return {
            "step_seq": tree["step_seq"],
            "reachable": [n for n in tree["nodes"] if not n["freed"]],
            "crash": crash and {
                "step_seq": crash["seq"], "torn_lsn": crash["torn_lsn"],
                "torn_page": "pg:%d" % crash["torn_page"],
            },
            "torn_pages": [n["stable_id"] for n in tree["nodes"] if n["torn"]],
        }

    @app.get("/api/invariants")
    def invariant_events(branch: str = "main"):
        _branch_or_404(branch)
        return [e for e in store.events(branch) if e["kind"] == "judgment"]

    @app.get("/api/diagnostics")
    def diagnostics():
        return store.list_diagnostics()

    class RulingIn(BaseModel):
        decision_point: str
        candidate: str
        action: str = "accept"
        base_head: str | None = None
        note: str = ""

    @app.post("/api/branches/{branch_id}/rulings")
    def save_ruling(branch_id: str, body: RulingIn):
        _branch_or_404(branch_id)
        payload = {"decision_point": body.decision_point,
                   "candidate": body.candidate, "action": body.action,
                   "note": body.note}
        try:
            event = store.save_guarded(branch_id, "ruling", payload,
                                       body.base_head)
        except VersionConflict as conflict:
            return JSONResponse(status_code=409, content={
                "error": "version_conflict",
                "current_head": conflict.current_head,
                "events_since_base": conflict.events_since_base,
                "attempted": conflict.attempted,
            })
        return event

    class UndoIn(BaseModel):
        base_head: str | None = None

    @app.post("/api/events/{event_id}/undo")
    def undo_ruling(event_id: str, body: UndoIn):
        event = store.get_event(event_id)
        if event is None or event["kind"] != "ruling":
            raise HTTPException(404, "no ruling event %s" % event_id)
        try:
            return store.save_guarded(event["branch"], "ruling_undo",
                                      {"undoes": event_id}, body.base_head)
        except VersionConflict as conflict:
            return JSONResponse(status_code=409, content={
                "error": "version_conflict",
                "current_head": conflict.current_head,
                "events_since_base": conflict.events_since_base,
                "attempted": conflict.attempted,
            })

    class ParamIn(BaseModel):
        changes: dict
        base_head: str | None = None
        note: str = ""

    @app.post("/api/branches/{branch_id}/params")
    def revise_params(branch_id: str, body: ParamIn):
        _branch_or_404(branch_id)
        visible = store.visible_state(branch_id)
        parent_rules = visible["rule_version"] or "rules-v1"
        parent = store.get_rule_version(parent_rules)
        new_params = dict(parent["params"])
        new_params.update(body.changes)
        version_id = "rules-v%02d" % (len(store.export_all()["rule_versions"]) + 1)
        store.add_rule_version(version_id, new_params, parent=parent_rules)
        try:
            revision = store.save_guarded(branch_id, "param_revision", {
                "new_rule_version": version_id, "changes": body.changes,
                "note": body.note,
            }, body.base_head, rule_version=version_id)
        except VersionConflict as conflict:
            return JSONResponse(status_code=409, content={
                "error": "version_conflict",
                "current_head": conflict.current_head,
                "events_since_base": conflict.events_since_base,
                "attempted": conflict.attempted,
            })
        invalidation, affected = store.invalidate_for_params(
            branch_id, parent_rules, list(body.changes.keys()),
            body.note or "parameter revision")
        return {"revision": revision, "invalidation": invalidation,
                "invalidated": affected}

    @app.get("/api/export")
    def export_all():
        return store.export_all()

    @app.post("/api/import")
    async def import_all(request: Request):
        bundle = await request.json()
        try:
            store.import_all(bundle)
        except Exception as exc:
            raise HTTPException(400, "import rolled back: %s" % exc)
        return {"imported": True,
                "events": len(store.export_all()["events"])}

    return app


app = create_app()
