"""SQLite persistence: immutable sources, versioned rules, append-only events.

Sources (raw material) and interpretations (events, judgments, rulings) are
stored separately. All writes are append-only; rulings create new events and
can be undone by another event. Concurrent saves use optimistic concurrency:
a stale base head yields a VersionConflict carrying both diffs.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

SCHEMA_VERSION = 2

V1_DDL = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, kind TEXT, content TEXT,
    hash TEXT, seq INTEGER, supersedes TEXT);
CREATE TABLE IF NOT EXISTS rule_versions(id TEXT PRIMARY KEY, params TEXT,
    parent TEXT, seq INTEGER);
CREATE TABLE IF NOT EXISTS branches(id TEXT PRIMARY KEY, name TEXT, parent TEXT,
    base_event TEXT, seq INTEGER);
CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, branch TEXT, seq INTEGER,
    kind TEXT, payload TEXT);
"""

V2_EXTRA = """
ALTER TABLE events ADD COLUMN rule_version TEXT;
CREATE TABLE IF NOT EXISTS diagnostics(id TEXT PRIMARY KEY, source TEXT,
    field TEXT, message TEXT, seq INTEGER);
"""

FULL_DDL = V1_DDL + V2_EXTRA + """
UPDATE meta SET v='2' WHERE k='schema_version';
"""


class VersionConflict(Exception):
    def __init__(self, current_head, events_since_base, attempted):
        self.current_head = current_head
        self.events_since_base = events_since_base
        self.attempted = attempted
        super().__init__("stale base head: %s" % current_head)


def _connect(path):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class Store:
    def __init__(self, path):
        self.path = path
        self.conn = _connect(path)
        self.migrate()

    def migrate(self):
        c = self.conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' "
                  "AND name='meta'")
        if c.fetchone() is None:
            c.executescript(V1_DDL)
            c.execute("INSERT INTO meta(k,v) VALUES('schema_version','1')")
        version = int(self._meta("schema_version"))
        if version < 2:
            c.executescript(V2_EXTRA)
            c.execute("UPDATE meta SET v='2' WHERE k='schema_version'")
        self.conn.commit()
        return SCHEMA_VERSION

    def _meta(self, key):
        row = self.conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return row["v"] if row else None

    # ---- sources / rules --------------------------------------------------

    def add_source(self, source_id, kind, content, supersedes=None):
        import hashlib
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        seq = self._next_seq("sources")
        self.conn.execute(
            "INSERT INTO sources(id,kind,content,hash,seq,supersedes) "
            "VALUES(?,?,?,?,?,?)",
            (source_id, kind, content, digest, seq, supersedes))
        self.conn.commit()
        return digest

    def get_source(self, source_id):
        row = self.conn.execute("SELECT * FROM sources WHERE id=?",
                                (source_id,)).fetchone()
        return dict(row) if row else None

    def add_rule_version(self, version_id, params, parent=None):
        seq = self._next_seq("rule_versions")
        self.conn.execute(
            "INSERT INTO rule_versions(id,params,parent,seq) VALUES(?,?,?,?)",
            (version_id, json.dumps(params, ensure_ascii=False), parent, seq))
        self.conn.commit()

    def get_rule_version(self, version_id):
        row = self.conn.execute("SELECT * FROM rule_versions WHERE id=?",
                                (version_id,)).fetchone()
        if not row:
            return None
        return {"id": row["id"], "params": json.loads(row["params"]),
                "parent": row["parent"]}

    def add_diagnostic(self, source, field, message):
        seq = self._next_seq("diagnostics")
        diag_id = "diag-%04d" % seq
        self.conn.execute(
            "INSERT INTO diagnostics(id,source,field,message,seq) "
            "VALUES(?,?,?,?,?)", (diag_id, source, field, message, seq))
        self.conn.commit()
        return diag_id

    def list_diagnostics(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM diagnostics ORDER BY seq")]

    # ---- branches / events ------------------------------------------------

    def _next_seq(self, table):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 AS n FROM %s" % table).fetchone()
        return row["n"]

    def create_branch(self, branch_id, name, parent=None, base_event=None):
        seq = self._next_seq("branches")
        self.conn.execute(
            "INSERT INTO branches(id,name,parent,base_event,seq) "
            "VALUES(?,?,?,?,?)", (branch_id, name, parent, base_event, seq))
        self.conn.commit()

    def list_branches(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branches ORDER BY seq")]

    def get_branch(self, branch_id):
        row = self.conn.execute("SELECT * FROM branches WHERE id=?",
                                (branch_id,)).fetchone()
        return dict(row) if row else None

    def head(self, branch_id):
        row = self.conn.execute(
            "SELECT id FROM events WHERE branch=? ORDER BY seq DESC LIMIT 1",
            (branch_id,)).fetchone()
        return row["id"] if row else None

    def add_event(self, branch_id, kind, payload, rule_version=None,
                  event_id=None):
        seq = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 AS n FROM events WHERE branch=?",
            (branch_id,)).fetchone()["n"]
        event_id = event_id or "evt-%s-%04d" % (branch_id, seq)
        self.conn.execute(
            "INSERT INTO events(id,branch,seq,kind,payload,rule_version) "
            "VALUES(?,?,?,?,?,?)",
            (event_id, branch_id, seq, kind,
             json.dumps(payload, ensure_ascii=False), rule_version))
        self.conn.commit()
        return self.get_event(event_id)

    def get_event(self, event_id):
        row = self.conn.execute("SELECT * FROM events WHERE id=?",
                                (event_id,)).fetchone()
        if not row:
            return None
        return {"id": row["id"], "branch": row["branch"], "seq": row["seq"],
                "kind": row["kind"], "payload": json.loads(row["payload"]),
                "rule_version": row["rule_version"]}

    def events(self, branch_id):
        return [self.get_event(r["id"]) for r in self.conn.execute(
            "SELECT id FROM events WHERE branch=? ORDER BY seq", (branch_id,))]

    def events_since(self, branch_id, base_event_id):
        if base_event_id is None:
            return self.events(branch_id)
        base = self.get_event(base_event_id)
        return [e for e in self.events(branch_id)
                if base is None or e["seq"] > base["seq"]]

    def save_guarded(self, branch_id, kind, payload, base_head,
                     rule_version=None):
        current = self.head(branch_id)
        if base_head != current:
            raise VersionConflict(
                current, self.events_since(branch_id, base_head),
                {"kind": kind, "payload": payload})
        return self.add_event(branch_id, kind, payload, rule_version)

    # ---- derived view at a point in history ------------------------------

    def visible_state(self, branch_id, at_event_id=None):
        """Fold events up to at_event: visible judgments, rulings, rules.

        Returning to an old event restores that event's rule version and
        visible conclusions instead of overlaying the latest computation.
        """
        events = self.events(branch_id)
        if at_event_id is not None:
            target = self.get_event(at_event_id)
            events = [e for e in events if e["seq"] <= target["seq"]]
        invalidated = set()
        rulings = {}
        decisions = {}
        rule_version = None
        judgments = {}
        for event in events:
            kind = event["kind"]
            payload = event["payload"]
            if event["rule_version"] and kind in ("import", "run_step",
                                                  "judgment"):
                rule_version = event["rule_version"]
            if kind == "param_revision":
                rule_version = payload["new_rule_version"]
            if kind == "invalidate":
                invalidated.update(payload["invalidated"])
            if kind == "decision_point":
                decisions[payload["decision_point"]] = payload
            if kind == "judgment":
                judgments[payload["judgment_id"]] = event
            if kind == "ruling":
                rulings[payload["decision_point"]] = payload
            if kind == "ruling_undo":
                rulings.pop(payload["decision_point"], None)
        visible_judgments = [
            {"event": e, "invalidated": e["payload"]["judgment_id"] in invalidated}
            for e in judgments.values()
        ]
        return {
            "at_event": at_event_id or self.head(branch_id),
            "rule_version": rule_version,
            "judgments": visible_judgments,
            "decisions": decisions,
            "rulings": rulings,
            "invalidated": sorted(invalidated),
        }

    def invalidate_for_params(self, branch_id, rule_version_id, changed_params,
                              reason):
        affected = []
        for event in self.events(branch_id):
            if event["kind"] != "judgment":
                continue
            params_used = event["payload"].get("params_used", [])
            if set(params_used) & set(changed_params):
                affected.append(event["payload"]["judgment_id"])
        event = self.add_event(branch_id, "invalidate", {
            "reason": reason, "changed_params": sorted(changed_params),
            "invalidated": affected, "scope": "dependency",
        })
        return event, affected

    def invalidate_for_sources(self, branch_id, old_source_id, steps, reason):
        affected = []
        step_set = set(steps)
        for event in self.events(branch_id):
            if event["kind"] != "judgment":
                continue
            payload = event["payload"]
            if old_source_id not in payload.get("sources", []):
                continue
            scoped = set(payload.get("scope", {}).get("steps", []))
            if scoped & step_set:
                affected.append(payload["judgment_id"])
        event = self.add_event(branch_id, "invalidate", {
            "reason": reason, "superseded_source": old_source_id,
            "steps": sorted(step_set),
            "invalidated": affected, "scope": "localized",
        })
        return event, affected

    # ---- export / import --------------------------------------------------

    def export_all(self):
        def rows(table, order):
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM %s ORDER BY %s" % (table, order))]
        return {
            "schema_version": SCHEMA_VERSION,
            "sources": rows("sources", "seq"),
            "rule_versions": rows("rule_versions", "seq"),
            "branches": rows("branches", "seq"),
            "events": rows("events", "seq"),
            "diagnostics": rows("diagnostics", "seq"),
        }

    def import_all(self, bundle, merge=False):
        """Load a bundle transactionally; failure rolls back this change only."""
        try:
            self.conn.execute("BEGIN")
            for table in ("sources", "rule_versions", "branches",
                          "diagnostics"):
                for row in bundle.get(table, []):
                    keys = [k for k in row.keys() if
                            k in self._columns(table)]
                    placeholders = ",".join("?" for _ in keys)
                    sql = "INSERT INTO %s(%s) VALUES(%s)" % (
                        table, ",".join(keys), placeholders)
                    self.conn.execute(sql, tuple(row[k] for k in keys))
            for row in bundle.get("events", []):
                keys = [k for k in row.keys() if k in self._columns("events")]
                placeholders = ",".join("?" for _ in keys)
                self.conn.execute(
                    "INSERT INTO events(%s) VALUES(%s)"
                    % (",".join(keys), placeholders),
                    tuple(row[k] for k in keys))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _columns(self, table):
        return {r[1] for r in self.conn.execute(
            "PRAGMA table_info(%s)" % table)}

    def close(self):
        self.conn.close()
