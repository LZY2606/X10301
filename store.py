"""SQLite persistence: sources, rules, branches, events, candidates, rulings.

Schema versioned with explicit migrations; export/import preserves stable
identity, ordering and provenance.
"""
import json
import sqlite3
import time

SCHEMA_VERSION = 2

SCHEMA_V1 = """
CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT);
CREATE TABLE sources (id TEXT PRIMARY KEY, kind TEXT, raw TEXT, seq INTEGER, imported_at TEXT);
CREATE TABLE rules (version INTEGER PRIMARY KEY, params TEXT, created_at TEXT);
CREATE TABLE branches (id TEXT PRIMARY KEY, name TEXT, parent_id TEXT, rule_version INTEGER, status TEXT);
CREATE TABLE events (id TEXT PRIMARY KEY, branch_id TEXT, seq INTEGER, kind TEXT, payload TEXT, depends_on TEXT, created_at TEXT);
CREATE TABLE candidates (id TEXT PRIMARY KEY, branch_id TEXT, scenario TEXT, label TEXT, adopted_default INTEGER, payload TEXT);
CREATE TABLE rulings (id TEXT PRIMARY KEY, branch_id TEXT, target_id TEXT, decision TEXT, rationale TEXT, base_version INTEGER, seq INTEGER, reverted INTEGER DEFAULT 0, created_at TEXT);
"""

MIGRATIONS = {
    2: [
        "ALTER TABLE events ADD COLUMN status TEXT NOT NULL DEFAULT 'current'",
        """CREATE TABLE diagnostics (
               id TEXT PRIMARY KEY, source_id TEXT, message TEXT, created_at TEXT)""",
    ],
}

TABLES = ["sources", "rules", "branches", "events", "candidates", "rulings", "diagnostics"]


class Conflict(Exception):
    def __init__(self, diff):
        super().__init__("version conflict")
        self.diff = diff


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


class Store:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
        if not cur.fetchone():
            self.conn.executescript(SCHEMA_V1)
            self.conn.execute("INSERT INTO schema_migrations VALUES (1, ?)", (_now(),))
            self.conn.commit()
        self.migrate()

    def migrate(self):
        row = self.conn.execute("SELECT MAX(version) v FROM schema_migrations").fetchone()
        version = row["v"] or 0
        while version < SCHEMA_VERSION:
            version += 1
            for sql in MIGRATIONS[version]:
                self.conn.execute(sql)
            self.conn.execute("INSERT INTO schema_migrations VALUES (?, ?)",
                              (version, _now()))
            self.conn.commit()
        return version

    # -- sources ---------------------------------------------------------
    def add_source(self, sid, kind, raw, seq):
        self.conn.execute(
            "INSERT INTO sources (id, kind, raw, seq, imported_at) VALUES (?,?,?,?,?)",
            (sid, kind, json.dumps(raw, ensure_ascii=False), seq, _now()))

    def sources(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM sources ORDER BY seq").fetchall()]

    def get_source(self, sid):
        r = self.conn.execute("SELECT * FROM sources WHERE id=?", (sid,)).fetchone()
        return dict(r) if r else None

    def revise_source(self, sid, new_raw):
        """Sources are immutable: a revision is a new row that supersedes the old."""
        old = self.get_source(sid)
        if not old:
            raise KeyError(sid)
        n = old["id"].count("#r") + 2 if "#r" in old["id"] else 2
        new_id = f"{sid}#r{n}"
        seq = self.conn.execute("SELECT COALESCE(MAX(seq),0)+1 s FROM sources").fetchone()["s"]
        self.add_source(new_id, old["kind"] + "-revision",
                        {"supersedes": sid, "raw": new_raw}, seq)
        return new_id

    # -- rules -----------------------------------------------------------
    def add_rules(self, params):
        row = self.conn.execute("SELECT COALESCE(MAX(version),0)+1 v FROM rules").fetchone()
        self.conn.execute("INSERT INTO rules (version, params, created_at) VALUES (?,?,?)",
                          (row["v"], json.dumps(params, ensure_ascii=False), _now()))
        return row["v"]

    def rules(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM rules ORDER BY version").fetchall()]

    # -- branches ----------------------------------------------------------
    def create_branch(self, bid, name, parent_id, rule_version):
        self.conn.execute(
            "INSERT INTO branches (id, name, parent_id, rule_version, status) VALUES (?,?,?,?,?)",
            (bid, name, parent_id, rule_version, "open"))

    def branches(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM branches").fetchall()]

    # -- events ------------------------------------------------------------
    def add_event(self, branch_id, kind, payload, depends_on, status="current"):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 s FROM events WHERE branch_id=?",
            (branch_id,)).fetchone()
        eid = f"evt-{branch_id}-{row['s']:05d}"
        self.conn.execute(
            "INSERT INTO events (id, branch_id, seq, kind, payload, depends_on, created_at, status)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (eid, branch_id, row["s"], kind, json.dumps(payload, ensure_ascii=False),
             json.dumps(sorted(depends_on)), _now(), status))
        return eid

    def events(self, branch_id=None, include_stale=True):
        sql = "SELECT * FROM events"
        args = []
        if branch_id:
            sql += " WHERE branch_id=?"
            args.append(branch_id)
        sql += " ORDER BY branch_id, seq"
        rows = [dict(r) for r in self.conn.execute(sql, args).fetchall()]
        for r in rows:
            r["payload"] = json.loads(r["payload"])
            r["depends_on"] = json.loads(r["depends_on"])
        if not include_stale:
            rows = [r for r in rows if r["status"] == "current"]
        return rows

    def get_event(self, eid):
        r = self.conn.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        d["depends_on"] = json.loads(d["depends_on"])
        return d

    def invalidate(self, branch_id, seed_ids, changed):
        """Mark seed events stale, then propagate along dependency edges.
        Records one invalidation event describing the scope."""
        rows = self.events(branch_id)
        scope, frontier = [], set(seed_ids)
        for r in rows:
            if r["id"] in frontier and r["status"] == "current":
                self.conn.execute("UPDATE events SET status='stale' WHERE id=?", (r["id"],))
                scope.append(r["id"])
        changed_in_loop = True
        while changed_in_loop:
            changed_in_loop = False
            for r in rows:
                if r["status"] != "current" or r["id"] in frontier:
                    continue
                if frontier & set(r["depends_on"]):
                    self.conn.execute("UPDATE events SET status='stale' WHERE id=?",
                                      (r["id"],))
                    scope.append(r["id"])
                    frontier.add(r["id"])
                    changed_in_loop = True
        if scope:
            self.add_event(branch_id, "invalidation",
                           {"changed": sorted(changed), "scope": scope},
                           depends_on=set())
        self.conn.commit()
        return scope

    # -- candidates / rulings --------------------------------------------
    def add_candidate(self, cid, branch_id, scenario, label, adopted_default, payload):
        self.conn.execute(
            "INSERT INTO candidates VALUES (?,?,?,?,?,?)",
            (cid, branch_id, scenario, label, int(adopted_default),
             json.dumps(payload, ensure_ascii=False)))

    def candidates(self, branch_id=None):
        sql = "SELECT * FROM candidates"
        args = []
        if branch_id:
            sql += " WHERE branch_id=?"
            args.append(branch_id)
        rows = [dict(r) for r in self.conn.execute(sql, args).fetchall()]
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def rulings(self, branch_id=None):
        sql = "SELECT * FROM rulings"
        args = []
        if branch_id:
            sql += " WHERE branch_id=?"
            args.append(branch_id)
        sql += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def head_version(self, branch_id):
        return self.conn.execute(
            "SELECT COUNT(*) c FROM rulings WHERE branch_id=?", (branch_id,)).fetchone()["c"]

    def candidate_states(self, branch_id):
        """Effective adopted state per candidate: defaults overridden by
        non-reverted rulings in order."""
        states = {c["id"]: bool(c["adopted_default"]) for c in self.candidates(branch_id)}
        for r in self.rulings(branch_id):
            if r["reverted"]:
                continue
            if r["decision"] == "accept":
                states[r["target_id"]] = True
            elif r["decision"] == "reject":
                states[r["target_id"]] = False
        return states

    def add_ruling(self, branch_id, target_id, decision, rationale, base_version):
        head = self.head_version(branch_id)
        if base_version != head:
            raise Conflict({
                "your_base_version": base_version,
                "current_head_version": head,
                "rulings_since_base": [r for r in self.rulings(branch_id)
                                       if r["seq"] > base_version],
                "current_candidate_states": self.candidate_states(branch_id),
            })
        rid = f"rul-{branch_id}-{head + 1:03d}"
        self.conn.execute(
            "INSERT INTO rulings VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, branch_id, target_id, decision, rationale, base_version,
             head + 1, 0, _now()))
        self.conn.commit()
        return rid

    def undo_ruling(self, ruling_id):
        r = self.conn.execute("SELECT * FROM rulings WHERE id=?", (ruling_id,)).fetchone()
        if not r or r["reverted"]:
            raise KeyError(ruling_id)
        self.conn.execute("UPDATE rulings SET reverted=1 WHERE id=?", (ruling_id,))
        head = self.head_version(r["branch_id"])
        rid = f"rul-{r['branch_id']}-{head + 1:03d}"
        self.conn.execute(
            "INSERT INTO rulings VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, r["branch_id"], ruling_id, "revert",
             f"undo {ruling_id}", head, head + 1, 0, _now()))
        self.conn.commit()
        return rid

    # -- diagnostics -------------------------------------------------------
    def add_diagnostic(self, did, source_id, message):
        self.conn.execute("INSERT INTO diagnostics VALUES (?,?,?,?)",
                          (did, source_id, message, _now()))

    def diagnostics(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM diagnostics").fetchall()]

    # -- export / import ---------------------------------------------------
    def export_all(self):
        out = {"schema_version": SCHEMA_VERSION, "tables": {}}
        for t in TABLES:
            try:
                out["tables"][t] = [dict(r) for r in
                                    self.conn.execute(f"SELECT * FROM {t}").fetchall()]
            except sqlite3.OperationalError:
                out["tables"][t] = []
        return out

    def import_all(self, data):
        with self.conn:
            for t in TABLES:
                self.conn.execute(f"DELETE FROM {t}")
            for t, rows in data["tables"].items():
                if t not in TABLES:
                    continue
                for row in rows:
                    cols = list(row.keys())
                    self.conn.execute(
                        f"INSERT INTO {t} ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
                        [row[c] for c in cols])

    def reset(self):
        with self.conn:
            for t in TABLES:
                self.conn.execute(f"DELETE FROM {t}")
