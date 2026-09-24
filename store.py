"""Append-only event archive for the page-split review case.

SQLite-backed, fully immutable history:
  * every logical observation, candidate, ruling, parameter revision, source
    change and invalidation is an event with a stable identity, global order
    (seq), provenance links (sources) and a parameter version;
  * rulings fork branches, never overwrite prior data; undo is itself an event;
  * saves carry an optimistic branch version; a stale commit gets a conflict
    plus the two sides' diff;
  * parameter / source changes compute an invalidation scope from dependency
    links instead of silently replacing conclusions;
  * export/import preserves identities, order and provenance and is one
    transaction (failed imports leave no partial records).
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

SCHEMA_VERSION = 2

DDL_V1 = """
INSERT INTO meta(key, value) VALUES ('schema_version', '1');

CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,
    branch_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    params_version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE branches (
    branch_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_branch TEXT,
    parent_event TEXT,
    head_event TEXT,
    version INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
"""


def migrate(conn: sqlite3.Connection) -> int:
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.executescript(DDL_V1)
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
    version = int(row["value"])
    if version < 2:
        _migrate_v2(conn)
        version = 2
    return version


def _migrate_v2(conn: sqlite3.Connection) -> None:
    conn.execute(
        "ALTER TABLE events ADD COLUMN diagnostics TEXT NOT NULL DEFAULT '[]'"
    )
    conn.execute(
        """CREATE TABLE annotations (
               annotation_id TEXT PRIMARY KEY,
               event_id TEXT NOT NULL,
               note TEXT NOT NULL,
               created_at TEXT NOT NULL
           )"""
    )
    conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
    conn.commit()


class ConflictError(Exception):
    """Optimistic concurrency loss; carries both sides for the UI diff."""

    def __init__(self, expected_version: int, current_version: int,
                 their_events: list[dict], your_payload: dict):
        self.expected_version = expected_version
        self.current_version = current_version
        self.their_events = their_events
        self.your_payload = your_payload
        super().__init__(
            f"version conflict: base v{expected_version}, current v{current_version}"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.schema_version = migrate(self.conn)

    def close(self) -> None:
        self.conn.close()

    # -- branches -----------------------------------------------------------
    def ensure_main_branch(self) -> str:
        row = self.conn.execute(
            "SELECT branch_id FROM branches WHERE name='main'"
        ).fetchone()
        if row:
            return row["branch_id"]
        return self.create_branch("main", parent_branch=None, parent_event=None)

    def create_branch(self, name: str, parent_branch: str | None,
                      parent_event: str | None = None,
                      branch_id: str | None = None) -> str:
        branch_id = branch_id or f"br-{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO branches
               (branch_id, name, parent_branch, parent_event, head_event,
                version, status, created_at)
               VALUES (?, ?, ?, ?, NULL, 0, 'open', ?)""",
            (branch_id, name, parent_branch, parent_event, _now()),
        )
        self.conn.commit()
        return branch_id

    def get_branch(self, branch_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM branches WHERE branch_id=?", (branch_id,)
        ).fetchone()
        if row is None:
            raise KeyError(branch_id)
        return dict(row)

    def list_branches(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM branches ORDER BY rowid").fetchall()]

    # -- events -------------------------------------------------------------
    def append_event(self, branch_id: str, kind: str, payload: dict,
                     sources: list[str] | None = None,
                     diagnostics: list | None = None,
                     event_id: str | None = None,
                     params_version: int | None = None,
                     commit: bool = True) -> dict:
        event_id = event_id or f"ev-{uuid.uuid4().hex[:16]}"
        branch = self.get_branch(branch_id)
        if params_version is None:
            params_version = self.current_params_version(branch_id)
        seq_row = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS s FROM events").fetchone()
        record = {
            "event_id": event_id,
            "seq": seq_row["s"],
            "branch_id": branch_id,
            "kind": kind,
            "payload": payload,
            "sources": sources or [],
            "params_version": params_version,
            "diagnostics": diagnostics or [],
            "created_at": _now(),
        }
        cols = ("event_id", "seq", "branch_id", "kind", "payload", "sources",
                "params_version", "created_at", "diagnostics")
        self.conn.execute(
            f"INSERT INTO events ({','.join(cols)}) VALUES (?,?,?,?,?,?,?,?,?)",
            (record["event_id"], record["seq"], branch_id, kind,
             json.dumps(payload, ensure_ascii=False),
             json.dumps(record["sources"]), params_version,
             record["created_at"], json.dumps(record["diagnostics"])),
        )
        self.conn.execute(
            "UPDATE branches SET head_event=?, version=version+1 WHERE branch_id=?",
            (event_id, branch_id),
        )
        if commit:
            self.conn.commit()
        return record

    def get_event(self, event_id: str) -> dict:
        return self._row_to_event(self.conn.execute(
            "SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone())

    def _row_to_event(self, row: sqlite3.Row) -> dict:
        return {
            "event_id": row["event_id"],
            "seq": row["seq"],
            "branch_id": row["branch_id"],
            "kind": row["kind"],
            "payload": json.loads(row["payload"]),
            "sources": json.loads(row["sources"]),
            "params_version": row["params_version"],
            "diagnostics": json.loads(row["diagnostics"]),
            "created_at": row["created_at"],
        }

    def branch_events(self, branch_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE branch_id=? ORDER BY seq", (branch_id,)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def visible_events(self, branch_id: str) -> list[dict]:
        """Events visible at a branch head: own events up to head plus each
        ancestor branch's events up to the fork point (immutable snapshot)."""
        cutoff_by_branch: dict[str, int] = {}
        current = branch_id
        child_branch: dict | None = None
        while current is not None:
            branch = self.get_branch(current)
            if child_branch is None:
                head = branch["head_event"]
                cutoff_by_branch[current] = 0
                if head is not None:
                    cutoff_by_branch[current] = self.get_event(head)["seq"]
            else:
                fork_event = child_branch["parent_event"]
                cutoff_by_branch[current] = (
                    0 if fork_event is None else self.get_event(fork_event)["seq"]
                )
            child_branch = branch
            current = branch["parent_branch"]
        rows = self.conn.execute("SELECT * FROM events ORDER BY seq").fetchall()
        events = []
        for row in rows:
            cutoff = cutoff_by_branch.get(row["branch_id"])
            if cutoff is not None and row["seq"] <= cutoff:
                events.append(self._row_to_event(row))
        return events

    def current_params_version(self, branch_id: str) -> int:
        versions = [e["params_version"] for e in self.visible_events(branch_id)]
        return max(versions, default=0)

    # -- adjudication workflow ---------------------------------------------
    def adjudicate(self, branch_id: str, candidate_event_id: str, decision: str,
                   expected_version: int, note: str = "") -> tuple[dict, str]:
        """Accept/reject a candidate: forks a new ruling branch.

        Raises ConflictError if expected_version is stale; the caller's store
        state is untouched.
        """
        if decision not in ("accept", "reject"):
            raise ValueError("decision must be accept or reject")
        branch = self.get_branch(branch_id)
        if branch["version"] != expected_version:
            raise ConflictError(
                expected_version, branch["version"],
                self.events_since(branch_id, expected_version),
                {"candidate_event_id": candidate_event_id, "decision": decision,
                 "note": note},
            )
        candidate = self.get_event(candidate_event_id)
        head = branch["head_event"]
        ruling_branch = self.create_branch(
            name=f"{branch['name']}::{decision}:{candidate['payload'].get('code', candidate_event_id)}",
            parent_branch=branch_id,
            parent_event=head,
        )
        event = self.append_event(
            ruling_branch,
            "adjudication",
            {"candidate_event_id": candidate_event_id,
             "decision": decision, "note": note,
             "candidate_code": candidate["payload"].get("code")},
            sources=[candidate_event_id] + candidate["sources"],
            params_version=self.current_params_version(branch_id),
        )
        return event, ruling_branch

    def undo(self, branch_id: str, expected_version: int, reason: str = "") -> dict:
        """Undo the last ruling on this branch; appends an undo event."""
        branch = self.get_branch(branch_id)
        if branch["version"] != expected_version:
            raise ConflictError(
                expected_version, branch["version"],
                self.events_since(branch_id, expected_version),
                {"action": "undo"},
            )
        own = self.branch_events(branch_id)
        rulings = [e for e in own if e["kind"] in ("adjudication", "undo")]
        restored = rulings[-2] if len(rulings) >= 2 else None
        event = self.append_event(
            branch_id, "undo",
            {"restored_event_id": None if restored is None else restored["event_id"],
             "reason": reason},
            sources=[] if restored is None else [restored["event_id"]],
        )
        return event

    def events_since(self, branch_id: str, version: int) -> list[dict]:
        """Other side of a conflict: events appended after the client base."""
        events = self.branch_events(branch_id)
        return events[version:]

    # -- parameter revisions / source changes / invalidation ---------------
    def revise_params(self, branch_id: str, changes: dict,
                      expected_version: int, note: str = "") -> dict:
        branch = self.get_branch(branch_id)
        if branch["version"] != expected_version:
            raise ConflictError(expected_version, branch["version"],
                                self.events_since(branch_id, expected_version),
                                {"action": "revise_params", "changes": changes})
        new_version = self.current_params_version(branch_id) + 1
        event = self.append_event(
            branch_id, "params",
            {"changes": changes, "note": note, "new_version": new_version},
            params_version=new_version,
        )
        affected = self._invalidation_scope(branch_id, changed_params=set(changes),
                                            changed_sources=set())
        invalidation = self.append_event(
            branch_id, "invalidation",
            {"reason": "params", "changed_params": sorted(changes),
             "affected_event_ids": affected, "params_version": new_version},
            sources=affected + [event["event_id"]],
            params_version=new_version,
        )
        return invalidation

    def change_source(self, branch_id: str, source_event_id: str, patch: dict,
                      expected_version: int, note: str = "") -> dict:
        """Localized source edit; only conclusions depending on it are invalidated."""
        branch = self.get_branch(branch_id)
        if branch["version"] != expected_version:
            raise ConflictError(expected_version, branch["version"],
                                self.events_since(branch_id, expected_version),
                                {"action": "change_source"})
        original = self.get_event(source_event_id)
        event = self.append_event(
            branch_id, "source_change",
            {"source_event_id": source_event_id, "patch": patch, "note": note},
            sources=[source_event_id],
        )
        affected = self._invalidation_scope(branch_id, changed_params=set(),
                                            changed_sources={source_event_id})
        return self.append_event(
            branch_id, "invalidation",
            {"reason": "source", "changed_sources": [source_event_id],
             "affected_event_ids": affected},
            sources=affected + [event["event_id"], source_event_id],
        )

    def _invalidation_scope(self, branch_id: str, *,
                            changed_params: set[str],
                            changed_sources: set[str]) -> list[str]:
        affected: list[str] = []
        for event in self.visible_events(branch_id):
            if event["kind"] not in ("verdict", "candidate"):
                continue
            deps = event["payload"].get("deps", {})
            dep_params = set(deps.get("params", []))
            dep_sources = set(deps.get("sources", []))
            if changed_params & dep_params or changed_sources & dep_sources:
                affected.append(event["event_id"])
        return affected

    def stale_event_ids(self, branch_id: str) -> set[str]:
        """Event ids flagged invalidated by any visible invalidation event."""
        stale: set[str] = set()
        for event in self.visible_events(branch_id):
            if event["kind"] == "invalidation":
                stale.update(event["payload"].get("affected_event_ids", []))
        return stale

    # -- export / import ----------------------------------------------------
    def export_dump(self) -> dict:
        events = [self._row_to_event(r) for r in self.conn.execute(
            "SELECT * FROM events ORDER BY seq").fetchall()]
        branches = [dict(r) for r in self.conn.execute(
            "SELECT * FROM branches ORDER BY rowid").fetchall()]
        return {
            "format": "page-split-archive/v1",
            "schema_version": self.schema_version,
            "exported_at": _now(),
            "branches": branches,
            "events": events,
        }

    def import_dump(self, dump: dict) -> dict:
        """Transactional import; identity/order/provenance are preserved."""
        if dump.get("format") != "page-split-archive/v1":
            raise ValueError("unsupported archive format")
        if self.conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]:
            raise ValueError("import target must be empty")
        tx = self.conn  # single transaction, rolls back on any failure
        try:
            with tx:
                for branch in dump["branches"]:
                    tx.execute(
                        """INSERT INTO branches
                           (branch_id, name, parent_branch, parent_event,
                            head_event, version, status, created_at)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (branch["branch_id"], branch["name"],
                         branch["parent_branch"], branch["parent_event"],
                         branch["head_event"], branch["version"],
                         branch.get("status", "open"), branch["created_at"]),
                    )
                for event in sorted(dump["events"], key=lambda e: e["seq"]):
                    tx.execute(
                        """INSERT INTO events
                           (event_id, seq, branch_id, kind, payload, sources,
                            params_version, created_at, diagnostics)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (event["event_id"], event["seq"], event["branch_id"],
                         event["kind"],
                         json.dumps(event["payload"], ensure_ascii=False),
                         json.dumps(event["sources"]),
                         event["params_version"], event["created_at"],
                         json.dumps(event.get("diagnostics", []))),
                    )
        except Exception:
            raise
        return {"branches": len(dump["branches"]),
                "events": len(dump["events"])}
