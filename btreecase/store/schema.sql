PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS import_batch (
  batch_id TEXT PRIMARY KEY,
  idempotency_key TEXT UNIQUE NOT NULL,
  source_label TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'committed'  -- committed | rolled_back
);

CREATE TABLE IF NOT EXISTS event (
  event_id TEXT PRIMARY KEY,
  batch_id TEXT NOT NULL REFERENCES import_batch(batch_id),
  branch_id TEXT NOT NULL,
  seq INTEGER NOT NULL,               -- stable order within branch
  kind TEXT NOT NULL,                 -- logical_op | crash_point | verdict | param_change | reinterpret
  payload TEXT NOT NULL,              -- raw JSON, original representation preserved
  payload_hash TEXT NOT NULL,
  source_ref TEXT,                    -- provenance: which batch/record produced this
  parent_event_id TEXT,               -- causal chain
  created_at TEXT NOT NULL,
  superseded_by TEXT                  -- NULL unless a later event supersedes (old stays readable)
);
CREATE INDEX IF NOT EXISTS idx_event_branch ON event(branch_id, seq);

CREATE TABLE IF NOT EXISTS branch (
  branch_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  parent_branch_id TEXT,
  fork_event_id TEXT,                 -- event at which this branch forked
  head_event_id TEXT,
  version INTEGER NOT NULL DEFAULT 0, -- optimistic concurrency
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open' -- open | merged | abandoned
);

CREATE TABLE IF NOT EXISTS verdict (
  verdict_id TEXT PRIMARY KEY,
  branch_id TEXT NOT NULL,
  metric_id TEXT NOT NULL,
  decision TEXT NOT NULL,             -- accept | reject | adjust
  rationale TEXT,
  prior_verdict_id TEXT,              -- undo chain: points at verdict this one reverts
  reverted_by TEXT,
  actor TEXT NOT NULL DEFAULT 'reviewer',
  created_at TEXT NOT NULL,
  base_version INTEGER NOT NULL       -- branch version seen by the reviewer
);

CREATE TABLE IF NOT EXISTS derived_cache (
  branch_id TEXT NOT NULL,
  metric_id TEXT NOT NULL,
  params_version TEXT NOT NULL,
  result TEXT NOT NULL,
  valid INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (branch_id, metric_id, params_version)
);

CREATE TABLE IF NOT EXISTS migration (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);
