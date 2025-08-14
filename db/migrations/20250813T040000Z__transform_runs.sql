-- Transform runs tracking for Silver population
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS transform_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  bronze_table TEXT NOT NULL,
  transform_name TEXT NOT NULL,
  transform_version TEXT NOT NULL DEFAULT 'v1',
  spec_hash TEXT,
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP,
  status TEXT NOT NULL DEFAULT 'running', -- running|completed|failed
  rows_in INTEGER DEFAULT 0,
  rows_out INTEGER DEFAULT 0,
  rows_rejected INTEGER DEFAULT 0,
  notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_transform_runs_bronze ON transform_runs(tenant_id, bronze_table);
CREATE INDEX IF NOT EXISTS idx_transform_runs_status ON transform_runs(status);


