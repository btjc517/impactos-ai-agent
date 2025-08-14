-- Transform jobs queue
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS transform_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL,
  sheet_registry_id INTEGER NOT NULL,
  bronze_table TEXT NOT NULL,
  spec_id TEXT NOT NULL,
  spec_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','running','succeeded','failed','skipped')),
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  started_at TIMESTAMP,
  finished_at TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_transform_jobs_dedup ON transform_jobs(tenant_id, sheet_registry_id, spec_hash);
CREATE INDEX IF NOT EXISTS idx_transform_jobs_status_created ON transform_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_transform_jobs_sheet ON transform_jobs(tenant_id, sheet_registry_id);

-- Extend transform_runs with job and sheet linkage
ALTER TABLE transform_runs ADD COLUMN transform_job_id INTEGER;
ALTER TABLE transform_runs ADD COLUMN sheet_registry_id INTEGER;


