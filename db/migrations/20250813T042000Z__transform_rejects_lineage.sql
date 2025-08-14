-- Transform rejects and lineage bridge tables
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS transform_rejects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  transform_run_id INTEGER NOT NULL,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  bronze_table TEXT NOT NULL,
  bronze_row_id INTEGER,
  reason TEXT NOT NULL,
  raw_snapshot TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (transform_run_id) REFERENCES transform_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transform_rejects_run ON transform_rejects(transform_run_id);
CREATE INDEX IF NOT EXISTS idx_transform_rejects_bronze ON transform_rejects(tenant_id, bronze_table);

CREATE TABLE IF NOT EXISTS fact_lineage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  fact_table TEXT NOT NULL,
  fact_id INTEGER NOT NULL,
  bronze_table TEXT NOT NULL,
  bronze_row_id INTEGER NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fact_lineage_fact ON fact_lineage(tenant_id, fact_table, fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_lineage_bronze ON fact_lineage(tenant_id, bronze_table, bronze_row_id);


