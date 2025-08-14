-- Sheet registry for Bronze ingestion metadata (SQLite)
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sheet_registry (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  source_file TEXT NOT NULL,
  sheet_name TEXT NOT NULL,
  bronze_table TEXT NOT NULL,
  column_types TEXT,              -- JSON string
  example_values TEXT,            -- JSON string
  inferred_date_cols TEXT,        -- JSON string array
  header_row_idx INTEGER,
  content_hash TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sheet_registry_tenant ON sheet_registry(tenant_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_sheet_registry_hash
ON sheet_registry(tenant_id, source_file, sheet_name, content_hash);


