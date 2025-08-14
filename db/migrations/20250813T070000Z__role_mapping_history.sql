-- Role mapping history for learning and success tracking
PRAGMA foreign_keys = ON;

-- Track success/failure of role-to-header mappings for learning
CREATE TABLE IF NOT EXISTS role_mapping_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  fact_key TEXT NOT NULL,
  role TEXT NOT NULL,
  header TEXT NOT NULL,
  normalized_header TEXT NOT NULL,
  success BOOLEAN NOT NULL,
  score REAL,
  component_scores TEXT,  -- JSON with detailed scores
  spec_hash TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_role_mapping_tenant_role_header 
ON role_mapping_history (tenant_id, role, normalized_header);

CREATE INDEX IF NOT EXISTS idx_role_mapping_fact 
ON role_mapping_history (fact_key);

CREATE INDEX IF NOT EXISTS idx_role_mapping_created 
ON role_mapping_history (created_at);

-- Track spec generation success for overall system learning
CREATE TABLE IF NOT EXISTS spec_generation_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  sheet_name TEXT NOT NULL,
  bronze_table TEXT NOT NULL,
  fact_key TEXT NOT NULL,
  spec_hash TEXT NOT NULL,
  success BOOLEAN NOT NULL,
  validation_errors TEXT,  -- JSON array of validation error messages
  quality_score REAL,
  generation_method TEXT,  -- 'semantic', 'heuristic', 'llm_assisted'
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_spec_generation_tenant_fact 
ON spec_generation_history (tenant_id, fact_key);

CREATE INDEX IF NOT EXISTS idx_spec_generation_sheet 
ON spec_generation_history (sheet_name);

CREATE INDEX IF NOT EXISTS idx_spec_generation_created 
ON spec_generation_history (created_at);
