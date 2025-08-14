-- Concept Graph schema for semantic resolution (SQLite)
PRAGMA foreign_keys = ON;

-- Core concepts (framework, framework_category, file_type, fact, unit, spec)
CREATE TABLE IF NOT EXISTS concept (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  type TEXT NOT NULL,              -- e.g., framework, framework_category, file_type, fact, unit, spec
  key TEXT NOT NULL,               -- canonical key (stable identifier)
  name TEXT NOT NULL,              -- human-friendly name/label
  description TEXT,                -- optional long description
  parent_id INTEGER,               -- parent relationship (e.g., category -> framework)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(type, key),
  FOREIGN KEY (parent_id) REFERENCES concept(id)
);

-- Aliases/synonyms for concepts (multi-language supported)
CREATE TABLE IF NOT EXISTS concept_alias (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_id INTEGER NOT NULL,
  alias TEXT NOT NULL,
  lang TEXT DEFAULT 'en',
  source TEXT,                     -- where this alias came from (seed, user, model, verification)
  confidence REAL DEFAULT 1.0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(concept_id, alias, lang),
  FOREIGN KEY (concept_id) REFERENCES concept(id) ON DELETE CASCADE
);

-- Relations between concepts (typed edges)
CREATE TABLE IF NOT EXISTS concept_relation (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  from_concept_id INTEGER NOT NULL,
  to_concept_id INTEGER NOT NULL,
  relation TEXT NOT NULL,          -- e.g., alias-of, maps-to, implies, parent, sibling
  weight REAL DEFAULT 1.0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (from_concept_id) REFERENCES concept(id) ON DELETE CASCADE,
  FOREIGN KEY (to_concept_id) REFERENCES concept(id) ON DELETE CASCADE
);

-- Resolution events for learning and audit
CREATE TABLE IF NOT EXISTS resolution_event (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  resolved_type TEXT NOT NULL,     -- what we tried to resolve (framework, file_type, spec, etc.)
  input_text TEXT NOT NULL,        -- raw input
  context_json TEXT,               -- optional JSON with headers/sample values/signals
  decided_concept_id INTEGER,      -- chosen concept (nullable if abstained)
  score REAL,                      -- decision score
  outcome TEXT,                    -- accepted/rejected/abstained
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (decided_concept_id) REFERENCES concept(id)
);

-- Seed minimal file types (data-level, not code-level)
INSERT OR IGNORE INTO concept (type, key, name, description) VALUES
  ('file_type','excel','Excel','Microsoft Excel workbook'),
  ('file_type','csv','CSV','Comma-separated values'),
  ('file_type','pdf','PDF','Portable Document Format');

INSERT OR IGNORE INTO concept_alias (concept_id, alias, lang, source, confidence)
SELECT c.id, a.alias, 'en', 'seed', 1.0
FROM concept c
JOIN (
  SELECT 'excel' AS key, 'xlsx' AS alias UNION ALL
  SELECT 'excel', 'xls' UNION ALL
  SELECT 'excel', 'xlsm' UNION ALL
  SELECT 'excel', 'excel' UNION ALL
  SELECT 'csv', 'csv' UNION ALL
  SELECT 'pdf', 'pdf'
) a ON c.key = a.key AND c.type = 'file_type';

-- Seed high-level frameworks as concepts; categories can be added over time
INSERT OR IGNORE INTO concept (type, key, name, description) VALUES
  ('framework','uk_sv_model','UK Social Value Model (MAC)','UK Social Value Model / MAC'),
  ('framework','un_sdgs','UN Sustainable Development Goals','UN 17 Sustainable Development Goals'),
  ('framework','toms','TOMs (Themes, Outcomes and Measures)','National TOMs framework'),
  ('framework','b_corp','B Corp Assessment','B Corporation impact assessment');

INSERT OR IGNORE INTO concept_alias (concept_id, alias, lang, source, confidence)
SELECT c.id, a.alias, 'en', 'seed', 1.0
FROM concept c
JOIN (
  SELECT 'uk_sv_model' AS key, 'uk sv model' AS alias UNION ALL
  SELECT 'uk_sv_model', 'mac' UNION ALL
  SELECT 'un_sdgs', 'un sdgs' UNION ALL
  SELECT 'un_sdgs', 'sdg' UNION ALL
  SELECT 'un_sdgs', 'sdgs' UNION ALL
  SELECT 'toms', 'toms' UNION ALL
  SELECT 'b_corp', 'b corp' UNION ALL
  SELECT 'b_corp', 'b-corp' UNION ALL
  SELECT 'b_corp', 'bcorp'
) a ON c.key = a.key AND c.type = 'framework';

-- Seed basic facts we support
INSERT OR IGNORE INTO concept (type, key, name, description) VALUES
  ('fact','fact_volunteering','Volunteering','Volunteer hours/time'),
  ('fact','fact_donations','Donations','Charitable donations'),
  ('fact','fact_procurement','Procurement','Responsible procurement spend'),
  ('fact','fact_energy','Energy','Energy consumption/emissions'),
  ('fact','fact_waste','Waste','Waste volumes');

INSERT OR IGNORE INTO concept_alias (concept_id, alias, lang, source, confidence)
SELECT c.id, a.alias, 'en', 'seed', 1.0
FROM concept c
JOIN (
  SELECT 'fact_volunteering' AS key, 'volunteering' AS alias UNION ALL
  SELECT 'fact_volunteering', 'volunteer' UNION ALL
  SELECT 'fact_volunteering', 'vol' UNION ALL
  SELECT 'fact_donations', 'donation' UNION ALL
  SELECT 'fact_donations', 'benevity' UNION ALL
  SELECT 'fact_procurement', 'procurement' UNION ALL
  SELECT 'fact_energy', 'energy' UNION ALL
  SELECT 'fact_energy', 'emissions' UNION ALL
  SELECT 'fact_waste', 'waste'
) a ON c.key = a.key AND c.type = 'fact';


