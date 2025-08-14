-- Medallion Architecture (Bronze → Silver → Gold) - SQLite DDL
-- Layer: Core tables for Silver facts/dimensions, Bronze registry, and Gold sample view

PRAGMA foreign_keys = ON;

-- Registry of dynamically created Bronze tables
CREATE TABLE IF NOT EXISTS bronze_registry (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  table_name TEXT NOT NULL UNIQUE,
  file_slug TEXT NOT NULL,
  sheet_slug TEXT NOT NULL,
  source_file TEXT NOT NULL,
  sheet_name TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimensions (Silver)
CREATE TABLE IF NOT EXISTS dim_date (
  date_key INTEGER PRIMARY KEY,                   -- yyyymmdd
  date_value DATE NOT NULL,
  year INTEGER NOT NULL,
  quarter INTEGER NOT NULL,
  month INTEGER NOT NULL,
  day INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_site (
  site_id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  site_code TEXT,
  site_name TEXT,
  city TEXT,
  region TEXT,
  country TEXT
);

CREATE TABLE IF NOT EXISTS dim_person (
  person_id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  external_ref TEXT,
  first_name TEXT,
  last_name TEXT,
  email TEXT
);

CREATE TABLE IF NOT EXISTS dim_supplier (
  supplier_id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  supplier_code TEXT,
  supplier_name TEXT,
  category TEXT
);

-- Facts (Silver) with lineage
CREATE TABLE IF NOT EXISTS fact_volunteering (
  fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  date_key INTEGER NOT NULL,
  person_id INTEGER,
  site_id INTEGER,
  hours REAL NOT NULL,
  unit TEXT DEFAULT 'hours',
  bronze_table TEXT NOT NULL,
  bronze_row_ids TEXT NOT NULL,                  -- comma-delimited list
  transform_version TEXT NOT NULL DEFAULT 'v1',
  FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
  FOREIGN KEY (person_id) REFERENCES dim_person(person_id),
  FOREIGN KEY (site_id) REFERENCES dim_site(site_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_volunteering_tenant ON fact_volunteering(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fact_volunteering_person ON fact_volunteering(person_id);
CREATE INDEX IF NOT EXISTS idx_fact_volunteering_site ON fact_volunteering(site_id);

CREATE TABLE IF NOT EXISTS fact_donations (
  fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  date_key INTEGER NOT NULL,
  site_id INTEGER,
  amount REAL NOT NULL,
  currency TEXT DEFAULT 'GBP',
  bronze_table TEXT NOT NULL,
  bronze_row_ids TEXT NOT NULL,
  transform_version TEXT NOT NULL DEFAULT 'v1',
  FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
  FOREIGN KEY (site_id) REFERENCES dim_site(site_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_donations_tenant ON fact_donations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fact_donations_site ON fact_donations(site_id);

CREATE TABLE IF NOT EXISTS fact_procurement (
  fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  date_key INTEGER NOT NULL,
  supplier_id INTEGER,
  amount REAL NOT NULL,
  currency TEXT DEFAULT 'GBP',
  local_flag INTEGER DEFAULT 0,
  bronze_table TEXT NOT NULL,
  bronze_row_ids TEXT NOT NULL,
  transform_version TEXT NOT NULL DEFAULT 'v1',
  FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
  FOREIGN KEY (supplier_id) REFERENCES dim_supplier(supplier_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_procurement_tenant ON fact_procurement(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fact_procurement_supplier ON fact_procurement(supplier_id);

CREATE TABLE IF NOT EXISTS fact_energy (
  fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  date_key INTEGER NOT NULL,
  site_id INTEGER,
  consumption_kwh REAL NOT NULL,
  scope TEXT,                                    -- scope1/2/3 or fuel type
  bronze_table TEXT NOT NULL,
  bronze_row_ids TEXT NOT NULL,
  transform_version TEXT NOT NULL DEFAULT 'v1',
  FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
  FOREIGN KEY (site_id) REFERENCES dim_site(site_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_energy_tenant ON fact_energy(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fact_energy_site ON fact_energy(site_id);

CREATE TABLE IF NOT EXISTS fact_waste (
  fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  date_key INTEGER NOT NULL,
  site_id INTEGER,
  tonnes REAL NOT NULL,
  stream TEXT,                                   -- recycled/general/hazardous
  bronze_table TEXT NOT NULL,
  bronze_row_ids TEXT NOT NULL,
  transform_version TEXT NOT NULL DEFAULT 'v1',
  FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
  FOREIGN KEY (site_id) REFERENCES dim_site(site_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_waste_tenant ON fact_waste(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fact_waste_site ON fact_waste(site_id);

-- Convenience indices
CREATE INDEX IF NOT EXISTS idx_fact_volunteering_date ON fact_volunteering(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_donations_date ON fact_donations(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_procurement_date ON fact_procurement(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_energy_date ON fact_energy(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_waste_date ON fact_waste(date_key);

-- Gold sample: monthly volunteer hours view
DROP VIEW IF EXISTS gold_volunteer_hours_monthly;
CREATE VIEW gold_volunteer_hours_monthly AS
SELECT 
  v.tenant_id,
  substr(CAST(d.date_key AS TEXT), 1, 6) AS yyyymm,
  SUM(v.hours) AS total_hours
FROM fact_volunteering v
JOIN dim_date d ON d.date_key = v.date_key
GROUP BY v.tenant_id, substr(CAST(d.date_key AS TEXT), 1, 6)
ORDER BY yyyymm;


