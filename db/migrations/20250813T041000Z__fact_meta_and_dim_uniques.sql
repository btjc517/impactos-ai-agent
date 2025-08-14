-- Add metadata columns to facts for unit/currency normalization and enforce dim uniques
PRAGMA foreign_keys = ON;

-- Pre-clean dimension duplicates, keeping lowest id per natural key
DELETE FROM dim_site WHERE site_id NOT IN (
  SELECT MIN(site_id) FROM dim_site GROUP BY tenant_id, site_code
);
DELETE FROM dim_person WHERE person_id NOT IN (
  SELECT MIN(person_id) FROM dim_person GROUP BY tenant_id, external_ref
);
DELETE FROM dim_supplier WHERE supplier_id NOT IN (
  SELECT MIN(supplier_id) FROM dim_supplier GROUP BY tenant_id, supplier_code
);

-- Unique indexes for natural keys
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_site_code ON dim_site(tenant_id, site_code);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_person_external_ref ON dim_person(tenant_id, external_ref);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_supplier_code ON dim_supplier(tenant_id, supplier_code);

-- Facts metadata: unit normalization
ALTER TABLE fact_volunteering ADD COLUMN unit_source TEXT;
ALTER TABLE fact_volunteering ADD COLUMN unit_norm_method TEXT;

ALTER TABLE fact_energy ADD COLUMN unit_source TEXT;
ALTER TABLE fact_energy ADD COLUMN unit_norm_method TEXT;

ALTER TABLE fact_waste ADD COLUMN unit_source TEXT;
ALTER TABLE fact_waste ADD COLUMN unit_norm_method TEXT;

-- Currency normalization scaffolding
ALTER TABLE fact_donations ADD COLUMN amount_base REAL;
ALTER TABLE fact_donations ADD COLUMN fx_rate REAL;
ALTER TABLE fact_donations ADD COLUMN fx_source TEXT;
ALTER TABLE fact_donations ADD COLUMN fx_date DATE;

ALTER TABLE fact_procurement ADD COLUMN amount_base REAL;
ALTER TABLE fact_procurement ADD COLUMN fx_rate REAL;
ALTER TABLE fact_procurement ADD COLUMN fx_source TEXT;
ALTER TABLE fact_procurement ADD COLUMN fx_date DATE;


