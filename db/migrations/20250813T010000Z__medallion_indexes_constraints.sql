-- Indexes and constraints for Medallion hardening (SQLite)
PRAGMA foreign_keys = ON;

-- Pre-clean: remove duplicate lineage rows (keep lowest fact_id)
DELETE FROM fact_volunteering
WHERE fact_id NOT IN (
  SELECT MIN(fact_id)
  FROM fact_volunteering
  GROUP BY tenant_id, bronze_table, bronze_row_ids, transform_version
);
DELETE FROM fact_donations
WHERE fact_id NOT IN (
  SELECT MIN(fact_id)
  FROM fact_donations
  GROUP BY tenant_id, bronze_table, bronze_row_ids, transform_version
);
DELETE FROM fact_procurement
WHERE fact_id NOT IN (
  SELECT MIN(fact_id)
  FROM fact_procurement
  GROUP BY tenant_id, bronze_table, bronze_row_ids, transform_version
);
DELETE FROM fact_energy
WHERE fact_id NOT IN (
  SELECT MIN(fact_id)
  FROM fact_energy
  GROUP BY tenant_id, bronze_table, bronze_row_ids, transform_version
);
DELETE FROM fact_waste
WHERE fact_id NOT IN (
  SELECT MIN(fact_id)
  FROM fact_waste
  GROUP BY tenant_id, bronze_table, bronze_row_ids, transform_version
);

-- Facts: Unique lineage to guard against duplicate transforms
CREATE UNIQUE INDEX IF NOT EXISTS uq_fact_volunteering_lineage
ON fact_volunteering(tenant_id, bronze_table, bronze_row_ids, transform_version);

CREATE UNIQUE INDEX IF NOT EXISTS uq_fact_donations_lineage
ON fact_donations(tenant_id, bronze_table, bronze_row_ids, transform_version);

CREATE UNIQUE INDEX IF NOT EXISTS uq_fact_procurement_lineage
ON fact_procurement(tenant_id, bronze_table, bronze_row_ids, transform_version);

CREATE UNIQUE INDEX IF NOT EXISTS uq_fact_energy_lineage
ON fact_energy(tenant_id, bronze_table, bronze_row_ids, transform_version);

CREATE UNIQUE INDEX IF NOT EXISTS uq_fact_waste_lineage
ON fact_waste(tenant_id, bronze_table, bronze_row_ids, transform_version);

-- Facts: Join and tenant indexes
CREATE INDEX IF NOT EXISTS idx_fact_volunteering_person ON fact_volunteering(person_id);
CREATE INDEX IF NOT EXISTS idx_fact_volunteering_site ON fact_volunteering(site_id);
CREATE INDEX IF NOT EXISTS idx_fact_volunteering_tenant ON fact_volunteering(tenant_id);

CREATE INDEX IF NOT EXISTS idx_fact_donations_site ON fact_donations(site_id);
CREATE INDEX IF NOT EXISTS idx_fact_donations_tenant ON fact_donations(tenant_id);

CREATE INDEX IF NOT EXISTS idx_fact_procurement_supplier ON fact_procurement(supplier_id);
CREATE INDEX IF NOT EXISTS idx_fact_procurement_tenant ON fact_procurement(tenant_id);

CREATE INDEX IF NOT EXISTS idx_fact_energy_site ON fact_energy(site_id);
CREATE INDEX IF NOT EXISTS idx_fact_energy_tenant ON fact_energy(tenant_id);

CREATE INDEX IF NOT EXISTS idx_fact_waste_site ON fact_waste(site_id);
CREATE INDEX IF NOT EXISTS idx_fact_waste_tenant ON fact_waste(tenant_id);

-- Dimensions: Tenant indexes (no uniqueness enforced to avoid migration conflicts)
CREATE INDEX IF NOT EXISTS idx_dim_site_tenant ON dim_site(tenant_id);
CREATE INDEX IF NOT EXISTS idx_dim_person_tenant ON dim_person(tenant_id);
CREATE INDEX IF NOT EXISTS idx_dim_supplier_tenant ON dim_supplier(tenant_id);

-- Registry tenant index
CREATE INDEX IF NOT EXISTS idx_bronze_registry_tenant ON bronze_registry(tenant_id);


