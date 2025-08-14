## ImpactOS Medallion Architecture (Bronze → Silver → Gold)

This adds a deterministic, SQL-first data model with lineage from Gold back to Silver and Bronze.

- Bronze: one table per source worksheet, named `bronze_{file_slug}__{sheet_slug}` with required columns:
  - `bronze_id` (PK)
  - `source_file`, `sheet_name`, `row_idx`, `cell_ref_range`
  - raw columns as TEXT/REAL/INTEGER/DATE
- Silver: canonical facts and dimensions
  - Facts: `fact_volunteering`, `fact_donations`, `fact_procurement`, `fact_energy`, `fact_waste`
  - Dimensions: `dim_date`, `dim_site`, `dim_person`, `dim_supplier`
  - Lineage columns on each fact: `bronze_table`, `bronze_row_ids`, `transform_version`
- Gold: metric-friendly views or materializations; example: `gold_volunteer_hours_monthly`

Lineage
- Each fact row stores the Bronze table name and row identifiers used.
- You can trace any Gold aggregate → fact rows (Silver) → Bronze rows.

Migrations
- SQLite DDL: `db/migrations/20250813T000000Z__medallion_v1.sql`

Seed Example
```bash
export IMPACTOS_DB_PATH=db/impactos.db
python src/seed_medallion.py
```
This will:
- Create dimensions and facts
- Create a Bronze table and insert 3 sample rows from `data/seed_bronze_volunteering.csv`
- Transform into `fact_volunteering` with lineage
- Create the Gold view `gold_volunteer_hours_monthly`

Testing/Guarantees
- Facts enforce NOT NULL on keys/values where applicable
- Unit tests should assert:
  - Bronze → Silver lineage populated
  - `date_key` not null; `hours/amount/consumption_kwh/tonnes` not null
  - Gold aggregates reflect Silver fact sums

Notes
- All datasets are tenant-scoped via `tenant_id` (default: `default`).
- SQL is compiled deterministically; no free-text execution.

