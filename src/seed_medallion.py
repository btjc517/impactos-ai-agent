"""
Seed script demonstrating Bronze → Silver → Gold using a tiny CSV.

Usage:
  IMPACTOS_DB_PATH=db/impactos.db python src/seed_medallion.py
"""

import os
import csv
import sqlite3
from pathlib import Path

from transforms_medallion import ensure_bronze_table, transform_bronze_to_fact_volunteering, create_gold_sample_view


def main():
    db_path = os.getenv('IMPACTOS_DB_PATH', 'db/impactos.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        # Run medallion DDL if present
        migrations_dir = Path('db/migrations')
        ddl = migrations_dir / '20250813T000000Z__medallion_v1.sql'
        if ddl.exists():
            with open(ddl, 'r') as f:
                conn.executescript(f.read())

        # Create a Bronze table and load rows from CSV (idempotent)
        csv_path = Path('data/seed_bronze_volunteering.csv')
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows.extend(reader)

        table = ensure_bronze_table(
            conn,
            file_slug='seed_volunteering',
            sheet_slug='sheet1',
            raw_columns={'person_ref': 'TEXT', 'site_code': 'TEXT', 'hours': 'REAL'},
        )

        # Ensure clean slate for this seed: delete prior facts and bronze rows
        try:
            conn.execute("DELETE FROM fact_volunteering WHERE tenant_id='default' AND bronze_table like 'bronze_seed_volunteering__sheet1%'")
        except Exception:
            pass
        try:
            conn.execute(f"DELETE FROM {table} WHERE tenant_id='default'")
        except Exception:
            pass

        # Insert Bronze rows (tenant set to 'default')
        insert_sql = f"""
            INSERT INTO {table} (tenant_id, source_file, sheet_name, row_idx, cell_ref_range, person_ref, site_code, hours)
            VALUES ('default', ?, ?, ?, ?, ?, ?, ?)
        """
        conn.executemany(
            insert_sql,
            [
                (
                    r['source_file'],
                    r['sheet_name'],
                    int(r['row_idx']),
                    r['cell_ref_range'],
                    r['person_ref'],
                    r['site_code'],
                    float(r['hours']) if r['hours'] else None,
                )
                for r in rows
            ],
        )
        conn.commit()

        # Transform to Silver fact
        inserted = transform_bronze_to_fact_volunteering(
            conn,
            bronze_table=table,
            mappings={'hours': 'hours', 'person_ref': 'person_ref', 'site_code': 'site_code'},
            default_date='2025-01-15',
        )

        # Create Gold sample view
        create_gold_sample_view(conn)

        # Show quick checks
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM fact_volunteering')
        facts = cur.fetchone()[0]
        cur.execute('SELECT * FROM gold_volunteer_hours_monthly ORDER BY yyyymm')
        gold = cur.fetchall()
        print(f"Inserted {inserted} fact_volunteering rows (total {facts}). Gold sample: {gold}")
    finally:
        conn.close()


if __name__ == '__main__':
    main()


