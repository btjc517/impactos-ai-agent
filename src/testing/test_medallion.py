import os
import sqlite3
import sys
from pathlib import Path


def setup_db(tmp_path):
    db_path = tmp_path / 'impactos.db'
    os.environ['IMPACTOS_DB_PATH'] = str(db_path)
    conn = sqlite3.connect(db_path)
    # Apply medallion DDL
    ddl_path = Path(__file__).resolve().parents[2] / 'db' / 'migrations' / '20250813T000000Z__medallion_v1.sql'
    with open(ddl_path, 'r') as f:
        conn.executescript(f.read())
    return conn


def test_lineage_and_not_null(tmp_path):
    # Ensure src is importable
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    conn = setup_db(tmp_path)
    try:
        # Seed via Python to avoid import cycles
        from transforms_medallion import ensure_bronze_table, transform_bronze_to_fact_volunteering, create_gold_sample_view

        table = ensure_bronze_table(
            conn,
            file_slug='seed_volunteering',
            sheet_slug='sheet1',
            raw_columns={'person_ref': 'TEXT', 'site_code': 'TEXT', 'hours': 'REAL'},
        )
        conn.executemany(
            f"INSERT INTO {table} (source_file, sheet_name, row_idx, cell_ref_range, person_ref, site_code, hours) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ('seed.xlsx', 'Sheet1', 1, 'A2:C2', 'u1', 'London', 2.0),
                ('seed.xlsx', 'Sheet1', 2, 'A3:C3', 'u2', 'London', 3.0),
            ],
        )
        conn.commit()

        inserted = transform_bronze_to_fact_volunteering(
            conn,
            bronze_table=table,
            mappings={'hours': 'hours', 'person_ref': 'person_ref', 'site_code': 'site_code'},
            default_date='2025-01-01',
        )
        assert inserted == 2

        # Check lineage and not-null
        cur = conn.cursor()
        cur.execute("SELECT bronze_table, bronze_row_ids, date_key, hours FROM fact_volunteering")
        rows = cur.fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r[0] == table
            assert r[1] is not None and len(str(r[1])) > 0
            assert r[2] is not None
            assert r[3] is not None

        # Gold view correctness
        create_gold_sample_view(conn)
        cur.execute("SELECT yyyymm, total_hours FROM gold_volunteer_hours_monthly")
        yyyymm, total = cur.fetchone()
        assert yyyymm == '202501'
        assert abs(total - 5.0) < 1e-6
    finally:
        conn.close()


