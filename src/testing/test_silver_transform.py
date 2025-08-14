import os
import sqlite3
from pathlib import Path
import pandas as pd


def test_transform_volunteering_lineage_and_units(tmp_path):
    db_path = tmp_path / 'impactos.db'
    os.environ['IMPACTOS_DB_PATH'] = str(db_path)
    conn = sqlite3.connect(db_path)
    try:
        # Apply core DDL
        root = Path(__file__).resolve().parents[2]
        with open(root / 'db' / 'migrations' / '20250813T000000Z__medallion_v1.sql') as f:
            conn.executescript(f.read())
        # Create transform_runs table and fact meta columns
        with open(root / 'db' / 'migrations' / '20250813T040000Z__transform_runs.sql') as f:
            conn.executescript(f.read())
        with open(root / 'db' / 'migrations' / '20250813T041000Z__fact_meta_and_dim_uniques.sql') as f:
            conn.executescript(f.read())
        # Create a simple bronze table
        conn.execute("CREATE TABLE bronze_test__vol (bronze_id INTEGER PRIMARY KEY, tenant_id TEXT DEFAULT 'default', source_file TEXT, sheet_name TEXT, row_idx INTEGER, row_cell_ref TEXT, Person TEXT, Site TEXT, Hours REAL, Date TEXT)")
        # Insert data (including a minutes example)
        rows = [
            (1, 'default', 'x.xlsx', 'S1', 1, 'A1:C1', 'u1', 'HQ', 2.0, '2025-01-01'),
            (2, 'default', 'x.xlsx', 'S1', 2, 'A2:C2', 'u2', 'HQ', 60.0, '2025-01-02'),
        ]
        conn.executemany("INSERT INTO bronze_test__vol (bronze_id, tenant_id, source_file, sheet_name, row_idx, row_cell_ref, Person, Site, Hours, Date) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()

        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from silver_transform import run_transform

        spec = {
            'fact': 'fact_volunteering',
            'mappings': {
                'value': 'Hours',
                'unit': 'hours',
                'date': 'Date',
                'person_ref': 'Person',
                'site_code': 'Site',
            }
        }
        res = run_transform(conn, 'bronze_test__vol', spec)
        assert res['rows_out'] == 2
        # Lineage present and unit normalized remains hours
        cur = conn.cursor()
        cur.execute("SELECT hours, unit, bronze_table, bronze_row_ids FROM fact_volunteering ORDER BY fact_id")
        vals = cur.fetchall()
        assert len(vals) == 2
        assert vals[0][1] == 'hours' and vals[1][1] == 'hours'
        assert vals[0][2] == 'bronze_test__vol'
        assert vals[0][3] == '1' and vals[1][3] == '2'
    finally:
        conn.close()


