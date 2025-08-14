import os
import sqlite3
from pathlib import Path
import pandas as pd


def setup_db(tmp_path):
    db_path = tmp_path / 'impactos.db'
    os.environ['IMPACTOS_DB_PATH'] = str(db_path)
    conn = sqlite3.connect(db_path)
    root = Path(__file__).resolve().parents[2]
    with open(root / 'db' / 'migrations' / '20250813T000000Z__medallion_v1.sql') as f:
        conn.executescript(f.read())
    with open(root / 'db' / 'migrations' / '20250813T040000Z__transform_runs.sql') as f:
        conn.executescript(f.read())
    with open(root / 'db' / 'migrations' / '20250813T041000Z__fact_meta_and_dim_uniques.sql') as f:
        conn.executescript(f.read())
    with open(root / 'db' / 'migrations' / '20250813T042000Z__transform_rejects_lineage.sql') as f:
        conn.executescript(f.read())
    with open(root / 'db' / 'migrations' / '20250813T010000Z__medallion_indexes_constraints.sql') as f:
        conn.executescript(f.read())
    return conn


def test_rejects_and_date_parsing(tmp_path):
    conn = setup_db(tmp_path)
    try:
        # Bronze with mixed date formats and blanks
        conn.execute("CREATE TABLE bronze_mixed__dates (bronze_id INTEGER PRIMARY KEY, tenant_id TEXT DEFAULT 'default', source_file TEXT, sheet_name TEXT, row_idx INTEGER, row_cell_ref TEXT, Value REAL, Unit TEXT, Date TEXT)")
        rows = [
            (1, 'default', 'x.xlsx', 'S1', 1, 'A1:C1', 10.0, 'minutes', '2025-01-01'),
            (2, 'default', 'x.xlsx', 'S1', 2, 'A2:C2', 20.0, 'minutes', '01/02/2025'),   # dd/mm/yyyy
            (3, 'default', 'x.xlsx', 'S1', 3, 'A3:C3', 30.0, 'minutes', 'February 3, 2025'),
            (4, 'default', 'x.xlsx', 'S1', 4, 'A4:C4', 40.0, 'minutes', ''),            # blank
        ]
        conn.executemany("INSERT INTO bronze_mixed__dates (bronze_id, tenant_id, source_file, sheet_name, row_idx, row_cell_ref, Value, Unit, Date) VALUES (?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()

        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from silver_transform import run_transform

        spec = {
            'fact': 'fact_volunteering',
            'mappings': {
                'value': 'Value',
                'unit': 'Unit',
                'date': 'Date',
            }
        }
        res = run_transform(conn, 'bronze_mixed__dates', spec)
        # Expect one reject (blank date)
        assert res['rows_in'] == 4
        assert res['rows_out'] == 3
        assert res['rows_rejected'] >= 1
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM transform_rejects WHERE bronze_table='bronze_mixed__dates'")
        assert cur.fetchone()[0] >= 1
    finally:
        conn.close()


def test_idempotency_and_spec_change(tmp_path):
    conn = setup_db(tmp_path)
    try:
        conn.execute("CREATE TABLE bronze_idem__ex (bronze_id INTEGER PRIMARY KEY, tenant_id TEXT DEFAULT 'default', source_file TEXT, sheet_name TEXT, row_idx INTEGER, row_cell_ref TEXT, Hours REAL, Date TEXT)")
        conn.executemany("INSERT INTO bronze_idem__ex (bronze_id, tenant_id, source_file, sheet_name, row_idx, row_cell_ref, Hours, Date) VALUES (?,?,?,?,?,?,?,?)",
                         [(1,'default','x','S',1,'A1:C1', 1.0, '2025-01-01')])
        conn.commit()
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from silver_transform import run_transform

        spec1 = {'fact': 'fact_volunteering', 'mappings': {'value': 'Hours', 'unit': 'hours', 'date': 'Date'}}
        r1 = run_transform(conn, 'bronze_idem__ex', spec1)
        r2 = run_transform(conn, 'bronze_idem__ex', spec1)
        # idempotent inserts because of unique lineage + transform_version(spec_hash)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM fact_volunteering WHERE bronze_table='bronze_idem__ex'")
        assert cur.fetchone()[0] == 1

        # spec change (unit literal different -> different hash) still same value but new transform_version
        spec2 = {'fact': 'fact_volunteering', 'mappings': {'value': 'Hours', 'unit': 'Minutes', 'date': 'Date'}}
        r3 = run_transform(conn, 'bronze_idem__ex', spec2)
        cur.execute("SELECT COUNT(DISTINCT transform_version) FROM fact_volunteering WHERE bronze_table='bronze_idem__ex'")
        assert cur.fetchone()[0] >= 2
    finally:
        conn.close()


def test_energy_procurement_transforms(tmp_path):
    conn = setup_db(tmp_path)
    try:
        # Energy bronze (MWh -> kWh)
        conn.execute("CREATE TABLE bronze_energy__ex (bronze_id INTEGER PRIMARY KEY, tenant_id TEXT DEFAULT 'default', source_file TEXT, sheet_name TEXT, row_idx INTEGER, row_cell_ref TEXT, Value REAL, Unit TEXT, Date TEXT, Site TEXT)")
        conn.executemany("INSERT INTO bronze_energy__ex (bronze_id, tenant_id, source_file, sheet_name, row_idx, row_cell_ref, Value, Unit, Date, Site) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         [(1,'default','e','S',1,'A1:C1', 1.0, 'MWh', '2025-01-01', 'HQ')])
        # Procurement bronze
        conn.execute("CREATE TABLE bronze_proc__ex (bronze_id INTEGER PRIMARY KEY, tenant_id TEXT DEFAULT 'default', source_file TEXT, sheet_name TEXT, row_idx INTEGER, row_cell_ref TEXT, Amount REAL, Currency TEXT, Date TEXT, Supplier TEXT)")
        conn.executemany("INSERT INTO bronze_proc__ex (bronze_id, tenant_id, source_file, sheet_name, row_idx, row_cell_ref, Amount, Currency, Date, Supplier) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         [(1,'default','p','S',1,'A1:C1', 100.0, 'USD', '2025-02-01', 'ACME')])
        conn.commit()

        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from silver_transform import run_transform

        energy_spec = {'fact': 'fact_energy', 'mappings': {'value': 'Value', 'unit': 'Unit', 'date': 'Date', 'site_code': 'Site', 'scope': 'scope1'}}
        proc_spec = {'fact': 'fact_procurement', 'mappings': {'value': 'Amount', 'currency': 'Currency', 'date': 'Date', 'supplier_code': 'Supplier'}}

        run_transform(conn, 'bronze_energy__ex', energy_spec)
        run_transform(conn, 'bronze_proc__ex', proc_spec)

        cur = conn.cursor()
        cur.execute("SELECT consumption_kwh, unit_source FROM fact_energy WHERE bronze_table='bronze_energy__ex'")
        kwh, unit_source = cur.fetchone()
        assert abs(kwh - 1000.0) < 1e-6
        assert unit_source.lower() in ('mwh', 'mwh')

        cur.execute("SELECT amount_base, currency FROM fact_procurement WHERE bronze_table='bronze_proc__ex'")
        ab, curr = cur.fetchone()
        assert abs(ab - 100.0) < 1e-6
        assert curr == 'GBP'
    finally:
        conn.close()


