"""
Silver transform engine: map Bronze tables to canonical facts/dimensions with lineage.
Config-driven (Python dict now; YAML-compatible later).
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import sqlite3
import pandas as pd

from transforms_medallion import upsert_dim_date


def _hash_spec(spec: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode('utf-8')).hexdigest()


def _normalize_unit(value: Any, unit: Optional[str]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    if value is None or value == "":
        return None, unit, None
    try:
        v = float(value)
    except Exception:
        return None, unit, None
    # simple example: minutes→hours
    if not unit:
        return v, unit, None
    u = unit.lower()
    # Resolve units via concept graph where possible; fallback to known conversions
    try:
        from semantic_resolver import SemanticResolver
        ur = SemanticResolver().resolve('unit', u)
        if ur.get('outcome') == 'accepted':
            u = ur.get('key') or u
    except Exception:
        pass
    if u in ("minute", "minutes", "mins"):
        return v / 60.0, "hours", "minutes_to_hours"
    if u in ("kwh",):
        return v, "kwh", None
    if u in ("mwh",):
        return v * 1000.0, "kwh", "mwh_to_kwh"
    if u in ("t", "tonne", "tonnes"):
        return v * 1000.0, "kg", "tonne_to_kg"
    if u in ("kg",):
        return v, "kg", None
    return v, unit, None


def _normalize_currency(value: Any, currency: Optional[str], base_currency: str = "GBP") -> Tuple[Optional[float], str, float, str, Optional[str]]:
    if value is None or value == "":
        return None, base_currency, 1.0, "static", None
    try:
        v = float(value)
    except Exception:
        return None, base_currency, 1.0, "static", None
    # Placeholder FX: assume 1.0 and mark as static
    return v, base_currency, 1.0, "static", None


def run_transform(
    conn: sqlite3.Connection,
    bronze_table: str,
    spec: Dict[str, Any],
    tenant_id: str = 'default',
) -> Dict[str, Any]:
    """Execute a transform from Bronze to Silver facts per spec.

    Spec structure (example for volunteering):
    {
      "fact": "fact_volunteering",
      "mappings": {
        "value": "Hours",            # column name in bronze
        "unit": "hours",            # literal or column name
        "date": "Date",             # column name if present
        "site_code": "Site",        # optional
        "person_ref": "Person",     # optional
      },
      "base_currency": "GBP"
    }
    """
    cur = conn.cursor()
    # Ensure transform_runs entry
    spec_hash = _hash_spec(spec)
    cur.execute(
        """
        INSERT INTO transform_runs (tenant_id, bronze_table, transform_name, transform_version, spec_hash, status, notes)
        VALUES (?, ?, ?, ?, ?, 'running', ?)
        """,
        (tenant_id, bronze_table, spec.get('fact') or 'unknown', spec_hash, spec_hash, json.dumps(spec, sort_keys=True)),
    )
    run_id = cur.lastrowid
    conn.commit()

    results = {"run_id": run_id, "rows_in": 0, "rows_out": 0, "rows_rejected": 0}
    try:
        df = pd.read_sql_query(f"SELECT * FROM {bronze_table}", conn)
        results["rows_in"] = int(len(df))
        if df.empty:
            cur.execute("UPDATE transform_runs SET status='completed', rows_in=?, rows_out=?, rows_rejected=?, completed_at=CURRENT_TIMESTAMP WHERE id=?", (0,0,0,run_id))
            conn.commit()
            return results

        mappings = spec.get('mappings', {})
        fact = spec.get('fact')
        base_currency = spec.get('base_currency', 'GBP')

        inserts: List[Tuple] = []
        rejects = 0

        cols = list(df.columns)
        cols_lc = {c.lower(): c for c in cols}

        def _get(row, name: Optional[str]):
            if not name:
                return None
            if name in row:
                return row.get(name)
            c = cols_lc.get(str(name).lower())
            return row.get(c) if c else None

        for _, row in df.iterrows():
            try:
                # Value + unit normalization
                raw_val = _get(row, mappings.get('value'))
                unit_src = mappings.get('unit')
                unit = unit_src
                if isinstance(unit_src, str):
                    # First treat as a column name; otherwise treat as literal
                    if unit_src in df.columns or str(unit_src).lower() in cols_lc:
                        unit = _get(row, unit_src)
                    else:
                        unit = unit_src
                val_norm, unit_final, unit_method = _normalize_unit(raw_val, unit)
                if val_norm is None:
                    rejects += 1
                    # store reject
                    cur.execute(
                        "INSERT INTO transform_rejects (transform_run_id, tenant_id, bronze_table, bronze_row_id, reason, raw_snapshot) VALUES (?,?,?,?,?,?)",
                        (run_id, tenant_id, bronze_table, int(row.get('bronze_id') or 0), 'invalid_or_missing_value', json.dumps(row.to_dict(), default=str)),
                    )
                    continue

                # Date
                date_str = None
                date_col = mappings.get('date')
                date_val = _get(row, date_col)
                if date_col and pd.notna(date_val):
                    try:
                        d = pd.to_datetime(date_val, dayfirst=('/' in str(date_val) and '-' not in str(date_val)))
                        date_str = d.strftime('%Y-%m-%d')
                    except Exception:
                        date_str = None
                if date_str is None:
                    rejects += 1
                    cur.execute(
                        "INSERT INTO transform_rejects (transform_run_id, tenant_id, bronze_table, bronze_row_id, reason, raw_snapshot) VALUES (?,?,?,?,?,?)",
                        (run_id, tenant_id, bronze_table, int(row.get('bronze_id') or 0), 'invalid_or_missing_date', json.dumps(row.to_dict(), default=str)),
                    )
                    continue
                date_key = upsert_dim_date(conn, date_str)

                # Optional dims
                site_code = mappings.get('site_code')
                person_ref = mappings.get('person_ref')
                site_id = None
                person_id = None
                if site_code and pd.notna(_get(row, site_code)):
                    # dim_site upsert minimal
                    v = _get(row, site_code)
                    cur.execute("INSERT OR IGNORE INTO dim_site (tenant_id, site_code, site_name) VALUES (?,?,?)", (tenant_id, v, v))
                    cur.execute("SELECT site_id FROM dim_site WHERE tenant_id=? AND site_code=?", (tenant_id, v))
                    r = cur.fetchone(); site_id = r[0] if r else None
                if person_ref and pd.notna(_get(row, person_ref)):
                    v = _get(row, person_ref)
                    cur.execute("INSERT OR IGNORE INTO dim_person (tenant_id, external_ref) VALUES (?,?)", (tenant_id, v))
                    cur.execute("SELECT person_id FROM dim_person WHERE tenant_id=? AND external_ref=?", (tenant_id, v))
                    r = cur.fetchone(); person_id = r[0] if r else None

                # Lineage
                bronze_row_id = row.get('bronze_id')
                lineage_ids = str(int(bronze_row_id)) if pd.notna(bronze_row_id) else None

                if fact == 'fact_volunteering':
                    inserts.append((tenant_id, date_key, person_id, site_id, float(val_norm), unit_final or 'hours', bronze_table, lineage_ids, spec_hash, unit, unit_method))
                elif fact == 'fact_donations':
                    amount, currency_b, fx_rate, fx_source, fx_date = _normalize_currency(raw_val, _get(row, mappings.get('currency')), base_currency)
                    inserts.append((tenant_id, date_key, site_id, float(amount or 0.0), currency_b or base_currency, bronze_table, lineage_ids, spec_hash, float(amount or 0.0), fx_rate, fx_source, fx_date))
                elif fact == 'fact_procurement':
                    amount, currency_b, fx_rate, fx_source, fx_date = _normalize_currency(raw_val, row.get(mappings.get('currency')), base_currency)
                    sup_code = mappings.get('supplier_code')
                    supplier_id = None
                    if sup_code and sup_code in df.columns and pd.notna(row.get(sup_code)):
                        cur.execute("INSERT OR IGNORE INTO dim_supplier (tenant_id, supplier_code, supplier_name) VALUES (?,?,?)", (tenant_id, row.get(sup_code), row.get(sup_code)))
                        cur.execute("SELECT supplier_id FROM dim_supplier WHERE tenant_id=? AND supplier_code=?", (tenant_id, row.get(sup_code)))
                        r = cur.fetchone(); supplier_id = r[0] if r else None
                    inserts.append((tenant_id, date_key, supplier_id, float(amount or 0.0), currency_b or base_currency, bronze_table, lineage_ids, spec_hash, float(amount or 0.0), fx_rate, fx_source, fx_date))
                elif fact == 'fact_energy':
                    v = float(val_norm)
                    scope = row.get(mappings.get('scope')) if mappings.get('scope') in df.columns else mappings.get('scope')
                    site_code2 = mappings.get('site_code')
                    site_id2 = None
                    if site_code2 and site_code2 in df.columns and pd.notna(row.get(site_code2)):
                        cur.execute("INSERT OR IGNORE INTO dim_site (tenant_id, site_code, site_name) VALUES (?,?,?)", (tenant_id, row.get(site_code2), row.get(site_code2)))
                        cur.execute("SELECT site_id FROM dim_site WHERE tenant_id=? AND site_code=?", (tenant_id, row.get(site_code2)))
                        r = cur.fetchone(); site_id2 = r[0] if r else None
                    inserts.append((tenant_id, date_key, site_id2, v, scope, bronze_table, lineage_ids, spec_hash, unit, unit_method))
                elif fact == 'fact_waste':
                    v = float(val_norm)
                    stream = row.get(mappings.get('stream')) if mappings.get('stream') in df.columns else mappings.get('stream')
                    site_code3 = mappings.get('site_code')
                    site_id3 = None
                    if site_code3 and site_code3 in df.columns and pd.notna(row.get(site_code3)):
                        cur.execute("INSERT OR IGNORE INTO dim_site (tenant_id, site_code, site_name) VALUES (?,?,?)", (tenant_id, row.get(site_code3), row.get(site_code3)))
                        cur.execute("SELECT site_id FROM dim_site WHERE tenant_id=? AND site_code=?", (tenant_id, row.get(site_code3)))
                        r = cur.fetchone(); site_id3 = r[0] if r else None
                    inserts.append((tenant_id, date_key, site_id3, v, stream, bronze_table, lineage_ids, spec_hash, unit, unit_method))
                else:
                    rejects += 1
                    cur.execute(
                        "INSERT INTO transform_rejects (transform_run_id, tenant_id, bronze_table, bronze_row_id, reason, raw_snapshot) VALUES (?,?,?,?,?,?)",
                        (run_id, tenant_id, bronze_table, int(row.get('bronze_id') or 0), 'unsupported_fact', json.dumps(row.to_dict(), default=str)),
                    )
            except Exception as e:
                rejects += 1
                try:
                    cur.execute(
                        "INSERT INTO transform_rejects (transform_run_id, tenant_id, bronze_table, bronze_row_id, reason, raw_snapshot) VALUES (?,?,?,?,?,?)",
                        (run_id, tenant_id, bronze_table, int(row.get('bronze_id') or 0), f'exception:{str(e)[:100]}', json.dumps(row.to_dict(), default=str)),
                    )
                except Exception:
                    pass

        if inserts:
            insert_sql_by_fact = {
                'fact_volunteering': (
                    """
                    INSERT OR IGNORE INTO fact_volunteering
                    (tenant_id, date_key, person_id, site_id, hours, unit, bronze_table, bronze_row_ids, transform_version, unit_source, unit_norm_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                ),
                'fact_donations': (
                    """
                    INSERT OR IGNORE INTO fact_donations
                    (tenant_id, date_key, site_id, amount, currency, bronze_table, bronze_row_ids, transform_version, amount_base, fx_rate, fx_source, fx_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                ),
                'fact_procurement': (
                    """
                    INSERT OR IGNORE INTO fact_procurement
                    (tenant_id, date_key, supplier_id, amount, currency, bronze_table, bronze_row_ids, transform_version, amount_base, fx_rate, fx_source, fx_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                ),
                'fact_energy': (
                    """
                    INSERT OR IGNORE INTO fact_energy
                    (tenant_id, date_key, site_id, consumption_kwh, scope, bronze_table, bronze_row_ids, transform_version, unit_source, unit_norm_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                ),
                'fact_waste': (
                    """
                    INSERT OR IGNORE INTO fact_waste
                    (tenant_id, date_key, site_id, tonnes, stream, bronze_table, bronze_row_ids, transform_version, unit_source, unit_norm_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                ),
            }
            sql = insert_sql_by_fact.get(fact)
            if not sql:
                raise ValueError(f"Unsupported fact '{fact}' in spec")
            cur.executemany(sql, inserts)
            conn.commit()

            # Update fact_lineage bridge for cheap joins
            lineage_query_by_fact = {
                'fact_volunteering': ("fact_volunteering", "SELECT fact_id, bronze_row_ids FROM fact_volunteering WHERE bronze_table=? AND transform_version=?"),
                'fact_donations': ("fact_donations", "SELECT fact_id, bronze_row_ids FROM fact_donations WHERE bronze_table=? AND transform_version=?"),
                'fact_procurement': ("fact_procurement", "SELECT fact_id, bronze_row_ids FROM fact_procurement WHERE bronze_table=? AND transform_version=?"),
                'fact_energy': ("fact_energy", "SELECT fact_id, bronze_row_ids FROM fact_energy WHERE bronze_table=? AND transform_version=?"),
                'fact_waste': ("fact_waste", "SELECT fact_id, bronze_row_ids FROM fact_waste WHERE bronze_table=? AND transform_version=?"),
            }
            fact_table, q = lineage_query_by_fact.get(fact, (None, None))
            if not fact_table:
                raise ValueError(f"Unsupported fact '{fact}' for lineage query")
            cur.execute(q, (bronze_table, spec_hash))
            for fid, rids in cur.fetchall():
                try:
                    rid = int(rids)
                    cur.execute(
                        "INSERT INTO fact_lineage (tenant_id, fact_table, fact_id, bronze_table, bronze_row_id) VALUES (?,?,?,?,?)",
                        (tenant_id, fact_table, fid, bronze_table, rid),
                    )
                except Exception:
                    pass
            conn.commit()

        results["rows_out"] = len(inserts)
        results["rows_rejected"] = rejects
        cur.execute(
            "UPDATE transform_runs SET status='completed', rows_in=?, rows_out=?, rows_rejected=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            (results["rows_in"], results["rows_out"], results["rows_rejected"], run_id),
        )
        conn.commit()
        return results
    except Exception as e:
        cur.execute("UPDATE transform_runs SET status='failed', notes=? WHERE id=?", (str(e), run_id))
        conn.commit()
        raise


# Example transform specs (Python dicts)
EXAMPLE_SPECS: Dict[str, Dict[str, Any]] = {
    # bronze table name → spec
    'bronze_seed_volunteering__sheet1': {
        'fact': 'fact_volunteering',
        'mappings': {
            'value': 'hours',
            'unit': 'hours',
            'date': None,  # not present in seed; will be rejected by engine
            'person_ref': 'person_ref',
            'site_code': 'site_code',
        },
        'base_currency': 'GBP'
    }
}


