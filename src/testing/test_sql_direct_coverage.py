import os
import time
import sqlite3
import sys
from pathlib import Path

# Ensure src is on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from schema import DatabaseSchema
from query import QuerySystem


class CountingQuerySystem(QuerySystem):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        # Force a truthy client so pipeline attempts to call GPT
        self.openai_client = True  # sentinel
        self.gpt_calls = 0

    def _generate_gpt_answer(self, question, results):
        # Count and return a deterministic mock answer
        self.gpt_calls += 1
        return "[MOCK GPT ANSWER]"


def setup_db(db_path: str):
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    schema = DatabaseSchema(db_path)
    schema.initialize_database()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        # One source per category
        sources = [
            ("TakingCare_Benevity_Synthetic_Data.xlsx", "excel"),
            ("TakingCare_Carbon_Reporting_Synthetic_Data.xlsx", "excel"),
            ("TakingCare_HCM_Synthetic_Data.xlsx", "excel"),
        ]
        source_ids = []
        for fn, ft in sources:
            cur.execute(
                "INSERT INTO sources (filename, file_type, file_size_bytes, processing_status) VALUES (?, ?, ?, ?)",
                (fn, ft, 12345, "processed"),
            )
            source_ids.append(cur.lastrowid)

        benevity_id, carbon_id, hcm_id = source_ids

        # Aggregation-friendly volunteering hours (consistent)
        cur.executemany(
            "INSERT INTO impact_metrics (source_id, metric_name, metric_value, metric_unit, metric_category) VALUES (?, ?, ?, ?, ?)",
            [
                (benevity_id, "volunteering_hours", 100.0, "hours", "volunteering"),
                (benevity_id, "volunteering_hours", 50.0, "hours", "volunteering"),
                (benevity_id, "volunteering_hours", 37.5, "hours", "volunteering"),
            ],
        )

        # Donations in GBP (consistent)
        cur.executemany(
            "INSERT INTO impact_metrics (source_id, metric_name, metric_value, metric_unit, metric_category) VALUES (?, ?, ?, ?, ?)",
            [
                (benevity_id, "donations", 1000.0, "£", "charitable_giving"),
                (benevity_id, "donations", 250.0, "£", "charitable_giving"),
            ],
        )

        # Carbon emissions in kg CO2e (consistent)
        cur.executemany(
            "INSERT INTO impact_metrics (source_id, metric_name, metric_value, metric_unit, metric_category) VALUES (?, ?, ?, ?, ?)",
            [
                (carbon_id, "emissions", 1200.0, "kg CO2e", "carbon"),
                (carbon_id, "emissions", 800.0, "kg CO2e", "carbon"),
            ],
        )

        # Mixed units case: same metric_name but different units (should bypass SQL-direct)
        cur.executemany(
            "INSERT INTO impact_metrics (source_id, metric_name, metric_value, metric_unit, metric_category) VALUES (?, ?, ?, ?, ?)",
            [
                (hcm_id, "mixed_hours", 60.0, "minutes", "volunteering"),
                (hcm_id, "mixed_hours", 1.0, "hours", "volunteering"),
            ],
        )

        conn.commit()


def run():
    db_path = "db/test_sql_coverage.db"
    setup_db(db_path)

    # Run with counting QS
    qs = CountingQuerySystem(db_path)

    scenarios = [
        ("What is the total amount of volunteering hours?", True),
        ("What is the total donations in £?", True),
        ("What are the total emissions?", True),
        ("What is the total mixed hours?", False),
        ("What sustainability initiatives exist?", False),
    ]

    results = []
    for question, expect_sql in scenarios:
        start = time.time()
        answer = qs.query(question)
        dur = time.time() - start
        used_sql = answer.startswith("Total ")
        results.append({
            'question': question,
            'expected_sql': expect_sql,
            'used_sql_direct': used_sql,
            'gpt_calls_so_far': qs.gpt_calls,
            'latency_ms': int(dur * 1000),
        })

    # Repeat a GPT-needed question to validate caching impact (pick descriptive)
    question_repeat = "What sustainability initiatives exist?"
    t1 = time.time(); _ = qs.query(question_repeat); t1 = (time.time() - t1) * 1000
    t2 = time.time(); _ = qs.query(question_repeat); t2 = (time.time() - t2) * 1000

    print("\n=== SQL-direct vs GPT usage report ===")
    for r in results:
        print(f"- Q: {r['question']}")
        print(f"  used_sql_direct={r['used_sql_direct']} expected_sql={r['expected_sql']} gpt_calls_so_far={r['gpt_calls_so_far']} latency_ms={r['latency_ms']}")
    print(f"\nCaching check (same descriptive query): first_ms={int(t1)} second_ms={int(t2)}")


if __name__ == "__main__":
    run() 

# Medallion sanity: ensure gold view exists after migration when executed in isolation
def test_gold_view_presence_if_migrated(tmp_path):
    db_path = tmp_path / 'impactos.db'
    os.environ['IMPACTOS_DB_PATH'] = str(db_path)
    conn = sqlite3.connect(db_path)
    try:
        ddl = Path(__file__).resolve().parents[2] / 'db' / 'migrations' / '20250813T000000Z__medallion_v1.sql'
        with open(ddl, 'r') as f:
            conn.executescript(f.read())
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='gold_volunteer_hours_monthly'")
        assert cur.fetchone() is not None
    finally:
        conn.close()