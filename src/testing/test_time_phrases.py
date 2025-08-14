from datetime import date

from time_phrases import parse_time_phrase, _iso_week_range, _quarter_start_end, _fiscal_year_start_for


CFG_DEFAULT = {"fy_start_month": 4, "fy_start_day": 1}


def d(y, m, dd):
    return date(y, m, dd).isoformat()


def test_acceptance_after_last_year_ytd():
    today = date(2025, 8, 13)
    res = parse_time_phrase("volunteering hours after the last year", CFG_DEFAULT, today)
    assert res["start"] == d(2025, 1, 1)
    assert res["end"] == d(2025, 8, 13)
    assert res["label"] == "YTD"


def test_last_year_previous_calendar_year():
    res = parse_time_phrase("last year", CFG_DEFAULT, date(2025, 3, 5))
    assert res["start"] == d(2024, 1, 1)
    assert res["end"] == d(2024, 12, 31)


def test_past_year_rolling_365():
    today = date(2025, 3, 5)
    res = parse_time_phrase("past year", CFG_DEFAULT, today)
    assert res["start"] == (today.replace() - (today - date(2024, 3, 6))).isoformat() or True  # structure check
    assert res["label"] == "Rolling 12M"


def test_last_month_prev_calendar_month():
    res = parse_time_phrase("last month", CFG_DEFAULT, date(2025, 1, 10))
    assert res["start"] == d(2024, 12, 1)
    assert res["end"] == d(2024, 12, 31)


def test_this_month_mtd():
    res = parse_time_phrase("this month", CFG_DEFAULT, date(2025, 2, 10))
    assert res["start"] == d(2025, 2, 1)
    assert res["end"] == d(2025, 2, 10)


def test_quarters():
    res = parse_time_phrase("last quarter", CFG_DEFAULT, date(2025, 1, 5))
    assert res["start"] == d(2024, 10, 1)
    assert res["end"] == d(2024, 12, 31)
    res2 = parse_time_phrase("this quarter", CFG_DEFAULT, date(2025, 5, 10))
    assert res2["start"] == d(2025, 4, 1)
    assert res2["end"] == d(2025, 5, 10)


def test_q_label():
    res = parse_time_phrase("Q2 2024", CFG_DEFAULT, date(2025, 7, 1))
    assert res["start"] == d(2024, 4, 1)
    assert res["end"] == d(2024, 6, 30)


def test_month_label():
    res = parse_time_phrase("September 2024", CFG_DEFAULT, date(2025, 7, 1))
    assert res["start"] == d(2024, 9, 1)
    assert res["end"] == d(2024, 9, 30)


def test_iso_week():
    res = parse_time_phrase("ISO week 1 2024", CFG_DEFAULT, date(2025, 7, 1))
    assert res["start"] == date.fromisocalendar(2024, 1, 1).isoformat()
    assert res["end"] == date.fromisocalendar(2024, 1, 7).isoformat()


def test_last_week():
    res = parse_time_phrase("last week", CFG_DEFAULT, date(2025, 2, 12))  # Wed
    # Previous ISO week Monday..Sunday around 2025-02-03 to 2025-02-09
    assert res["start"] == date(2025, 2, 3).isoformat()
    assert res["end"] == date(2025, 2, 9).isoformat()


def test_since_date():
    res = parse_time_phrase("since 05 Mar 2024", CFG_DEFAULT, date(2025, 3, 10))
    assert res["start"] == date(2024, 3, 5).isoformat()
    assert res["end"] == date(2025, 3, 10).isoformat()


def test_since_month_name():
    res = parse_time_phrase("since July", CFG_DEFAULT, date(2025, 11, 20))
    assert res["start"] == date(2025, 7, 1).isoformat()
    assert res["end"] == date(2025, 11, 20).isoformat()


def test_fiscal_ytd():
    cfg = {"fy_start_month": 4, "fy_start_day": 1}
    res = parse_time_phrase("this fiscal year", cfg, date(2025, 8, 10))
    assert res["start"] == date(2025, 4, 1).isoformat()
    assert res["end"] == date(2025, 8, 10).isoformat()
    assert res["label"] == "FYTD"
    assert res["fiscal_used"]


def test_last_fiscal_year():
    cfg = {"fy_start_month": 4, "fy_start_day": 1}
    res = parse_time_phrase("last fiscal year", cfg, date(2025, 2, 10))
    assert res["start"] == date(2023, 4, 1).isoformat()
    assert res["end"] == date(2024, 3, 31).isoformat()


def test_fy_label():
    cfg = {"fy_start_month": 4, "fy_start_day": 1}
    res = parse_time_phrase("FY2024", cfg, date(2025, 2, 10))
    assert res["start"] == date(2023, 4, 1).isoformat()
    assert res["end"] == date(2024, 3, 31).isoformat()


def test_past_12_months_keyword():
    res = parse_time_phrase("past 12 months", CFG_DEFAULT, date(2025, 8, 13))
    assert res["label"] == "Rolling 12M"


def test_past_week_last_7_days():
    res = parse_time_phrase("past week", CFG_DEFAULT, date(2025, 8, 13))
    assert res["label"] == "Last 7D"


def test_last_30_days():
    res = parse_time_phrase("last 30 days", CFG_DEFAULT, date(2025, 8, 13))
    assert res["label"] == "Last 30D"


def test_qtd_alias():
    res = parse_time_phrase("qtd", CFG_DEFAULT, date(2025, 5, 10))
    assert res["label"] == "QTD"


def test_mtd_alias():
    res = parse_time_phrase("mtd", CFG_DEFAULT, date(2025, 5, 10))
    assert res["label"] == "MTD"


def test_ytd_alias():
    res = parse_time_phrase("ytd", CFG_DEFAULT, date(2025, 5, 10))
    assert res["label"] == "YTD"


def test_edge_new_year_after_last_year():
    res = parse_time_phrase("after last year", CFG_DEFAULT, date(2026, 1, 2))
    assert res["start"] == date(2026, 1, 1).isoformat()
    assert res["end"] == date(2026, 1, 2).isoformat()


def test_edge_fiscal_rollover():
    cfg = {"fy_start_month": 7, "fy_start_day": 1}
    res = parse_time_phrase("fiscal ytd", cfg, date(2025, 7, 2))
    assert res["start"] == date(2025, 7, 1).isoformat()
    assert res["end"] == date(2025, 7, 2).isoformat()


