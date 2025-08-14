from datetime import date

from time_phrases import parse_time_phrase


CFG_JULY = {"fy_start_month": 7, "fy_start_day": 1}


def test_year_boundary_jan1():
    res = parse_time_phrase("after last year", {"fy_start_month": 4, "fy_start_day": 1}, date(2026, 1, 1))
    assert res["start"] == date(2026, 1, 1).isoformat()
    assert res["end"] == date(2026, 1, 1).isoformat()


def test_year_boundary_dec31():
    res = parse_time_phrase("last year", {"fy_start_month": 4, "fy_start_day": 1}, date(2025, 12, 31))
    assert res["start"] == date(2024, 1, 1).isoformat()
    assert res["end"] == date(2024, 12, 31).isoformat()


def test_leap_day():
    res = parse_time_phrase("since 29 Feb 2024", {"fy_start_month": 4, "fy_start_day": 1}, date(2024, 3, 5))
    assert res["start"] == date(2024, 2, 29).isoformat()
    assert res["end"] == date(2024, 3, 5).isoformat()


def test_aliases_ttm_t12m():
    ttm = parse_time_phrase("TTM", {"fy_start_month": 4, "fy_start_day": 1}, date(2025, 8, 13))
    t12m = parse_time_phrase("T12M", {"fy_start_month": 4, "fy_start_day": 1}, date(2025, 8, 13))
    assert ttm["label"] == "Rolling 12M"
    assert t12m["label"] == "Rolling 12M"


def test_last_90_days_last_6_months():
    r90 = parse_time_phrase("last 90 days", {"fy_start_month": 4, "fy_start_day": 1}, date(2025, 8, 13))
    r6m = parse_time_phrase("last 6 months", {"fy_start_month": 4, "fy_start_day": 1}, date(2025, 8, 13))
    assert r90["policy_id"] in ("rolling.ndays", "rolling.12m")  # structure check
    assert r6m["policy_id"] in ("rolling.ndays", "rolling.12m")


def test_between_and_until():
    between = parse_time_phrase("between May and July 2025", CFG_JULY, date(2025, 10, 1))
    assert between["start"] == date(2025, 5, 1).isoformat()
    assert between["end"] == date(2025, 7, 31).isoformat()
    until = parse_time_phrase("up to May 2025", CFG_JULY, date(2025, 10, 1))
    assert until["start"] == date(2025, 1, 1).isoformat()
    assert until["end"] == date(2025, 5, 31).isoformat()


def test_ambiguous_unparsed():
    res = parse_time_phrase("recently", CFG_JULY, date(2025, 8, 13))
    assert res["policy_id"] == "unparsed"


