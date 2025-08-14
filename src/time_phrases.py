"""
Time phrase interpreter with fiscal/business semantics.

Policy (v1):
- "last year" => previous calendar year (Jan 1 → Dec 31 of prior year).
- "past year", "past 12 months", "rolling 12 months", "trailing 12 months", "ttm", "t12m" => rolling 365 days ending today (inclusive).
- Range phrases:
  * "between <Month> and <Month> YYYY" => start at first day of first month, end at last day of end month.
  * "until <Month> YYYY" / "up to <Month> YYYY" => from Jan 1 of that year to end of that month.
- "after last year" => current calendar YTD (Jan 1 of current year → today).
- "this year", "year to date", "ytd" => current calendar YTD.
- "last month" => previous calendar month.
- "this month" => month-to-date.
- "last quarter" => previous calendar quarter.
- "this quarter", "quarter to date", "qtd" => calendar quarter-to-date.
- "fiscal" variants use client_cfg fiscal start (fy_start_month, fy_start_day):
  * "fyYYYY" or "fiscal year YYYY" => fiscal year labeled by end year (e.g., FY2024 with Apr 1 start is 2023-04-01 → 2024-03-31).
  * "this fiscal year", "fiscal ytd", "fytd" => fiscal year-to-date (from current FY start → today).
  * "last fiscal year" => previous full fiscal year.
- Specific calendar references:
  * "Qn YYYY" (calendar quarters) => the specified quarter range.
  * "<Month> YYYY" => that calendar month range.
  * "ISO week WW YYYY" / "week WW YYYY" => ISO week (Monday→Sunday) range.
- Relative days:
  * "last N days" => previous N days ending today (inclusive).
  * "past week" => last 7 days ending today (inclusive).
- "since <date|month>" => from specified date or month start (current year if no year) → today.

Return shape: { start: YYYY-MM-DD, end: YYYY-MM-DD, label: str, policy_id: str, fiscal_used: bool }
"""

from __future__ import annotations

import re
import calendar
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple


MonthMap = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
for i, m in enumerate(calendar.month_abbr):
    if i and m:
        MonthMap[m.lower()] = i


def _as_date(d: date | datetime) -> date:
    return d.date() if isinstance(d, datetime) else d


def _fmt(d: date) -> str:
    return d.isoformat()


def _start_of_month(d: date) -> date:
    return d.replace(day=1)


def _end_of_month(d: date) -> date:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return d.replace(day=last_day)


def _calendar_quarter(dt: date) -> int:
    return ((dt.month - 1) // 3) + 1


def _quarter_start_end(year: int, quarter: int) -> Tuple[date, date]:
    start_month = (quarter - 1) * 3 + 1
    start = date(year, start_month, 1)
    end_month = start_month + 2
    end = _end_of_month(date(year, end_month, 1))
    return start, end


def _fiscal_year_start_for(dt: date, fy_start_month: int, fy_start_day: int) -> date:
    start_this_year = date(dt.year, fy_start_month, fy_start_day)
    if dt >= start_this_year:
        return start_this_year
    return date(dt.year - 1, fy_start_month, fy_start_day)


def _fiscal_year_label(start: date, fy_start_month: int, fy_start_day: int) -> int:
    # Label FY by end year (common practice)
    end = _fiscal_year_end(start, fy_start_month, fy_start_day)
    return end.year


def _fiscal_year_end(start: date, fy_start_month: int, fy_start_day: int) -> date:
    # One year minus one day from start
    next_start_year = start.year + 1
    next_start = date(next_start_year, fy_start_month, fy_start_day)
    return next_start - timedelta(days=1)


def _iso_week_range(year: int, week: int) -> Tuple[date, date]:
    start = date.fromisocalendar(year, week, 1)
    end = date.fromisocalendar(year, week, 7)
    return start, end


def _last_calendar_year_range(today: date) -> Tuple[date, date]:
    return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)


def _calendar_ytd(today: date) -> Tuple[date, date]:
    return date(today.year, 1, 1), today


def _rolling_days(today: date, days: int) -> Tuple[date, date]:
    return today - timedelta(days=days), today


def _last_month_range(today: date) -> Tuple[date, date]:
    first_this = date(today.year, today.month, 1)
    last_prev = first_this - timedelta(days=1)
    start_prev = date(last_prev.year, last_prev.month, 1)
    return start_prev, last_prev


def _this_month_to_date(today: date) -> Tuple[date, date]:
    return date(today.year, today.month, 1), today


def _last_quarter_range(today: date) -> Tuple[date, date]:
    q = _calendar_quarter(today)
    year = today.year
    if q == 1:
        return _quarter_start_end(year - 1, 4)
    return _quarter_start_end(year, q - 1)


def _this_quarter_to_date(today: date) -> Tuple[date, date]:
    q = _calendar_quarter(today)
    start, _ = _quarter_start_end(today.year, q)
    return start, today


def parse_time_phrase(text: str, client_cfg: Dict[str, int | str | bool], today: Optional[date | datetime] = None) -> Dict[str, str | bool]:
    """Parse a time phrase and return {start, end, label, policy_id, fiscal_used}.

    client_cfg expects keys: fy_start_month (1-12), fy_start_day (1-31). Optional.
    """
    tz = _as_date(today) if today else date.today()
    s = (text or '').strip().lower()
    fiscal_used = False

    fy_m = int(client_cfg.get('fy_start_month') or 1)
    fy_d = int(client_cfg.get('fy_start_day') or 1)

    # Acceptance: detect phrase anywhere (e.g., "... after the last year")
    if re.search(r"\bafter (the )?last year\b", s):
        start, end = _calendar_ytd(tz)
        return {"start": _fmt(start), "end": _fmt(end), "label": "YTD", "policy_id": "calendar.ytd.after_last_year", "fiscal_used": False}

    # YTD variants
    if re.search(r"\b(ytd|year to date|this year)\b", s):
        start, end = _calendar_ytd(tz)
        return {"start": _fmt(start), "end": _fmt(end), "label": "YTD", "policy_id": "calendar.ytd", "fiscal_used": False}

    # Last year
    if re.search(r"\b(last year|previous year)\b", s):
        start, end = _last_calendar_year_range(tz)
        return {"start": _fmt(start), "end": _fmt(end), "label": "LY", "policy_id": "calendar.last_year", "fiscal_used": False}

    # Rolling 12 months
    if re.search(r"\b(past (year|12 months)|rolling 12 months|trailing 12 months|last 12 months|ttm|t12m)\b", s):
        start, end = _rolling_days(tz, 365)
        return {"start": _fmt(start), "end": _fmt(end), "label": "Rolling 12M", "policy_id": "rolling.12m", "fiscal_used": False}

    # Last N days / past week
    m = re.search(r"\blast (\d+) days\b", s)
    if m:
        n = int(m.group(1))
        start, end = _rolling_days(tz, n)
        return {"start": _fmt(start), "end": _fmt(end), "label": f"Last {n}D", "policy_id": "rolling.ndays", "fiscal_used": False}
    if re.search(r"\bpast week\b", s):
        start, end = _rolling_days(tz, 7)
        return {"start": _fmt(start), "end": _fmt(end), "label": "Last 7D", "policy_id": "rolling.7d", "fiscal_used": False}

    # Month and quarter
    if re.search(r"\blast month\b", s):
        start, end = _last_month_range(tz)
        return {"start": _fmt(start), "end": _fmt(end), "label": "Last Month", "policy_id": "calendar.last_month", "fiscal_used": False}
    if re.search(r"\b(this month|month to date|mtd)\b", s):
        start, end = _this_month_to_date(tz)
        return {"start": _fmt(start), "end": _fmt(end), "label": "MTD", "policy_id": "calendar.mtd", "fiscal_used": False}
    if re.search(r"\blast quarter\b", s):
        start, end = _last_quarter_range(tz)
        return {"start": _fmt(start), "end": _fmt(end), "label": "Last Quarter", "policy_id": "calendar.last_quarter", "fiscal_used": False}
    if re.search(r"\b(this quarter|quarter to date|qtd)\b", s):
        start, end = _this_quarter_to_date(tz)
        return {"start": _fmt(start), "end": _fmt(end), "label": "QTD", "policy_id": "calendar.qtd", "fiscal_used": False}

    qcal = re.search(r"\bq([1-4])\s*(\d{4})\b", s)
    if qcal:
        q = int(qcal.group(1))
        yr = int(qcal.group(2))
        start, end = _quarter_start_end(yr, q)
        return {"start": _fmt(start), "end": _fmt(end), "label": f"Q{q} {yr}", "policy_id": "calendar.q_label", "fiscal_used": False}

    mcal = re.search(r"\b([a-zA-Z]{3,9})\s+(\d{4})\b", s)
    if mcal and mcal.group(1).lower() in MonthMap:
        mon = MonthMap[mcal.group(1).lower()]
        yr = int(mcal.group(2))
        start = date(yr, mon, 1)
        end = _end_of_month(start)
        return {"start": _fmt(start), "end": _fmt(end), "label": f"{calendar.month_abbr[mon]} {yr}", "policy_id": "calendar.month_label", "fiscal_used": False}

    # Between Month and Month YYYY
    bet = re.search(r"\bbetween\s+([a-zA-Z]{3,9})\s+and\s+([a-zA-Z]{3,9})\s+(\d{4})\b", s)
    if bet:
        m1 = bet.group(1).lower()
        m2 = bet.group(2).lower()
        yr = int(bet.group(3))
        if m1 in MonthMap and m2 in MonthMap:
            start = date(yr, MonthMap[m1], 1)
            end = _end_of_month(date(yr, MonthMap[m2], 1))
            return {"start": _fmt(start), "end": _fmt(end), "label": f"{calendar.month_abbr[MonthMap[m1]]}-{calendar.month_abbr[MonthMap[m2]]} {yr}", "policy_id": "calendar.between_months", "fiscal_used": False}

    # Until / up to Month YYYY (Jan 1 -> end of month)
    until = re.search(r"\b(until|up to)\s+([a-zA-Z]{3,9})\s+(\d{4})\b", s)
    if until and until.group(2).lower() in MonthMap:
        mon = MonthMap[until.group(2).lower()]
        yr = int(until.group(3))
        start = date(yr, 1, 1)
        end = _end_of_month(date(yr, mon, 1))
        return {"start": _fmt(start), "end": _fmt(end), "label": f"YTD to {calendar.month_abbr[mon]} {yr}", "policy_id": "calendar.until_month", "fiscal_used": False}

    # ISO week
    w = re.search(r"\b(iso\s+week|week)\s+(\d{1,2})\s+(\d{4})\b", s)
    if w:
        week = int(w.group(2))
        yr = int(w.group(3))
        start, end = _iso_week_range(yr, week)
        return {"start": _fmt(start), "end": _fmt(end), "label": f"ISO Week {week} {yr}", "policy_id": "calendar.iso_week", "fiscal_used": False}
    if re.search(r"\blast week\b", s):
        # Monday..Sunday of previous ISO week
        this_mon = tz - timedelta(days=tz.isoweekday() - 1)
        start = this_mon - timedelta(days=7)
        end = start + timedelta(days=6)
        return {"start": _fmt(start), "end": _fmt(end), "label": "Last Week", "policy_id": "calendar.last_week", "fiscal_used": False}

    # Since <date|month>
    s_since = re.search(r"\bsince\s+(.+)$", s)
    if s_since:
        tail = s_since.group(1).strip()
        # Try DD MMM YYYY
        mdy = re.search(r"(\d{1,2})\s+([a-zA-Z]{3,9})\s+(\d{4})", tail)
        if mdy and mdy.group(2).lower() in MonthMap:
            day = int(mdy.group(1))
            mon = MonthMap[mdy.group(2).lower()]
            yr = int(mdy.group(3))
            start = date(yr, mon, day)
            return {"start": _fmt(start), "end": _fmt(tz), "label": "Since Date", "policy_id": "calendar.since_date", "fiscal_used": False}
        # Try month only (current year)
        if tail.split()[0].lower() in MonthMap:
            mon = MonthMap[tail.split()[0].lower()]
            start = date(tz.year, mon, 1)
            return {"start": _fmt(start), "end": _fmt(tz), "label": "Since Month", "policy_id": "calendar.since_month", "fiscal_used": False}

    # Fiscal phrases
    if re.search(r"\b(fytd|fiscal ytd|this fiscal year)\b", s):
        fiscal_used = True
        fy_start = _fiscal_year_start_for(tz, fy_m, fy_d)
        return {"start": _fmt(fy_start), "end": _fmt(tz), "label": "FYTD", "policy_id": "fiscal.ytd", "fiscal_used": True}

    if re.search(r"\blast fiscal year\b", s):
        fiscal_used = True
        curr_start = _fiscal_year_start_for(tz, fy_m, fy_d)
        last_start = _fiscal_year_start_for(curr_start - timedelta(days=1), fy_m, fy_d)
        start, end = last_start, _fiscal_year_end(last_start, fy_m, fy_d)
        label = f"FY{_fiscal_year_label(start, fy_m, fy_d)}"
        return {"start": _fmt(start), "end": _fmt(end), "label": label, "policy_id": "fiscal.last_year", "fiscal_used": True}

    fy_label = re.search(r"\bfy\s*(\d{4})\b|\bfiscal year\s*(\d{4})\b", s)
    if fy_label:
        fiscal_used = True
        year = int(fy_label.group(1) or fy_label.group(2))
        # Label by end year -> start is previous year at fy start
        start = date(year - 1, fy_m, fy_d)
        end = _fiscal_year_end(start, fy_m, fy_d)
        return {"start": _fmt(start), "end": _fmt(end), "label": f"FY{year}", "policy_id": "fiscal.fy_label", "fiscal_used": True}

    # Fallback: if input has no temporal hint, return unparsed for caller to handle
    if s:
        return {"start": _fmt(_calendar_ytd(tz)[0]), "end": _fmt(tz), "label": "", "policy_id": "unparsed", "fiscal_used": False}
    # Empty -> default YTD
    start, end = _calendar_ytd(tz)
    return {"start": _fmt(start), "end": _fmt(end), "label": "YTD", "policy_id": "calendar.ytd.fallback", "fiscal_used": False}


def sql_window_clause(col_expr: str, col_type: str, window: Dict[str, str]) -> Tuple[str, Tuple[str, str]]:
    """Return SQL clause and params implementing inclusivity rules for a window.

    - If col_type == 'DATE': date_col BETWEEN :start AND :end (inclusive)
    - If col_type == 'TIMESTAMP': ts_col >= :start AND ts_col < DATE(:end, '+1 day')
    """
    start = window.get('start')
    end = window.get('end')
    if col_type.upper() == 'DATE':
        return f"{col_expr} BETWEEN ? AND ?", (start, end)
    # TIMESTAMP default
    return f"{col_expr} >= ? AND {col_expr} < date(?, '+1 day')", (start, end)


__all__ = [
    'parse_time_phrase', 'sql_window_clause',
    '_iso_week_range', '_quarter_start_end', '_fiscal_year_start_for', '_fiscal_year_end',
]


