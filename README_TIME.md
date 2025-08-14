Time Phrase Policy (v1)

This document summarizes the time window interpretation policy used by the planner and compiler.

Policy highlights
- last year: previous calendar year (Jan 1 → Dec 31 of prior year)
- past year / past 12 months / rolling 12 months / trailing 12 months / TTM / T12M: rolling 365 days ending today (inclusive)
- after last year: current calendar YTD (Jan 1 → today)
- this year / year to date / YTD: current calendar YTD
- last month: previous calendar month
- this month / MTD: month-to-date
- last quarter: previous calendar quarter (calendar quarters)
- this quarter / QTD: quarter-to-date (calendar)
- fiscal (uses fy_start_month/day per tenant):
  - this fiscal year / fiscal YTD / FYTD: from current FY start → today
  - last fiscal year: previous full fiscal year
  - FYYYYY or fiscal year YYYY: FY labeled by end year
- labels:
  - Qn YYYY (calendar quarter)
  - <Month> YYYY (calendar month)
- ISO week: "ISO week WW YYYY" or "week WW YYYY" → Monday..Sunday
- Relative days: last N days, past week (7 days)
- Since: "since <DD Mon YYYY>" or "since <Month>" (current year) → today
- Range phrases: "between <Month> and <Month> YYYY", "until <Month> YYYY" / "up to <Month> YYYY"

API
- parse_time_phrase(text, client_cfg, today=None) → {start, end, label, policy_id, fiscal_used}
- sql_window_clause(col_expr, col_type, window) → (clause, params)

Inclusivity rules
- DATE columns: date_col BETWEEN :start AND :end (inclusive)
- TIMESTAMP columns: ts_col >= :start AND ts_col < DATE(:end, '+1 day')

Timezone
- Use tenant/client timezone when provided (X-Timezone / request.timezone) or environment TZ_DEFAULT/TIMEZONE. Fallback to UTC.

Unparseable phrases
- Return policy_id="unparsed"; planner should not invent dates and must show a friendly error.

Config knobs
- TIMEZONE/TZ_DEFAULT (e.g., Europe/Paris)
- FY_START_MONTH/FY_START_DAY
- DEFAULT_WINDOW (fallback behavior; currently YTD)

Acceptance example
- “volunteering hours after the last year” on 2025-08-13 ⇒ {start: 2025-01-01, end: 2025-08-13, label: "YTD"}

