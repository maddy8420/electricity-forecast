# Data Integrity Audit & Historical Findings
### India Electricity Demand, Generation & Energy Mix Forecasting Project

---

## 1. Data Sources

| Source | Coverage | Resolution |
|---|---|---|
| CEA Daily Power Generation by Source | 2013-03 to 2023-03 | Daily |
| CEA/Grid India Daily State-wise Demand & Energy Met | 2015-01 to 2023-03 | Daily |
| Hourly National + Regional Load (Grid India/RLDC) | 2019-01 to 2024-09 | Hourly |
| State-wise Electricity Consumption | 2013-01 to 2024-09 | Daily |
| Monthly Temperature (IMD-derived) | 2019-2021 | Monthly |

Two live API sources were evaluated and abandoned: CEA's public API endpoints
(11/11 tested endpoints timed out identically, indicating a deprecated
backend, not transient slowness) and data.gov.in (account/API-key friction
outweighed the marginal value for a historical-forecasting use case).
Curated, pre-validated datasets proved more reliable than live government
API access for this project's needs.

---

## 2. Data Integrity Issues Found & Fixed

A clean-looking pipeline was not the goal — a *validated* one was. Three
issues were found that would have silently produced confidently wrong
conclusions if not caught before analysis.

### 2.1 Source-category double-counting (hierarchy problem)

The generation-by-source file contains both parent categories (`Thermal
(Coal & Lignite)`, `RES (Wind, Solar, Biomass & Others)`) and their
itemized children (`Coal`, `Lignite`, `Wind`, `Solar`) — but CEA reports
them on **different, non-overlapping date ranges** as its own publishing
format evolved:

- `Thermal (Coal & Lignite)`: 2017-05-26 to 2018-11-27
- `Coal` / `Lignite` (itemized separately): 2018-11-29 onward
- `RES (combined)`: 2017-05-26 to 2023-03-31
- `Wind` / `Solar` (itemized separately): 2013-2017 only

**Naively summing every non-"Total" row per date double-counts** on any
date both a parent and child category happen to be present. Fixed by
building harmonized columns that select the correct granularity per date
rather than summing indiscriminately, and by trusting CEA's own reported
`Total` row directly rather than recomputing it from parts.

### 2.2 A ~5-month reporting blackout at the format-transition boundary

Investigating a sharp anomaly in the generation trend around May 2017
revealed that CEA's own `Total` row was barely published during the
transition: March 2017 had 31/31 days reported, April had 28, **May had
only 8, June and July had zero days reported at all**, August had 4,
September had 1. This is a genuine gap in the source data, not a script
bug — confirmed by cross-checking against `Hydro`, which was reported
continuously throughout the same window.

### 2.3 Uneven monthly day-coverage silently corrupting growth rates

Aggregating daily data to monthly resolution via `SUM()` (the initial,
intuitive approach) proved fragile: any month with unusually low
day-coverage — e.g. **December 2022 had only 4 of ~30 days reported** —
produced an artificially tiny monthly total, which then generated a false
**~630% year-over-year growth spike** when compared against a normally-
reported December 2023. Nine months across the dataset had fewer than 20
days reported. **Fixed by switching monthly aggregation to average daily
value** (robust to uneven coverage) with an explicit day-coverage flag
retained per metric, so any future analysis can identify and exclude
unreliable months rather than being silently misled by them.

### 2.4 A misleading percentage-share baseline

The first data point in an early version of the renewable-share
trend (May 2017) showed renewables at 23% of generation — an apparent
sharp decline to 11% by 2023 that would have made a striking, wrong
headline. Root cause: `Coal`/`Lignite` were `NaN` for all but one day
that month due to the same reporting transition (§2.2), and standard
sum-based aggregation silently treats `NaN` as zero — inflating
renewables' *share* of an artificially small denominator. Fixed by
scoping the source-mix trend to the window where all components are
reliably co-reported (Dec 2018 onward).

---

## 3. Historical Findings (validated)

With the above corrections in place, the following patterns are supported
by the data and cross-checked for plausibility against known real-world
events:

- **2014-2019: steady baseline growth.** Generation and demand grew
  4-11% YoY most years — consistent with pre-pandemic Indian electricity
  demand trends, unremarkable but a solid baseline for forecasting.
- **Mid-2020: COVID-19 demand crash.** A sharp V-shaped dip to
  approximately **−25% YoY**, consistent with widely reported real-world
  impacts of India's April-May 2020 national lockdown.
- **2021: apparent ~40% YoY growth is a base effect, not a second boom.**
  This reflects comparison against the artificially depressed 2020
  lockdown month, not a genuine demand surge — a distinction worth making
  explicitly, since the raw number alone overstates the story.
- **2022-2023: genuine sustained double-digit growth (9-16% YoY),** past
  the COVID base-effect window — plausibly tied to post-pandemic economic
  recovery and the well-documented 2022-2023 heatwave-driven peak demand
  events in India (cross-reference with contemporary news sources
  recommended before final publication).
- **Renewables share grew modestly from ~7.6% (Dec 2018) to ~11.2% (Mar
  2023),** with coal remaining dominant (65-80%) and a consistent
  seasonal dip each monsoon season as hydro output rises — a physically
  sensible pattern that lends confidence to the underlying data.
- **Peak-to-average demand gap rose from 9.2% (2019) to 10.3% (2023)**
  (full calendar years only; 2024 excluded from this comparison due to
  partial-year coverage) — a modest but real signal of increasingly
  "peaky" demand, plausibly linked to rising AC penetration.
- **Demand-temperature correlation: r = 0.37** (moderate positive) —
  directionally as expected, though the underlying temperature dataset's
  limited window (36 months) constrains confidence in this estimate.

---

## 4. Methodology Note on AI-Assisted Development

This pipeline was built with AI assistance (Claude) for code generation
and iteration speed. Every one of the four issues in §2 was caught through
**iterative validation against the data itself** — inspecting raw
category values, checking date-range coverage, and cross-referencing
computed totals against reported ones — not assumed correct because the
code ran without errors. A single-shot AI-generated version of this
pipeline would very likely have reproduced all four issues silently,
since none of them raise an exception; they simply produce confidently
wrong numbers. This validation process is treated as a core project
deliverable, not incidental debugging.

---

## 5. Next Steps

1. Feature engineering: lag features, rolling statistics, exogenous
   regressors (temperature, calendar effects) for the forecasting stage.
2. Baseline forecasting models (Naive, Moving Average, Exponential
   Smoothing) as a floor for comparison.
3. ARIMA/SARIMA/Prophet with walk-forward validation.
4. Benchmark forecasts against CEA's own published Electric Power Survey
   projections.
5. Demand-supply gap projection and energy-mix scenario modeling.