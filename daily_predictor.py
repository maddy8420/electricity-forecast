"""Predict daily electricity demand from monthly forecasts and daily profiles. Run with: python daily_predictor.py."""

import warnings
import numpy as np
import pandas as pd

_HOLIDAYS_AVAILABLE = False
try:
    import holidays as _holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    pass

# holidays.India supports 2001-2035 as of v0.107.
_HOLIDAYS_SUPPORTED_YEARS = set(range(2001, 2036))

COVID_START = pd.Timestamp("2020-03-24")
COVID_END   = pd.Timestamp("2020-06-30")

# Outlier threshold: flag days where energy_met deviates from month median
# Ratios above 1.4 or below 0.6 are treated as duplicate-reporting artefacts.
OUTLIER_HI = 1.4
OUTLIER_LO = 0.6

# Minimum days in a month for the month to be used
MIN_DAYS_MONTH = 20

def _get_india_holidays(year: int) -> dict:
    """Return {date: name} for India national holidays in a given year."""
    if not _HOLIDAYS_AVAILABLE:
        return {}
    if year not in _HOLIDAYS_SUPPORTED_YEARS:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dict(_holidays_lib.India(years=[year]))


def _holiday_name(d: pd.Timestamp) -> "str | None":
    """Return holiday name for date d, or None if not a holiday."""
    h = _get_india_holidays(d.year)
    return h.get(d.date(), None)



def build_profile(daily: pd.DataFrame) -> dict:
    """Build a day-of-week demand profile from reported daily data."""
    df = daily[["date", "national_energy_met_mu", "national_max_demand_mw"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["national_energy_met_mu"])
    df = df.sort_values("date").reset_index(drop=True)

    # Exclude the COVID lockdown window from profile calculations.
    df = df[~((df["date"] >= COVID_START) & (df["date"] <= COVID_END))].copy()

    df["_ym"] = df["date"].dt.to_period("M")
    df["_month_median"] = df.groupby("_ym")["national_energy_met_mu"].transform("median")
    df["_ratio_raw"] = df["national_energy_met_mu"] / df["_month_median"]
    df["_flagged"] = (df["_ratio_raw"] > OUTLIER_HI) | (df["_ratio_raw"] < OUTLIER_LO)

    non_flagged = df[~df["_flagged"]]
    cov = non_flagged.groupby("_ym")["date"].count().rename("n_days")
    good_months = cov[cov >= MIN_DAYS_MONTH].index
    df = df[df["_ym"].isin(good_months)].copy()

    clean = df[~df["_flagged"]].copy()
    clean_mean = clean.groupby("_ym")["national_energy_met_mu"].mean().rename("_clean_mean")
    df = df.join(clean_mean, on="_ym")
    df["ratio"] = df["national_energy_met_mu"] / df["_clean_mean"]

    df["month"] = df["date"].dt.month
    df["dow"]   = df["date"].dt.dayofweek   # 0=Mon…6=Sun

    if _HOLIDAYS_AVAILABLE:
        holiday_cache: dict = {}
        def _is_holiday(d):
            yr = d.year
            if yr not in holiday_cache:
                holiday_cache[yr] = _get_india_holidays(yr)
            return d.date() in holiday_cache[yr]
        df["_is_holiday"] = df["date"].apply(lambda d: _is_holiday(d))
    else:
        df["_is_holiday"] = False

    work = df[~df["_flagged"] & ~df["_is_holiday"]].copy()

    p01 = work["ratio"].quantile(0.01)
    p99 = work["ratio"].quantile(0.99)
    work["ratio_w"] = work["ratio"].clip(p01, p99)

    def _stats(group):
        r = group["ratio_w"]
        return pd.Series({
            "mean": r.mean(),
            "q05":  r.quantile(0.05),
            "q95":  r.quantile(0.95),
            "n":    len(r),
        })

    dow_df = work.groupby("dow").apply(_stats, include_groups=False)
    profile = {d: row.to_dict() for d, row in dow_df.iterrows()}
    dow_fallback = profile  # same for API compatibility

    clean_pf = df[~df["_flagged"]].copy()
    clean_pf["_pf"] = (
        clean_pf["national_max_demand_mw"]
        / (clean_pf["national_energy_met_mu"] * 1000.0 / 24.0)
    )
    peak_factor = clean_pf.groupby("month")["_pf"].median().to_dict()

    hol_df = df[df["_is_holiday"] & ~df["_flagged"]].copy()
    if _HOLIDAYS_AVAILABLE and len(hol_df) > 0:
        name_cache: dict = {}
        def _hname(d):
            yr = d.year
            if yr not in name_cache:
                name_cache[yr] = _get_india_holidays(yr)
            return name_cache[yr].get(d.date(), "Unknown")
        hol_df["_hname"] = hol_df["date"].apply(lambda d: _hname(d))

        def _prof_mean(row):
            dow_key = row["dow"]
            return profile.get(dow_key, {"mean": 1.0})["mean"]
        hol_df["_prof_mean"] = hol_df.apply(_prof_mean, axis=1)
        hol_df["_ratio_w"] = hol_df["ratio"].clip(p01, p99)
        hol_df["_hol_residual"] = hol_df["_ratio_w"] / hol_df["_prof_mean"]

        hol_stats = hol_df.groupby("_hname")["_hol_residual"].agg(["median", "count"])

        # Shrinkage: m = 1 + (raw - 1) * n / (n + 5)  when n >= 5, else 1.0
        def _shrunk_mult(row):
            n = row["count"]
            raw = row["median"]
            if n >= 5:
                return 1.0 + (raw - 1.0) * n / (n + 5.0)
            else:
                return 1.0
        hol_stats["multiplier"] = hol_stats.apply(_shrunk_mult, axis=1)
        hol_stats = hol_stats.sort_values("multiplier")
        holiday_name_to_mult = hol_stats["multiplier"].to_dict()
        holiday_mults = hol_stats.reset_index().rename(columns={"_hname": "holiday", "count": "n"})
    else:
        holiday_name_to_mult = {}
        holiday_mults = pd.DataFrame(columns=["holiday", "median", "n", "multiplier"])

    flagged_set = set(df[df["_flagged"]]["date"])
    flagged_per_year = (
        df[df["_flagged"]].set_index("date")
        .resample("YE")
        .size()
        .rename("count")
        .pipe(lambda s: {k.year: v for k, v in s.items() if v > 0})
    )

    return {
        "profile":               profile,
        "dow_fallback":          dow_fallback,
        "peak_factor":           peak_factor,
        "holiday_name_to_mult":  holiday_name_to_mult,
        "holiday_mults":         holiday_mults,
        "df":                    df,
        "flagged_set":           flagged_set,
        "flagged_per_year":      flagged_per_year,
    }



def _get_day_profile(d: pd.Timestamp, prof: dict) -> dict:
    """Return {mean, q05, q95} for a given date using the DOW profile."""
    dow = d.dayofweek
    return prof["profile"].get(dow, {"mean": 1.0, "q05": 0.9, "q95": 1.1, "n": 0})



def _compute_consumption_ratio(master_monthly: pd.DataFrame, flagged_set: set,
                                daily_df: pd.DataFrame) -> tuple:
    """Compute the calibrated consumption-to-demand ratio from eligible months."""
    mm = master_monthly.copy()
    mm["year_month"] = mm["year_month"].astype(str).str[:7]

    mm = mm[(mm["year_month"] >= "2019-01") & (mm["year_month"] <= "2022-12")].copy()

    cons_col = "avg_daily_consumption" if "avg_daily_consumption" in mm.columns else None
    if cons_col is None or "avg_daily_demand" not in mm.columns:
        return 1.0, 0, True

    mm = mm.dropna(subset=[cons_col, "avg_daily_demand"])

    if "days_reported_consumption" in mm.columns:
        mm = mm[mm["days_reported_consumption"] >= 25]

    if "low_coverage_flag" in mm.columns:
        mm = mm[mm["low_coverage_flag"] == False]

    if flagged_set and daily_df is not None:
        df_tmp = daily_df[["date", "national_energy_met_mu"]].copy()
        df_tmp["date"] = pd.to_datetime(df_tmp["date"])
        df_tmp["_ym"] = df_tmp["date"].dt.to_period("M").dt.strftime("%Y-%m")
        months_with_flags = set(
            df_tmp[df_tmp["date"].isin(flagged_set)]["_ym"]
        )
        mm = mm[~mm["year_month"].isin(months_with_flags)]

    if len(mm) == 0:
        return 1.0, 0, True

    ratio_series = mm[cons_col] / mm["avg_daily_demand"]
    r = float(ratio_series.median())
    n = len(ratio_series)
    r_is_identity = abs(r - 1.0) < 0.005
    return r, n, r_is_identity


def predict_day(
    d,
    demand_fc: pd.DataFrame,
    gen_fc: pd.DataFrame,
    daily: pd.DataFrame,
    master_monthly: pd.DataFrame,
    profile: dict,
) -> dict:
    """Predict electricity demand for a single day."""
    notes = []
    d = pd.Timestamp(d)
    month_start = d.to_period("M").to_timestamp()

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    flagged_set = profile.get("flagged_set", set())

    holiday_name = None
    holiday_mult = 1.0
    try:
        holiday_name = _holiday_name(d)
        if holiday_name:
            holiday_mult = profile["holiday_name_to_mult"].get(holiday_name, 1.0)
    except Exception as e:
        notes.append(f"Holiday lookup failed: {e}; multiplier set to 1.0")
        holiday_mult = 1.0

    dp = _get_day_profile(d, profile)
    weekday_factor = dp["mean"]
    q05 = dp["q05"]
    q95 = dp["q95"]

    peak_factor = profile["peak_factor"].get(d.month, None)

    HIST_END = pd.Timestamp("2023-03-31")

    mm_tmp = master_monthly.copy()
    mm_tmp["year_month"] = mm_tmp["year_month"].astype(str).str[:7]
    cons_col = "avg_daily_consumption" if "avg_daily_consumption" in mm_tmp.columns else None
    if cons_col:
        cons_valid = mm_tmp.dropna(subset=[cons_col])
        if len(cons_valid) > 0:
            last_cons_month = cons_valid["year_month"].max()
        else:
            last_cons_month = None
    else:
        last_cons_month = None

    ym_str = d.to_period("M").strftime("%Y-%m")

    if d <= HIST_END:
        mode = "historical"
        validated = True

        mm_row = mm_tmp[mm_tmp["year_month"] == ym_str]
        if len(mm_row) == 0:
            return {
                "date": d.date(), "mode": "out_of_range",
                "notes": [f"Month {ym_str} not found in MASTER_monthly."]
            }

        days_reported = mm_row["days_reported_demand"].values[0]
        if pd.isna(days_reported) or days_reported < MIN_DAYS_MONTH:
            return {
                "date": d.date(), "mode": "historical",
                "baseline_mu": None,
                "notes": [f"Insufficient daily data for {ym_str} ({days_reported} days reported < {MIN_DAYS_MONTH})."]
            }

        df_month = daily[daily["date"].dt.to_period("M").dt.strftime("%Y-%m") == ym_str].copy()
        df_month_clean = df_month[~df_month["date"].isin(flagged_set)]
        if len(df_month_clean) >= MIN_DAYS_MONTH:
            baseline = float(df_month_clean["national_energy_met_mu"].mean())
        else:
            baseline = mm_row["avg_daily_demand"].values[0]
            if pd.isna(baseline):
                return {
                    "date": d.date(), "mode": "historical",
                    "baseline_mu": None,
                    "notes": [f"avg_daily_demand is NaN for {ym_str}."]
                }
            notes.append(f"Cleaned days < {MIN_DAYS_MONTH} for {ym_str}; using stored avg_daily_demand.")

        baseline_lo = baseline
        baseline_hi = baseline

        actual_row = daily[daily["date"] == d]
        if len(actual_row) == 1:
            actual_mu = float(actual_row["national_energy_met_mu"].values[0])
            if d in flagged_set:
                notes.append(
                    f"This day is flagged as an outlier (energy_met deviates from "
                    f"month median by >{OUTLIER_HI:.0%} or <{OUTLIER_LO:.0%}). "
                    f"Actual value ({actual_mu:,.1f} MU) may reflect reporting artefact."
                )
        else:
            actual_mu = None

    elif (last_cons_month is not None and ym_str <= last_cons_month
          and cons_col is not None):
        mode = "actual_monthly"
        validated = False
        actual_mu = None

        mm_row = mm_tmp[mm_tmp["year_month"] == ym_str]
        if len(mm_row) == 0 or pd.isna(mm_row[cons_col].values[0]):
            return {
                "date": d.date(), "mode": "actual_monthly",
                "baseline_mu": None,
                "notes": [f"avg_daily_consumption not available for {ym_str}."]
            }

        raw_cons = float(mm_row[cons_col].values[0])

        if "days_reported_consumption" in mm_row.columns:
            days_rep_cons = mm_row["days_reported_consumption"].values[0]
            if not pd.isna(days_rep_cons) and days_rep_cons < 20:
                notes.append(
                    f"days_reported_consumption for {ym_str} = {days_rep_cons:.0f} "
                    f"(< 20); consumption figure may be unreliable."
                )

        r, n_cal, r_is_identity = _compute_consumption_ratio(
            master_monthly, flagged_set, daily
        )
        if r_is_identity:
            baseline = raw_cons
        else:
            baseline = raw_cons / r
            notes.append(
                f"Consumption-to-demand calibration: r={r:.4f} "
                f"(over {n_cal} months 2019-01..2022-12); "
                f"baseline = consumption / r = {raw_cons:,.1f} / {r:.4f} = {baseline:,.1f} MU/day."
            )

        baseline_lo = baseline
        baseline_hi = baseline

    elif month_start in demand_fc.index:
        mode = "forecast"
        row = demand_fc.loc[month_start]
        baseline    = float(row["forecast"])
        baseline_lo = float(row["ci_lower_95"])
        baseline_hi = float(row["ci_upper_95"])
        validated   = bool(row["is_validated_horizon"])
        actual_mu   = None

    else:
        return {
            "date": d.date(), "mode": "out_of_range",
            "baseline_mu": None,
            "notes": [f"Date {d.date()} is outside all available data and forecast ranges."]
        }

    estimate = baseline    * weekday_factor * holiday_mult
    low      = baseline_lo * q05            * holiday_mult
    high     = baseline_hi * q95            * holiday_mult

    peak_mw = None
    if peak_factor is not None:
        peak_mw = estimate * 1000.0 / 24.0 * peak_factor

    result = {
        "date":             d.date(),
        "mode":             mode,
        "baseline_mu":      round(baseline, 2),
        "estimate_mu":      round(estimate, 2),
        "low_mu":           round(low, 2),
        "high_mu":          round(high, 2),
        "actual_mu":        round(actual_mu, 2) if actual_mu is not None else None,
        "weekday_factor":   round(weekday_factor, 4),
        "holiday_name":     holiday_name,
        "holiday_mult":     round(holiday_mult, 4),
        "peak_mw_estimate": round(peak_mw, 0) if peak_mw is not None else None,
        "validated":        validated,
        "notes":            notes,
    }
    return result



def backtest(daily: pd.DataFrame, holdout_months: int = 12) -> dict:
    """Backtest day-of-week profiles on full-coverage holdout months."""
    df_all = daily.copy()
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.dropna(subset=["national_energy_met_mu"])

    # Exclude the COVID lockdown window from backtest scoring.
    df_all = df_all[~((df_all["date"] >= COVID_START) & (df_all["date"] <= COVID_END))].copy()

    df_all["_ym"] = df_all["date"].dt.to_period("M")
    df_all["_month_median"] = df_all.groupby("_ym")["national_energy_met_mu"].transform("median")
    df_all["_flagged"] = (
        (df_all["national_energy_met_mu"] / df_all["_month_median"] > OUTLIER_HI) |
        (df_all["national_energy_met_mu"] / df_all["_month_median"] < OUTLIER_LO)
    )

    cov = df_all[~df_all["_flagged"]].groupby("_ym")["date"].count()
    full_months = cov[cov >= MIN_DAYS_MONTH].index.sort_values()

    if len(full_months) <= 2 * holdout_months:
        return {"error": "Not enough full-coverage months for two-holdout backtest."}

    if _HOLIDAYS_AVAILABLE:
        holiday_cache: dict = {}
        def _is_holiday(d):
            yr = d.year
            if yr not in holiday_cache:
                holiday_cache[yr] = _get_india_holidays(yr)
            return d.date() in holiday_cache[yr]
        df_all["_is_holiday"] = df_all["date"].apply(lambda d: _is_holiday(d))
    else:
        df_all["_is_holiday"] = False

    df_all["dow"] = df_all["date"].dt.dayofweek

    variants = ["flat", "dow", "dow_shrunk", "dow_raw"]
    results_h1 = {}
    results_h2 = {}

    for holdout_idx, ho_results in enumerate([results_h1, results_h2]):
        if holdout_idx == 0:
            test_months  = full_months[-holdout_months:]
            train_months = full_months[:-holdout_months]
        else:
            test_months  = full_months[-(2 * holdout_months):-holdout_months]
            train_months = full_months[:-(2 * holdout_months)]

        if len(train_months) == 0:
            for v in variants:
                ho_results[v] = float("nan")
            continue

        train_df = df_all[df_all["_ym"].isin(train_months) & ~df_all["_flagged"]].copy()
        test_df  = df_all[df_all["_ym"].isin(test_months) & ~df_all["_flagged"]].copy()

        test_df["_clean_mean"] = test_df.groupby("_ym")["national_energy_met_mu"].transform("mean")

        train_clean_mean = train_df.groupby("_ym")["national_energy_met_mu"].transform("mean")
        train_df = train_df.copy()
        train_df["ratio_w"] = train_df["national_energy_met_mu"] / train_clean_mean
        p01 = train_df["ratio_w"].quantile(0.01)
        p99 = train_df["ratio_w"].quantile(0.99)
        train_df["ratio_w"] = train_df["ratio_w"].clip(p01, p99)

        hol_train = train_df[train_df["_is_holiday"]].copy()

        def _base_dow(df_in):
            g = df_in[~df_in["_is_holiday"]].groupby("dow")["ratio_w"]
            return g.mean().to_dict()

        base_dow = _base_dow(train_df)

        for v in variants:
            if v == "flat":
                pred = test_df["_clean_mean"].values
            elif v == "dow":
                test_df["_pf"] = test_df["dow"].map(base_dow).fillna(1.0)
                pred = (test_df["_clean_mean"] * test_df["_pf"]).values
            elif v in ("dow_shrunk", "dow_raw"):
                hol_name_mult = {}
                if _HOLIDAYS_AVAILABLE and len(hol_train) > 0:
                    name_cache2: dict = {}
                    def _hn(d):
                        yr = d.year
                        if yr not in name_cache2:
                            name_cache2[yr] = _get_india_holidays(yr)
                        return name_cache2[yr].get(d.date(), "Unknown")
                    hol_train = hol_train.copy()
                    hol_train["_hname"] = hol_train["date"].apply(lambda d: _hn(d))
                    hol_train["_dpm"] = hol_train["dow"].map(base_dow).fillna(1.0)
                    hol_train["_resid"] = hol_train["ratio_w"] / hol_train["_dpm"]
                    hs = hol_train.groupby("_hname")["_resid"].agg(["median", "count"])
                    for hname, row in hs.iterrows():
                        raw_m = row["median"]
                        n = row["count"]
                        if v == "dow_shrunk":
                            m = 1.0 + (raw_m - 1.0) * n / (n + 5.0) if n >= 5 else 1.0
                        else:  # dow_raw
                            m = raw_m if n >= 5 else 1.0
                        hol_name_mult[hname] = m

                def _test_mult(row):
                    if not row["_is_holiday"]:
                        return base_dow.get(row["dow"], 1.0)
                    hdate = row["date"]
                    hname_key = _get_india_holidays(hdate.year).get(hdate.date(), None)
                    hm = hol_name_mult.get(hname_key, 1.0) if hname_key else 1.0
                    return base_dow.get(row["dow"], 1.0) * hm

                test_df["_pf2"] = test_df.apply(_test_mult, axis=1)
                pred = (test_df["_clean_mean"] * test_df["_pf2"]).values
            else:
                pred = test_df["_clean_mean"].values

            actual = test_df["national_energy_met_mu"].values
            mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
            ho_results[v] = round(mape, 3)

    # Pick winner: lowest H1+H2 sum; tie-break: flat < dow < dow_shrunk < dow_raw
    tb_order = {"flat": 0, "dow": 1, "dow_shrunk": 2, "dow_raw": 3}
    best_v = min(
        variants,
        key=lambda v: (
            (results_h1.get(v, 999) + results_h2.get(v, 999)),
            tb_order[v]
        )
    )

    test_months  = full_months[-holdout_months:]
    train_months = full_months[:-holdout_months]
    train_df = df_all[df_all["_ym"].isin(train_months) & ~df_all["_flagged"]].copy()
    test_df  = df_all[df_all["_ym"].isin(test_months) & ~df_all["_flagged"]].copy()
    test_df["_clean_mean"] = test_df.groupby("_ym")["national_energy_met_mu"].transform("mean")
    train_clean_mean = train_df.groupby("_ym")["national_energy_met_mu"].transform("mean")
    train_df = train_df.copy()
    train_df["ratio_w"] = train_df["national_energy_met_mu"] / train_clean_mean
    p01 = train_df["ratio_w"].quantile(0.01)
    p99 = train_df["ratio_w"].quantile(0.99)
    train_df["ratio_w"] = train_df["ratio_w"].clip(p01, p99)
    base_dow_final = _base_dow(train_df)
    test_df["_pf_final"] = test_df["dow"].map(base_dow_final).fillna(1.0)
    test_df["estimate"]  = test_df["_clean_mean"] * test_df["_pf_final"]
    test_df["naive"]     = test_df["_clean_mean"]

    actual   = test_df["national_energy_met_mu"].values
    estimate = test_df["estimate"].values
    naive    = test_df["naive"].values

    def _q05(d):
        dow_key = int(d)
        p = train_df[train_df["dow"] == dow_key]["ratio_w"]
        return float(p.quantile(0.05)) if len(p) > 0 else 0.9
    def _q95(d):
        dow_key = int(d)
        p = train_df[train_df["dow"] == dow_key]["ratio_w"]
        return float(p.quantile(0.95)) if len(p) > 0 else 1.1

    test_df["_q05"] = test_df["dow"].apply(_q05)
    test_df["_q95"] = test_df["dow"].apply(_q95)
    low  = test_df["_clean_mean"].values * test_df["_q05"].values
    high = test_df["_clean_mean"].values * test_df["_q95"].values

    mae      = float(np.mean(np.abs(actual - estimate)))
    mape     = float(np.mean(np.abs((actual - estimate) / actual)) * 100)
    mape_naive = float(np.mean(np.abs((actual - naive) / actual)) * 100)
    coverage = float(np.mean((actual >= low) & (actual <= high)))

    variant_mape_table = {
        v: {"H1": results_h1.get(v), "H2": results_h2.get(v)}
        for v in variants
    }

    return {
        "n_days":               len(test_df),
        "holdout_months":       holdout_months,
        "MAE_MU":               round(mae, 2),
        "MAPE_pct":             round(mape, 2),
        "MAPE_naive_pct":       round(mape_naive, 2),
        "coverage_q05_q95":     round(coverage, 4),
        "winning_variant":      best_v,
        "variant_mape_table":   variant_mape_table,
        "note": (
            "Backtest isolates day-allocation error only. "
            "Monthly baseline = TRUE cleaned mean (not SARIMA forecast). "
            "Scores non-flagged days only. "
            "SARIMA forecast error adds additional uncertainty in live use."
        ),
    }
