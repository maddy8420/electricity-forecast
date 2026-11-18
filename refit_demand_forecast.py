"""
refit_demand_forecast.py  (Phase C)
-------------------------------------
Re-fits SARIMAX(1,1,2)×(0,1,1,12) on avg_daily_consumption exactly as
final_forecast.py does (same preprocessing and training window logic,
same order and options) but selects the optimizer setting that gives the
most stable result, and writes the forecast to a NEW file so the original
avg_daily_demand_forecast_2040.xlsx is not overwritten.

Output:
    avg_daily_demand_forecast_2040_refit.xlsx
    (same schema: index=date, columns=forecast/ci_lower_95/ci_upper_95/
     is_validated_horizon; monthly range 2024-10 onward; 24-month
     validated flag)

Usage:
    python refit_demand_forecast.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------------------------------------------------------
# Constants — same as final_forecast.py
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).resolve().parent
INPUT_FILE     = BASE_DIR / "india_electricity_features.xlsx"
OUTPUT_FILE    = BASE_DIR / "avg_daily_demand_forecast_2040_refit.xlsx"
SAVED_FILE     = BASE_DIR / "avg_daily_demand_forecast_2040.xlsx"

TARGET_COL            = "avg_daily_consumption"   # same as what final_forecast.py used
BEST_ORDER            = (1, 1, 2)
BEST_SEASONAL_ORDER   = (0, 1, 1, 12)
FORECAST_END          = "2040-12-01"
VALIDATED_HORIZON_MONTHS = 24

COMPARISON_DATES = [
    pd.Timestamp("2024-10-01"),
    pd.Timestamp("2025-10-01"),
    pd.Timestamp("2030-12-01"),
    pd.Timestamp("2035-12-01"),
    pd.Timestamp("2040-12-01"),
]

HOLDOUT_START = "2023-10"   # first month of holdout (12 months)
HOLDOUT_END   = "2024-09"   # last month of holdout


# ---------------------------------------------------------------------------
# Preprocessing — identical to final_forecast.py load_series()
# ---------------------------------------------------------------------------
def load_series() -> pd.Series:
    df = pd.read_excel(INPUT_FILE, sheet_name="features")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    series = df[TARGET_COL].dropna()
    series = series.asfreq("MS").interpolate(method="linear")
    return series


def fit_model(series: pd.Series, method: str = "lbfgs",
              enforce_stationarity: bool = False,
              enforce_invertibility: bool = False):
    model = SARIMAX(
        series,
        order=BEST_ORDER,
        seasonal_order=BEST_SEASONAL_ORDER,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    ).fit(disp=False, method=method)
    return model


def make_forecast(model, last_date: pd.Timestamp) -> pd.DataFrame:
    n_steps = (
        pd.Period(FORECAST_END, freq="M") - pd.Period(last_date, freq="M")
    ).n
    fc_res = model.get_forecast(steps=n_steps)
    fc_df  = fc_res.summary_frame(alpha=0.05)
    fc_df.index.name = "date"
    fc_df = fc_df.rename(columns={
        "mean":           "forecast",
        "mean_ci_lower":  "ci_lower_95",
        "mean_ci_upper":  "ci_upper_95",
    })[["forecast", "ci_lower_95", "ci_upper_95"]]

    validated_cutoff = last_date + pd.DateOffset(months=VALIDATED_HORIZON_MONTHS)
    fc_df["is_validated_horizon"] = fc_df.index <= validated_cutoff
    return fc_df


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------
def holdout_mape(series: pd.Series, method: str, enforce: bool) -> dict:
    """
    Fit on data before HOLDOUT_START, score on HOLDOUT_START..HOLDOUT_END.
    Returns dict with mape, forecast df for the holdout window.
    """
    train = series[series.index < HOLDOUT_START]
    test  = series[(series.index >= HOLDOUT_START) & (series.index <= HOLDOUT_END)]

    if len(train) == 0 or len(test) == 0:
        return {"mape": float("nan"), "forecast_df": None}

    model = fit_model(train, method=method,
                      enforce_stationarity=enforce,
                      enforce_invertibility=enforce)
    fc_df = make_forecast(model, train.index[-1])

    # Align with test
    common = fc_df.index.intersection(test.index)
    if len(common) == 0:
        return {"mape": float("nan"), "forecast_df": fc_df}

    pred   = fc_df.loc[common, "forecast"]
    actual = test.loc[common]
    mape   = float(np.mean(np.abs((actual.values - pred.values) / actual.values)) * 100)
    return {"mape": mape, "forecast_df": fc_df.loc[common]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("refit_demand_forecast.py")
    print("=" * 70)

    series = load_series()
    print(
        f"\nSeries: {TARGET_COL}  |  "
        f"{series.index.min().date()} to {series.index.max().date()}  |  "
        f"n={len(series)}\n"
    )

    # ---- Holdout evaluation: enforce=False vs enforce=True ----
    print("Holdout evaluation (train < 2023-10, test 2023-10..2024-09):")
    print(f"{'Setting':<35} {'Holdout MAPE':>14}")
    print("-" * 52)

    holdout_results = {}
    for enforce in [False, True]:
        for method in ["lbfgs", "powell"]:
            label = f"enforce={'True' if enforce else 'False'}, method={method}"
            res = holdout_mape(series, method=method, enforce=enforce)
            mape = res["mape"]
            holdout_results[label] = mape
            print(f"  {label:<33} {mape:>13.2f}%")

    # ---- First 12 months: enforce=False lbfgs vs enforce=True lbfgs ----
    print("\nFirst 12 holdout months — enforce=False (lbfgs) vs enforce=True (lbfgs):")
    res_ef  = holdout_mape(series, method="lbfgs", enforce=False)
    res_et  = holdout_mape(series, method="lbfgs", enforce=True)
    fc_ef   = res_ef["forecast_df"]
    fc_et   = res_et["forecast_df"]

    test_months = series[(series.index >= HOLDOUT_START) & (series.index <= HOLDOUT_END)]
    print(f"\n{'Month':<12} {'Actual':>10} {'Fcast(EF=F)':>12} {'Fcast(EF=T)':>12}")
    print("-" * 50)
    for m in test_months.index:
        actual_val = test_months[m]
        ef_val = float(fc_ef.loc[m, "forecast"]) if (fc_ef is not None and m in fc_ef.index) else float("nan")
        et_val = float(fc_et.loc[m, "forecast"]) if (fc_et is not None and m in fc_et.index) else float("nan")
        print(f"  {str(m.date()):<12} {actual_val:>10.1f} {ef_val:>12.1f} {et_val:>12.1f}")

    # Oct-2024 comparison
    oct24 = pd.Timestamp("2024-10-01")
    ef_oct = float(fc_ef.loc[oct24, "forecast"]) if (fc_ef is not None and oct24 in fc_ef.index) else float("nan")
    et_oct = float(fc_et.loc[oct24, "forecast"]) if (fc_et is not None and oct24 in fc_et.index) else float("nan")
    diff_pct = abs(ef_oct - et_oct) / max(abs(ef_oct), abs(et_oct)) * 100
    print(f"\nOct-2024: enforce=False -> {ef_oct:.2f}, enforce=True -> {et_oct:.2f}, "
          f"diff={diff_pct:.2f}% ({'> 3% UNSTABLE' if diff_pct > 3 else '<= 3% stable'})")

    # Pick better enforce setting based on holdout MAPE
    mape_ef = holdout_results.get("enforce=False, method=lbfgs", 999)
    mape_et = holdout_results.get("enforce=True, method=lbfgs", 999)
    if not np.isnan(mape_ef) and not np.isnan(mape_et):
        best_enforce = mape_ef <= mape_et
    else:
        best_enforce = False
    print(f"\nBest setting: enforce_stationarity=enforce_invertibility="
          f"{'True' if best_enforce else 'False'} "
          f"(lower holdout MAPE: {min(mape_ef, mape_et):.2f}%)")

    # ---- Fit on FULL data with best setting ----
    print("\n" + "=" * 70)
    print("Fitting on full data...")
    model_full = fit_model(
        series, method="lbfgs",
        enforce_stationarity=best_enforce,
        enforce_invertibility=best_enforce,
    )
    fc_full = make_forecast(model_full, series.index[-1])
    print(f"Forecast: {fc_full.index.min().date()} to {fc_full.index.max().date()}, "
          f"n={len(fc_full)}")

    # ---- Stability check on full fit ----
    print("\nFull-fit stability (Oct-2024 forecast across optimizers):")
    oct24_vals = {}
    for meth in ["lbfgs", "powell", "nm", "bfgs"]:
        try:
            m2 = fit_model(series, method=meth,
                           enforce_stationarity=best_enforce,
                           enforce_invertibility=best_enforce)
            fc2 = make_forecast(m2, series.index[-1])
            v = float(fc2.loc[oct24, "forecast"]) if oct24 in fc2.index else float("nan")
            oct24_vals[meth] = v
            print(f"  method={meth:<8} -> Oct-2024 = {v:.2f}")
        except Exception as e:
            print(f"  method={meth}: failed ({e})")

    vals_ok = [v for v in oct24_vals.values() if not np.isnan(v)]
    if len(vals_ok) > 1:
        spread = max(vals_ok) - min(vals_ok)
        spread_pct = spread / min(vals_ok) * 100
        print(f"  Spread: {spread:.2f} MU ({spread_pct:.2f}%)")

    # ---- Load saved forecast for comparison ----
    print("\n" + "=" * 70)
    print("Comparison: saved vs refit")
    print(f"{'Date':<14} {'Saved fcast':>12} {'Refit fcast':>13} {'Diff':>8} {'Diff%':>7}")
    print("-" * 60)
    try:
        saved_fc = pd.read_excel(SAVED_FILE, index_col=0)
        saved_fc.index = pd.to_datetime(saved_fc.index)
    except Exception as e:
        print(f"  Could not load saved forecast: {e}")
        saved_fc = None

    for cd in COMPARISON_DATES:
        s_val = (
            float(saved_fc.loc[cd, "forecast"])
            if (saved_fc is not None and cd in saved_fc.index)
            else float("nan")
        )
        r_val = (
            float(fc_full.loc[cd, "forecast"])
            if cd in fc_full.index
            else float("nan")
        )
        diff     = r_val - s_val if not (np.isnan(s_val) or np.isnan(r_val)) else float("nan")
        diff_pct2 = diff / s_val * 100 if not np.isnan(diff) and s_val != 0 else float("nan")
        s_str  = f"{s_val:,.2f}"  if not np.isnan(s_val)   else "N/A"
        r_str  = f"{r_val:,.2f}"  if not np.isnan(r_val)   else "N/A"
        d_str  = f"{diff:+,.2f}"  if not np.isnan(diff)     else "N/A"
        dp_str = f"{diff_pct2:+.2f}%" if not np.isnan(diff_pct2) else "N/A"
        print(f"  {str(cd.date()):<14} {s_str:>12} {r_str:>13} {d_str:>8} {dp_str:>7}")

    # ---- Write output ----
    print(f"\nWriting {OUTPUT_FILE} ...")
    fc_full.to_excel(OUTPUT_FILE)
    print("Done.")

    # Verify it doesn't overwrite the saved file
    assert OUTPUT_FILE.resolve() != SAVED_FILE.resolve(), (
        "BUG: would have overwritten the original forecast file!"
    )
    print(f"Verified: output is a NEW file ({OUTPUT_FILE.name}), "
          f"original ({SAVED_FILE.name}) unchanged.")

    print("\n" + "=" * 70)
    print("IMPORTANT: To use the refit forecast in the dashboard, update")
    print("  DEMAND_FORECAST_FILE in dashboard.py to point to:")
    print(f"  {OUTPUT_FILE.name}")
    print("Or copy avg_daily_demand_forecast_2040_refit.xlsx over the")
    print("original ONLY after verifying the numbers are acceptable.")
    print("=" * 70)


if __name__ == "__main__":
    main()
