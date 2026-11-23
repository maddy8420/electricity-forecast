"""Refit the demand forecast ensemble. Run with: python refit_demand_forecast.py."""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE_DIR       = Path(__file__).resolve().parent
INPUT_FILE     = BASE_DIR / "india_electricity_features.xlsx"
OUTPUT_FILE    = BASE_DIR / "avg_daily_demand_forecast_2040_refit.xlsx"
SAVED_FILE     = BASE_DIR / "avg_daily_demand_forecast_2040.xlsx"

TARGET_COL            = "avg_daily_consumption"
BEST_ORDER            = (1, 1, 2)
BEST_SEASONAL_ORDER   = (0, 1, 1, 12)
FORECAST_END          = "2040-12-01"
VALIDATED_HORIZON_MONTHS = 24
ENSEMBLE_TOLERANCE    = 0.1   # pp: include configs within this of best MAPE

HOLDOUT_START = "2023-10"
HOLDOUT_END   = "2024-09"

COMPARISON_DATES = [
    pd.Timestamp("2024-10-01"),
    pd.Timestamp("2025-10-01"),
    pd.Timestamp("2030-12-01"),
    pd.Timestamp("2035-12-01"),
    pd.Timestamp("2040-12-01"),
]

ALL_CONFIGS = [
    {"enforce": False, "method": "lbfgs"},
    {"enforce": False, "method": "powell"},
    {"enforce": True,  "method": "lbfgs"},
    {"enforce": True,  "method": "powell"},
]


def load_series() -> pd.Series:
    df = pd.read_excel(INPUT_FILE, sheet_name="features")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    series = df[TARGET_COL].dropna()
    series = series.asfreq("MS").interpolate(method="linear")
    return series


def fit_model(series: pd.Series, method: str = "lbfgs",
              enforce: bool = False):
    return SARIMAX(
        series,
        order=BEST_ORDER,
        seasonal_order=BEST_SEASONAL_ORDER,
        enforce_stationarity=enforce,
        enforce_invertibility=enforce,
    ).fit(disp=False, method=method)


def make_forecast(model, last_date: pd.Timestamp) -> pd.DataFrame:
    n_steps = (
        pd.Period(FORECAST_END, freq="M") - pd.Period(last_date, freq="M")
    ).n
    fc_res = model.get_forecast(steps=n_steps)
    fc_df  = fc_res.summary_frame(alpha=0.05)
    fc_df.index.name = "date"
    fc_df = fc_df.rename(columns={
        "mean":          "forecast",
        "mean_ci_lower": "ci_lower_95",
        "mean_ci_upper": "ci_upper_95",
    })[["forecast", "ci_lower_95", "ci_upper_95"]]
    validated_cutoff = last_date + pd.DateOffset(months=VALIDATED_HORIZON_MONTHS)
    fc_df["is_validated_horizon"] = fc_df.index <= validated_cutoff
    return fc_df


def holdout_mape(series: pd.Series, method: str, enforce: bool) -> dict:
    train = series[series.index < HOLDOUT_START]
    test  = series[(series.index >= HOLDOUT_START) & (series.index <= HOLDOUT_END)]
    if len(train) == 0 or len(test) == 0:
        return {"mape": float("nan")}
    model = fit_model(train, method=method, enforce=enforce)
    fc_df = make_forecast(model, train.index[-1])
    common = fc_df.index.intersection(test.index)
    if len(common) == 0:
        return {"mape": float("nan")}
    pred   = fc_df.loc[common, "forecast"]
    actual = test.loc[common]
    mape   = float(np.mean(np.abs((actual.values - pred.values) / actual.values)) * 100)
    return {"mape": mape}


def main():
    print("=" * 70)
    print("refit_demand_forecast.py  [ensemble]")
    print("=" * 70)

    series = load_series()
    print(
        f"\nSeries: {TARGET_COL}  |  "
        f"{series.index.min().date()} to {series.index.max().date()}  |  "
        f"n={len(series)}\n"
    )

    print("Holdout evaluation (train < 2023-10, test 2023-10..2024-09):")
    print(f"  {'Config':<35} {'Holdout MAPE':>14}  {'In ensemble?':>13}")
    print("  " + "-" * 65)

    mape_map = {}
    for cfg in ALL_CONFIGS:
        label = f"enforce={cfg['enforce']}, method={cfg['method']}"
        res   = holdout_mape(series, method=cfg["method"], enforce=cfg["enforce"])
        mape_map[label] = res["mape"]

    best_mape = min(v for v in mape_map.values() if not np.isnan(v))

    # Select ensemble members by holdout MAPE within the configured tolerance.
    ensemble_configs = []
    for cfg in ALL_CONFIGS:
        label = f"enforce={cfg['enforce']}, method={cfg['method']}"
        mape  = mape_map[label]
        in_ens = (not np.isnan(mape)) and (mape - best_mape <= ENSEMBLE_TOLERANCE)
        if in_ens:
            ensemble_configs.append(cfg)
        flag = "YES" if in_ens else "no"
        mape_str = f"{mape:.3f}%" if not np.isnan(mape) else "N/A"
        print(f"  {label:<35} {mape_str:>14}  {flag:>13}")

    print(f"\n  Best holdout MAPE: {best_mape:.3f}%  |  Tolerance: {ENSEMBLE_TOLERANCE} pp")
    print(f"  Ensemble members ({len(ensemble_configs)}):")
    for cfg in ensemble_configs:
        print(f"    enforce={cfg['enforce']}, method={cfg['method']}")

    print("\n" + "=" * 70)
    print("Fitting ensemble members on full data...")

    member_forecasts = []
    for cfg in ensemble_configs:
        label = f"enforce={cfg['enforce']}, method={cfg['method']}"
        try:
            m = fit_model(series, method=cfg["method"], enforce=cfg["enforce"])
            fc = make_forecast(m, series.index[-1])
            member_forecasts.append({"label": label, "fc": fc})
            oct24 = pd.Timestamp("2024-10-01")
            v = float(fc.loc[oct24, "forecast"]) if oct24 in fc.index else float("nan")
            print(f"  {label}: Oct-2024 = {v:.2f}")
        except Exception as e:
            print(f"  {label}: FAILED ({e})")

    if not member_forecasts:
        print("ERROR: no ensemble members fitted. Aborting.")
        return

    all_idx = member_forecasts[0]["fc"].index
    for mf in member_forecasts[1:]:
        all_idx = all_idx.intersection(mf["fc"].index)

    fc_ensemble = pd.DataFrame(index=all_idx)
    fc_ensemble["forecast"]    = np.mean(
        [mf["fc"].loc[all_idx, "forecast"]    for mf in member_forecasts], axis=0)
    fc_ensemble["ci_lower_95"] = np.mean(
        [mf["fc"].loc[all_idx, "ci_lower_95"] for mf in member_forecasts], axis=0)
    fc_ensemble["ci_upper_95"] = np.mean(
        [mf["fc"].loc[all_idx, "ci_upper_95"] for mf in member_forecasts], axis=0)
    # is_validated_horizon from first member (all members have same cutoff)
    fc_ensemble["is_validated_horizon"] = member_forecasts[0]["fc"].loc[all_idx, "is_validated_horizon"]
    fc_ensemble.index.name = "date"

    print("\n" + "=" * 70)
    print("Per-member and ensemble forecasts:")
    check_dates = [
        pd.Timestamp("2024-10-01"),
        pd.Timestamp("2025-10-01"),
        pd.Timestamp("2030-12-01"),
        pd.Timestamp("2040-12-01"),
    ]

    col_w = 15
    hdr = f"  {'Date':<12}"
    for mf in member_forecasts:
        short = mf["label"].replace("enforce=", "enf=").replace(", method=", "/")
        hdr += f" {short[:col_w]:>{col_w}}"
    hdr += f" {'ENSEMBLE':>{col_w}} {'spread%':>8}"
    print(hdr)
    print("  " + "-" * (12 + col_w * (len(member_forecasts) + 1) + 10))

    for cd in check_dates:
        row = f"  {str(cd.date()):<12}"
        vals = []
        for mf in member_forecasts:
            v = float(mf["fc"].loc[cd, "forecast"]) if cd in mf["fc"].index else float("nan")
            vals.append(v)
            row += f" {v:>{col_w}.2f}"
        ens_v = float(fc_ensemble.loc[cd, "forecast"]) if cd in fc_ensemble.index else float("nan")
        row += f" {ens_v:>{col_w}.2f}"
        ok_vals = [v for v in vals if not np.isnan(v)]
        spread_pct = (max(ok_vals) - min(ok_vals)) / ens_v * 100 if ok_vals and ens_v else float("nan")
        row += f" {spread_pct:>7.2f}%"
        print(row)

    print("\n" + "=" * 70)
    print("Comparison: saved vs ensemble refit")
    print(f"  {'Date':<12} {'Saved':>10} {'Refit':>10} {'Diff':>8} {'Diff%':>7} {'CI-width saved':>15} {'CI-width refit':>15}")
    print("  " + "-" * 85)
    try:
        saved_fc = pd.read_excel(SAVED_FILE, index_col=0)
        saved_fc.index = pd.to_datetime(saved_fc.index)
    except Exception as e:
        print(f"  Could not load saved: {e}")
        saved_fc = None

    for cd in COMPARISON_DATES:
        s_v = float(saved_fc.loc[cd, "forecast"]) if (saved_fc is not None and cd in saved_fc.index) else float("nan")
        r_v = float(fc_ensemble.loc[cd, "forecast"]) if cd in fc_ensemble.index else float("nan")
        diff = r_v - s_v if not (np.isnan(s_v) or np.isnan(r_v)) else float("nan")
        dpct = diff / s_v * 100 if not np.isnan(diff) and s_v != 0 else float("nan")
        s_ci = (float(saved_fc.loc[cd, "ci_upper_95"]) - float(saved_fc.loc[cd, "ci_lower_95"])) \
               if (saved_fc is not None and cd in saved_fc.index) else float("nan")
        r_ci = float(fc_ensemble.loc[cd, "ci_upper_95"]) - float(fc_ensemble.loc[cd, "ci_lower_95"]) \
               if cd in fc_ensemble.index else float("nan")
        print(f"  {str(cd.date()):<12} {s_v:>10.2f} {r_v:>10.2f} {diff:>+8.2f} {dpct:>+6.1f}% "
              f"{s_ci:>15.2f} {r_ci:>15.2f}")

    print(f"\nWriting {OUTPUT_FILE} ...")
    fc_ensemble.to_excel(OUTPUT_FILE)
    assert OUTPUT_FILE.resolve() != SAVED_FILE.resolve(), \
        "BUG: would overwrite original!"
    print(f"Done. Rows={len(fc_ensemble)}, "
          f"first={fc_ensemble.index.min().date()}, "
          f"last={fc_ensemble.index.max().date()}")
    print(f"Original {SAVED_FILE.name} first-row forecast: "
          f"{float(saved_fc.iloc[0]['forecast']):.5f} (unchanged check)")


if __name__ == "__main__":
    main()
