"""
Final Long-Horizon Forecast — to 2040
--------------------------------------------------
Uses the validated SARIMA(1,1,2)x(0,1,1,12) order (selected via AIC grid
search) to forecast the chosen TARGET_COL from the end of the data through
Dec 2040, with 95% confidence intervals.

IMPORTANT FRAMING (see printed notes and chart): only horizons up to
~24 months were empirically validated via walk-forward testing. Anything
beyond that — which is most of this forecast, since the target year is
2040 — should be presented as a trend-based scenario projection, not a
precise prediction. The confidence intervals will visibly widen with
horizon; this is the model correctly expressing its own growing
uncertainty, not a bug.

Change TARGET_COL below to run this against demand, generation, or
consumption — filenames and chart titles adapt automatically.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

INPUT_FILE = "india_electricity_features.xlsx"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

TARGET_COL = "avg_daily_generation"   # change to avg_daily_demand / avg_daily_consumption
BEST_ORDER = (1, 1, 2)
BEST_SEASONAL_ORDER = (0, 1, 1, 12)
FORECAST_END = "2040-12-01"
VALIDATED_HORIZON_MONTHS = 24  # from the multi-horizon walk-forward test
CHECKPOINT_YEARS = [2026, 2030, 2035, 2040]


def load_series():
    df = pd.read_excel(INPUT_FILE, sheet_name="features")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    series = df[TARGET_COL].dropna()
    series = series.asfreq("MS").interpolate(method="linear")
    return series


def fit_and_forecast(series, method="lbfgs"):
    model = SARIMAX(
        series, order=BEST_ORDER, seasonal_order=BEST_SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False, method=method)

    last_date = series.index[-1]
    n_steps = (pd.Period(FORECAST_END, freq="M") - pd.Period(last_date, freq="M")).n

    forecast_result = model.get_forecast(steps=n_steps)
    forecast_df = forecast_result.summary_frame(alpha=0.05)  # 95% CI
    forecast_df.index.name = "date"
    forecast_df = forecast_df.rename(columns={
        "mean": "forecast",
        "mean_ci_lower": "ci_lower_95",
        "mean_ci_upper": "ci_upper_95",
    })[["forecast", "ci_lower_95", "ci_upper_95"]]

    validated_cutoff = last_date + pd.DateOffset(months=VALIDATED_HORIZON_MONTHS)
    forecast_df["is_validated_horizon"] = forecast_df.index <= validated_cutoff

    return forecast_df, model


def print_checkpoints(forecast_df):
    print("=" * 70)
    print("CHECKPOINT FORECASTS (December of each target year)")
    print("=" * 70)
    for year in CHECKPOINT_YEARS:
        target_date = pd.Timestamp(f"{year}-12-01")
        if target_date in forecast_df.index:
            row = forecast_df.loc[target_date]
            validated = "validated horizon" if row["is_validated_horizon"] else "SCENARIO EXTRAPOLATION"
            print(f"\n{year}: {row['forecast']:,.0f} {TARGET_COL} (MU)")
            print(f"  95% CI: [{row['ci_lower_95']:,.0f}, {row['ci_upper_95']:,.0f}]")
            print(f"  ({validated} — {'trust this' if row['is_validated_horizon'] else 'directional trend only, not a precise prediction'})")
        else:
            print(f"\n{year}: outside forecast range")


def plot_forecast(series, forecast_df):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(series.index, series.values, label="Historical (actual)", color="tab:blue")

    validated = forecast_df[forecast_df["is_validated_horizon"]]
    extrapolated = forecast_df[~forecast_df["is_validated_horizon"]]

    ax.plot(validated.index, validated["forecast"], color="tab:green",
            label="Forecast (validated horizon, ≤24mo)")
    ax.fill_between(validated.index, validated["ci_lower_95"], validated["ci_upper_95"],
                     color="tab:green", alpha=0.2)

    ax.plot(extrapolated.index, extrapolated["forecast"], color="tab:orange",
            linestyle="--", label="Forecast (scenario extrapolation, >24mo)")
    ax.fill_between(extrapolated.index, extrapolated["ci_lower_95"], extrapolated["ci_upper_95"],
                     color="tab:orange", alpha=0.15)

    for year in CHECKPOINT_YEARS:
        target_date = pd.Timestamp(f"{year}-12-01")
        if target_date in forecast_df.index:
            ax.axvline(target_date, color="gray", linestyle=":", alpha=0.5)

    label = TARGET_COL.replace("avg_daily_", "").replace("_", " ").title()
    ax.set_title(f"India Avg Daily {label} — Forecast to 2040\n"
                  "(orange region = beyond validated 24-month horizon — trend scenario, not precise prediction)")
    ax.set_ylabel(f"Avg Daily {label} (MU)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fname = FIG_DIR / f"{TARGET_COL}_forecast_2040.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"\nSaved: {fname}")


def main():
    series = load_series()
    print(f"Fitting SARIMA{BEST_ORDER}x{BEST_SEASONAL_ORDER} on {len(series)} months "
          f"of history ({series.index.min().date()} to {series.index.max().date()})...\n")

    # Stability check: fit with a couple of different optimizers and compare
    # the 2040 point forecast. If these disagree substantially, it confirms
    # the fit is sensitive to optimization path — important to know and
    # disclose before treating any single run's numbers as final.
    print("Stability check — fitting with multiple optimizers to check "
          "how sensitive the result is to the optimization path...\n")
    stability_results = {}
    for method in ["lbfgs", "powell", "nm"]:
        try:
            fc, _ = fit_and_forecast(series, method=method)
            val_2040 = fc.loc["2040-12-01", "forecast"] if pd.Timestamp("2040-12-01") in fc.index else None
            stability_results[method] = val_2040
            print(f"  method={method:10s} -> 2040 forecast: {val_2040:,.0f}" if val_2040 else f"  method={method}: out of range")
        except Exception as e:
            print(f"  method={method}: failed ({e})")

    if len(stability_results) > 1:
        vals = [v for v in stability_results.values() if v is not None]
        spread = max(vals) - min(vals)
        spread_pct = spread / min(vals) * 100
        print(f"\n  Spread across optimizers: {spread:,.0f} MU ({spread_pct:.1f}% of the lowest estimate)")
        if spread_pct > 5:
            print("  This is a meaningful spread — the fit is sensitive to optimization "
                  "path. Report the RANGE across methods as an additional uncertainty "
                  "signal, not just the confidence interval from a single fit.")
        else:
            print("  Spread is small — the fit looks reasonably stable across optimizers.")

    # Use lbfgs (statsmodels default) as the primary reported result
    forecast_df, model = fit_and_forecast(series, method="lbfgs")

    print_checkpoints(forecast_df)
    plot_forecast(series, forecast_df)

    forecast_df.to_excel(f"{TARGET_COL}_forecast_2040.xlsx")
    print(f"\nSaved: {TARGET_COL}_forecast_2040.xlsx")

    n_extrapolated = int((~forecast_df["is_validated_horizon"]).sum())
    n_total = len(forecast_df)
    print(f"\nIMPORTANT: {n_extrapolated} of {n_total} forecasted months "
          f"({n_extrapolated/n_total*100:.0f}%) fall beyond the empirically "
          f"validated 24-month horizon. Present these as trend-based scenarios "
          f"in your report, not as precise predictions — the widening confidence "
          f"intervals in the chart are the honest reflection of this.")


if __name__ == "__main__":
    main()