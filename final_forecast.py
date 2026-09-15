"""
Final Long-Horizon Forecast — Demand to 2040
--------------------------------------------------
Uses the validated SARIMA(1,1,2)x(0,1,1,12) order (selected via AIC grid
search) to forecast avg_daily_demand from the end of the data through
Dec 2040, with 95% confidence intervals.

IMPORTANT FRAMING (see printed notes and chart): only horizons up to
~24 months were empirically validated via walk-forward testing. Anything
beyond that — which is most of this forecast, since the target year is
2040 — should be presented as a trend-based scenario projection, not a
precise prediction. The confidence intervals will visibly widen with
horizon; this is the model correctly expressing its own growing
uncertainty, not a bug.
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

TARGET_COL = "avg_daily_generation"
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


def fit_and_forecast(series):
    model = SARIMAX(
        series, order=BEST_ORDER, seasonal_order=BEST_SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)

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
            print(f"\n{year}: {row['forecast']:,.0f} avg daily demand (MU)")
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

    ax.set_title("India Avg Daily Electricity Demand — Forecast to 2040\n"
                  "(orange region = beyond validated 24-month horizon — trend scenario, not precise prediction)")
    ax.set_ylabel("Avg Daily Demand (MU)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fname = FIG_DIR / "demand_forecast_2040.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"\nSaved: {fname}")


def main():
    series = load_series()
    print(f"Fitting SARIMA{BEST_ORDER}x{BEST_SEASONAL_ORDER} on {len(series)} months "
          f"of history ({series.index.min().date()} to {series.index.max().date()})...\n")

    forecast_df, model = fit_and_forecast(series)

    print_checkpoints(forecast_df)
    plot_forecast(series, forecast_df)

    forecast_df.to_excel("demand_forecast_2040.xlsx")
    print("\nSaved: demand_forecast_2040.xlsx")

    n_extrapolated = int((~forecast_df["is_validated_horizon"]).sum())
    n_total = len(forecast_df)
    print(f"\nIMPORTANT: {n_extrapolated} of {n_total} forecasted months "
          f"({n_extrapolated/n_total*100:.0f}%) fall beyond the empirically "
          f"validated 24-month horizon. Present these as trend-based scenarios "
          f"in your report, not as precise predictions — the widening confidence "
          f"intervals in the chart are the honest reflection of this.")


if __name__ == "__main__":
    main()