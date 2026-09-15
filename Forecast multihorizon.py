"""
Multi-Horizon Walk-Forward Validation
------------------------------------------
The single-step (1-month-ahead) comparison in forecast_models.py doesn't
tell us which model will actually hold up for a 2026-2040 forecast — some
models (Naive) are only ever accurate 1 step ahead and become useless
further out. This script tests each model at HORIZONS = [1, 3, 6, 12, 24]
months ahead, which is what actually justifies a model choice for
long-range forecasting.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

INPUT_FILE = "india_electricity_features.xlsx"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

TARGET_COL = "avg_daily_demand"
SEASONAL_PERIOD = 12
HORIZONS = [1, 3, 6, 12, 24]
N_TEST_ORIGINS = 12   # number of different starting points to test each horizon from
MIN_TRAIN_SIZE = 48


def load_series():
    df = pd.read_excel(INPUT_FILE, sheet_name="features")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    series = df[TARGET_COL].dropna()
    # Explicitly set monthly-start frequency — fixes the ValueWarning and
    # is required for reliable out-of-sample date-based forecasting later.
    series = series.asfreq("MS")
    series = series.interpolate(method="linear")  # in case asfreq introduced any new gaps
    return series


def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))


def mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))


# ----------------------------------------------------------------------
# FORECASTERS — each takes a training series + horizon, returns an array
# of `horizon` forecasted values
# ----------------------------------------------------------------------

def forecast_naive(train, horizon):
    return np.repeat(train.iloc[-1], horizon)


def forecast_seasonal_naive(train, horizon):
    vals = []
    for h in range(1, horizon + 1):
        idx = -SEASONAL_PERIOD + ((h - 1) % SEASONAL_PERIOD)
        if len(train) >= abs(idx):
            vals.append(train.iloc[idx])
        else:
            vals.append(train.iloc[-1])
    return np.array(vals)


def forecast_arima(train, horizon, order=(1, 1, 1)):
    model = ARIMA(train, order=order).fit()
    return model.forecast(horizon).values


def forecast_sarima(train, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(
        train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    return model.forecast(horizon).values


def forecast_exp_smoothing(train, horizon):
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIOD
    ).fit()
    return model.forecast(horizon).values


MODELS = {
    "Naive": forecast_naive,
    "Seasonal Naive": forecast_seasonal_naive,
    "Exponential Smoothing": forecast_exp_smoothing,
    "ARIMA": forecast_arima,
    "SARIMA": forecast_sarima,
}


def evaluate_at_horizon(series, horizon):
    """
    For a given horizon, pick N_TEST_ORIGINS different points in the
    series, forecast `horizon` steps ahead from each, and compare to what
    actually happened. Returns MAE/RMSE/MAPE per model, averaged across
    origins (each origin contributes its horizon-step-ahead error, not
    every intermediate step, to keep this focused on "how good is the
    forecast AT that horizon", not an average across 1..horizon).
    """
    n = len(series)
    latest_possible_origin = n - horizon
    earliest_origin = MIN_TRAIN_SIZE
    if latest_possible_origin <= earliest_origin:
        return None

    origins = np.linspace(earliest_origin, latest_possible_origin - 1, N_TEST_ORIGINS).astype(int)
    origins = sorted(set(origins))

    results = {name: {"actual": [], "predicted": []} for name in MODELS}

    for origin in origins:
        train = series.iloc[:origin]
        actual = series.iloc[origin + horizon - 1]  # the value exactly `horizon` steps ahead

        for name, fn in MODELS.items():
            try:
                preds = fn(train, horizon)
                pred_at_horizon = preds[-1]
                results[name]["actual"].append(actual)
                results[name]["predicted"].append(pred_at_horizon)
            except Exception:
                pass

    rows = []
    for name, data in results.items():
        if not data["actual"]:
            continue
        rows.append({
            "Horizon (months)": horizon,
            "Model": name,
            "MAE": mae(data["actual"], data["predicted"]),
            "RMSE": rmse(data["actual"], data["predicted"]),
            "MAPE (%)": mape(data["actual"], data["predicted"]),
        })
    return rows


def main():
    series = load_series()
    print(f"Target: {TARGET_COL}")
    print(f"Series length: {len(series)} months ({series.index.min().date()} to {series.index.max().date()})")
    print(f"Testing horizons: {HORIZONS} months ahead\n")

    all_rows = []
    for h in HORIZONS:
        print(f"Evaluating horizon = {h} month(s)...")
        rows = evaluate_at_horizon(series, h)
        if rows:
            all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)

    print("\n" + "=" * 70)
    print("MULTI-HORIZON RESULTS (MAPE %, lower is better)")
    print("=" * 70)
    pivot = results_df.pivot(index="Model", columns="Horizon (months)", values="MAPE (%)")
    pivot = pivot[HORIZONS]  # keep column order
    print(pivot.round(2).to_string())

    # Plot MAPE vs horizon per model — this is the chart that actually
    # answers "which model should I trust for a 15-year-ahead forecast"
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in pivot.index:
        ax.plot(HORIZONS, pivot.loc[model], marker="o", label=model)
    ax.set_xlabel("Forecast horizon (months ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Forecast accuracy degradation by horizon")
    ax.legend()
    plt.tight_layout()
    fname = FIG_DIR / "mape_by_horizon.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"\nSaved: {fname}")

    results_df.to_excel("multi_horizon_results.xlsx", index=False)
    print("Saved: multi_horizon_results.xlsx")

    print("\nLook at the 12 and 24-month columns specifically — that's the "
          "regime your actual 2026-2040 forecast will live in, not the "
          "1-month column from the earlier comparison.")


if __name__ == "__main__":
    main()