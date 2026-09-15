"""
Forecasting Model Comparison — Walk-Forward Validation
----------------------------------------------------------
Evaluates multiple forecasting approaches on a single target series using
walk-forward (expanding window, 1-month-ahead) validation, which is far
more honest than a single train/test split — it tests the model across
many different points in time, not just one.

Models compared:
  - Naive (last value)
  - Seasonal Naive (same month, prior year)
  - Moving Average (3-month)
  - Holt-Winters Exponential Smoothing (additive seasonal)
  - ARIMA
  - SARIMA
  - Prophet (skipped gracefully if not installed)

Metrics: MAE, RMSE, MAPE — aggregated across all walk-forward folds.

Change TARGET_COL below to run this against generation or consumption
instead of demand.
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

TARGET_COL = "avg_daily_demand"   # change to avg_daily_generation / avg_daily_consumption
SEASONAL_PERIOD = 12
TEST_MONTHS = 24                   # size of the walk-forward evaluation window
MIN_TRAIN_SIZE = 36                # don't start forecasting until we have 3 years of history


def load_series():
    df = pd.read_excel(INPUT_FILE, sheet_name="features")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    series = df[TARGET_COL].dropna()
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
# FORECASTERS — each takes a training series, returns a 1-step forecast
# ----------------------------------------------------------------------

def forecast_naive(train):
    return train.iloc[-1]


def forecast_seasonal_naive(train):
    if len(train) >= SEASONAL_PERIOD:
        return train.iloc[-SEASONAL_PERIOD]
    return train.iloc[-1]


def forecast_moving_average(train, window=3):
    return train.iloc[-window:].mean()


def forecast_exp_smoothing(train):
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIOD
    ).fit()
    return model.forecast(1).iloc[0]


def forecast_arima(train, order=(1, 1, 1)):
    model = ARIMA(train, order=order).fit()
    return model.forecast(1).iloc[0]


def forecast_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(
        train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    return model.forecast(1).iloc[0]


def forecast_prophet(train):
    try:
        from prophet import Prophet
    except ImportError:
        return None
    df = pd.DataFrame({"ds": train.index, "y": train.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=1, freq="MS")
    forecast = model.predict(future)
    return forecast["yhat"].iloc[-1]


MODELS = {
    "Naive": forecast_naive,
    "Seasonal Naive": forecast_seasonal_naive,
    "Moving Average (3mo)": forecast_moving_average,
    "Exponential Smoothing": forecast_exp_smoothing,
    "ARIMA": forecast_arima,
    "SARIMA": forecast_sarima,
    "Prophet": forecast_prophet,
}


def walk_forward_validate(series):
    n = len(series)
    if n < MIN_TRAIN_SIZE + TEST_MONTHS:
        raise ValueError(
            f"Not enough data: need at least {MIN_TRAIN_SIZE + TEST_MONTHS} months, "
            f"have {n}."
        )

    test_start_idx = n - TEST_MONTHS
    results = {name: {"actual": [], "predicted": []} for name in MODELS}
    prophet_available = True

    print(f"Walk-forward validation: {TEST_MONTHS} folds, expanding window "
          f"(starting with {test_start_idx} months of training history)\n")

    for i in range(test_start_idx, n):
        train = series.iloc[:i]
        actual = series.iloc[i]
        fold_date = series.index[i]

        for name, fn in MODELS.items():
            if name == "Prophet" and not prophet_available:
                continue
            try:
                pred = fn(train)
                if pred is None:
                    prophet_available = False
                    print("  Prophet not installed — skipping "
                          "(pip install prophet if you want it included).")
                    continue
                results[name]["actual"].append(actual)
                results[name]["predicted"].append(pred)
            except Exception as e:
                print(f"  [{name}] failed at {fold_date.date()}: {e}")

        if (i - test_start_idx + 1) % 6 == 0:
            print(f"  ...completed fold {i - test_start_idx + 1}/{TEST_MONTHS}")

    return results


def summarize_results(results):
    rows = []
    for name, data in results.items():
        if not data["actual"]:
            continue
        rows.append({
            "Model": name,
            "MAE": mae(data["actual"], data["predicted"]),
            "RMSE": rmse(data["actual"], data["predicted"]),
            "MAPE (%)": mape(data["actual"], data["predicted"]),
            "Folds evaluated": len(data["actual"]),
        })
    summary = pd.DataFrame(rows).sort_values("RMSE")
    return summary


def plot_best_vs_actual(series, results, best_model_name):
    data = results[best_model_name]
    test_dates = series.index[-len(data["actual"]):]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values, label="Full series (actual)", alpha=0.4)
    ax.plot(test_dates, data["actual"], label="Actual (test period)", marker="o")
    ax.plot(test_dates, data["predicted"], label=f"{best_model_name} (predicted)", marker="x")
    ax.set_title(f"Best model ({best_model_name}) — walk-forward predictions vs actual")
    ax.legend()
    plt.tight_layout()
    fname = FIG_DIR / "best_model_vs_actual.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"\nSaved: {fname}")


def main():
    series = load_series()
    print(f"Target: {TARGET_COL}")
    print(f"Series length: {len(series)} months ({series.index.min().date()} to {series.index.max().date()})\n")

    results = walk_forward_validate(series)
    summary = summarize_results(results)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (sorted by RMSE, lower is better)")
    print("=" * 70)
    print(summary.to_string(index=False))

    best_model = summary.iloc[0]["Model"]
    print(f"\nBest performing model: {best_model}")
    plot_best_vs_actual(series, results, best_model)

    summary.to_excel("model_comparison_results.xlsx", index=False)
    print("Saved: model_comparison_results.xlsx")


if __name__ == "__main__":
    main()