"""
SARIMA Order Selection — Small Grid Search
------------------------------------------------
The (1,1,1)x(1,1,1,12) order used so far was a reasonable starting guess,
not a validated choice — and it threw convergence warnings. This does a
small grid search over plausible orders, scored by AIC (a standard
model-selection criterion that penalizes unnecessary complexity), and
reports the best one to actually use for the final forecast.

Kept deliberately small (candidates chosen from typical ranges for
monthly economic/demand series) rather than an exhaustive search, since
an exhaustive grid over a small dataset risks overfitting the order
selection itself.
"""

import warnings
warnings.filterwarnings("ignore")

import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

INPUT_FILE = "india_electricity_features.xlsx"
TARGET_COL = "avg_daily_demand"

# Candidate ranges — kept small and standard for monthly seasonal series
p_range = range(0, 3)
d_range = [1]              # we know from the trend chart this series needs differencing
q_range = range(0, 3)
P_range = range(0, 2)
D_range = [1]              # seasonal differencing — series has a clear yearly cycle
Q_range = range(0, 2)
SEASONAL_PERIOD = 12


def load_series():
    df = pd.read_excel(INPUT_FILE, sheet_name="features")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    series = df[TARGET_COL].dropna()
    series = series.asfreq("MS").interpolate(method="linear")
    return series


def grid_search(series):
    results = []
    combos = list(itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range))
    print(f"Testing {len(combos)} candidate orders...\n")

    for p, d, q, P, D, Q in combos:
        order = (p, d, q)
        seasonal_order = (P, D, Q, SEASONAL_PERIOD)
        try:
            model = SARIMAX(
                series, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)
            results.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "AIC": model.aic,
                "BIC": model.bic,
                "converged": model.mle_retvals.get("converged", "n/a") if hasattr(model, "mle_retvals") else "n/a",
            })
        except Exception:
            continue

    results_df = pd.DataFrame(results).sort_values("AIC")
    return results_df


def main():
    series = load_series()
    print(f"Series length: {len(series)} months\n")

    results_df = grid_search(series)

    print("=" * 70)
    print("TOP 10 CANDIDATE ORDERS (sorted by AIC, lower is better)")
    print("=" * 70)
    print(results_df.head(10).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest order: {best['order']}, seasonal_order: {best['seasonal_order']}")
    print(f"AIC: {best['AIC']:.2f}, BIC: {best['BIC']:.2f}")
    print("\nUse this order in the next script (final forecast generation) "
          "instead of the placeholder (1,1,1)x(1,1,1,12).")

    results_df.to_excel("sarima_order_search.xlsx", index=False)
    print("Saved: sarima_order_search.xlsx")


if __name__ == "__main__":
    main()