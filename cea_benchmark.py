"""
CEA Benchmark Comparison
----------------------------
Compares this project's SARIMA trend-based demand forecast against India's
own official demand projections, published by the Central Electricity
Authority (CEA) via the 20th Electric Power Survey (EPS) and subsequent
planning documents.

Official figures (energy requirement in Billion Units, BU) are hardcoded
below with their source citations — these are published government figures,
not scraped/API data, so they're recorded directly rather than fetched.

Sources:
  - CEA National Electricity Plan (Vol-I Generation), 2022-32, per 20th EPS
    projections. Notified via PIB press release, PRID=1928750.
  - CEA Long-Term National Resource Adequacy Plan (LT-NRAP) 2026-27 to
    2035-36, released ~April 2026, reflecting the mid-term review of the
    20th EPS (revised upward from the original NEP 2022-32 figures).
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DEMAND_FORECAST_FILE = "avg_daily_demand_forecast_2040.xlsx"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Official CEA projections — energy requirement in Billion Units (BU).
# 1 BU = 1,000 MU (million units), matching this project's avg_daily_* unit (MU).
CEA_BENCHMARKS = [
    {
        "label": "2026-27",
        "compare_date": "2026-12-01",
        "cea_energy_bu": 1907.8,
        "cea_peak_demand_gw": 277.2,
        "source": "CEA National Electricity Plan 2022-32 (20th EPS)",
    },
    {
        "label": "2031-32",
        "compare_date": "2032-12-01",
        "cea_energy_bu": 2473.8,
        "cea_peak_demand_gw": 366.4,
        "source": "CEA National Electricity Plan 2022-32 (20th EPS)",
    },
    {
        "label": "2035-36",
        "compare_date": "2035-12-01",
        "cea_energy_bu": 3365.0,
        "cea_peak_demand_gw": 458.7,
        "source": "CEA Long-Term National Resource Adequacy Plan (2026), 20th EPS mid-term review",
    },
]

# CEA's own stated CAGR for total energy requirement, 2024-25 to 2035-36,
# per the LT-NRAP mid-term review — cited directly, not recomputed.
CEA_STATED_CAGR_PCT = 6.41


def load_my_forecast():
    df = pd.read_excel(DEMAND_FORECAST_FILE, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def main():
    my_forecast = load_my_forecast()

    print("=" * 78)
    print("CEA BENCHMARK COMPARISON — My SARIMA Forecast vs Official CEA Projections")
    print("=" * 78)
    print(f"{'Year':<10}{'My Forecast (BU)':<20}{'CEA Official (BU)':<20}{'Gap':<12}{'Source'}")
    print("-" * 100)

    rows = []
    for b in CEA_BENCHMARKS:
        target_date = pd.Timestamp(b["compare_date"])
        if target_date not in my_forecast.index:
            print(f"{b['label']}: date {target_date.date()} outside forecast range, skipping")
            continue

        my_avg_daily = my_forecast.loc[target_date, "forecast"]
        my_annual_bu = my_avg_daily * 365 / 1000  # MU/day -> BU/year

        gap_pct = (my_annual_bu - b["cea_energy_bu"]) / b["cea_energy_bu"] * 100

        rows.append({
            "year": b["label"],
            "my_forecast_bu": my_annual_bu,
            "cea_official_bu": b["cea_energy_bu"],
            "gap_pct": gap_pct,
            "source": b["source"],
        })

        print(f"{b['label']:<10}{my_annual_bu:<20.1f}{b['cea_energy_bu']:<20.1f}"
              f"{gap_pct:>+6.1f}%     {b['source']}")

    results_df = pd.DataFrame(rows)

    # Implied CAGR from my model vs CEA's own stated CAGR
    first_val = my_forecast.loc["2026-12-01", "forecast"]
    last_val = my_forecast.loc["2035-12-01", "forecast"]
    years = 2035 - 2026
    my_cagr = ((last_val / first_val) ** (1 / years) - 1) * 100

    print()
    print("=" * 78)
    print("IMPLIED GROWTH RATE COMPARISON")
    print("=" * 78)
    print(f"My SARIMA model implied CAGR (2026-2035):  {my_cagr:.2f}%")
    print(f"CEA's own stated CAGR (2024-25 to 2035-36): {CEA_STATED_CAGR_PCT:.2f}%")
    print(f"\nMy trend-based model's growth assumption is roughly "
          f"{CEA_STATED_CAGR_PCT / my_cagr:.1f}x LOWER than CEA's own official planning assumption.")
    print("\nInterpretation: a pure historical-trend SARIMA model cannot capture")
    print("forward-looking structural demand drivers that CEA explicitly builds")
    print("into its own projections — EV adoption targets, green hydrogen")
    print("production goals, and continued electrification (Saubhagya scheme,")
    print("rooftop solar mandates). This suggests the demand-supply SURPLUS found")
    print("elsewhere in this project may be optimistic if India's electrification")
    print("accelerates as officially planned — the true demand trajectory may sit")
    print("meaningfully above this project's pure-trend extrapolation.")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    years_list = [2026, 2032, 2035]
    my_vals = results_df["my_forecast_bu"].tolist()
    cea_vals = results_df["cea_official_bu"].tolist()

    ax.plot(years_list, my_vals, marker="o", label="This project (SARIMA trend forecast)", color="tab:blue")
    ax.plot(years_list, cea_vals, marker="s", label="CEA official projection (20th EPS)", color="tab:red")
    ax.fill_between(years_list, my_vals, cea_vals, alpha=0.15, color="gray")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Energy Requirement (Billion Units)")
    ax.set_title("Demand Forecast: This Project vs CEA's Official Projection")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = FIG_DIR / "cea_benchmark_comparison.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"\nSaved: {fname}")

    results_df.to_excel("cea_benchmark_comparison.xlsx", index=False)
    print("Saved: cea_benchmark_comparison.xlsx")


if __name__ == "__main__":
    main()
