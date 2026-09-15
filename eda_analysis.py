"""
India Electricity EDA
------------------------
Reads india_electricity_master.xlsx and produces:
  - Summary stats / missing-value check
  - Trend & seasonality decomposition (generation + demand)
  - Year-over-year growth rates
  - A reporting-era boundary check (2017-05-26) so real growth isn't
    confused with a data-collection artifact
  - Source-mix evolution chart (2017-2023 only, per known data limitation)
  - Demand vs temperature correlation
  - Peak vs average demand gap over time

All charts saved as PNGs in ./figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path

INPUT_FILE = "india_electricity_master.xlsx"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

ERA_CUTOFF = pd.Timestamp("2017-05-26")


def load_sheets():
    master = pd.read_excel(INPUT_FILE, sheet_name="MASTER_monthly")
    master["year_month"] = pd.PeriodIndex(master["year_month"], freq="M")
    master["date"] = master["year_month"].dt.to_timestamp()

    harmonized = pd.read_excel(INPUT_FILE, sheet_name="gen_by_source_harmonized")
    harmonized["date"] = pd.to_datetime(harmonized["date"])

    hourly = pd.read_excel(INPUT_FILE, sheet_name="hourly_agg_daily")
    hourly["date"] = pd.to_datetime(hourly["date"])

    return master, harmonized, hourly


def sanity_check(master):
    print("=" * 70)
    print("1. SANITY CHECK — MASTER_monthly")
    print("=" * 70)
    print(f"Shape: {master.shape}")
    print(f"Date range: {master['date'].min().date()} to {master['date'].max().date()}")
    print("\nMissing values per column:")
    print(master.isna().sum())
    print("\nDescribe (numeric columns):")
    print(master.describe().T)
    print()


def trend_seasonality(master):
    print("=" * 70)
    print("2. TREND & SEASONALITY DECOMPOSITION")
    print("=" * 70)

    for col, label in [
        ("national_total_generation_mu", "Generation"),
        ("total_energy_met_mu", "Demand (Energy Met)"),
    ]:
        series = master.set_index("date")[col].dropna()
        if len(series) < 24:
            print(f"  Skipping {label}: not enough data points for decomposition.")
            continue

        result = seasonal_decompose(series, model="additive", period=12)

        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        result.observed.plot(ax=axes[0], title=f"{label} — Observed")
        result.trend.plot(ax=axes[1], title="Trend")
        result.seasonal.plot(ax=axes[2], title="Seasonal")
        result.resid.plot(ax=axes[3], title="Residual")
        for ax in axes:
            ax.axvline(ERA_CUTOFF, color="red", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fname = FIG_DIR / f"decomposition_{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  Saved: {fname}")
    print()


def yoy_growth(master):
    print("=" * 70)
    print("3. YEAR-OVER-YEAR GROWTH")
    print("=" * 70)

    df = master.set_index("date").sort_index()
    for col, label in [
        ("national_total_generation_mu", "Generation"),
        ("total_energy_met_mu", "Demand"),
        ("total_consumption_mu", "Consumption"),
    ]:
        if col not in df.columns:
            continue
        yoy = df[col].pct_change(periods=12) * 100
        yearly = yoy.resample("YE").mean()
        print(f"\n{label} — average YoY growth % by year:")
        print(yearly.round(2).to_string())

        fig, ax = plt.subplots(figsize=(10, 4))
        yoy.plot(ax=ax, title=f"{label} — YoY growth (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(ERA_CUTOFF, color="red", linestyle="--", alpha=0.5, label="reporting era change")
        ax.legend()
        plt.tight_layout()
        fname = FIG_DIR / f"yoy_{label.lower()}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
    print()


def source_mix_evolution(harmonized):
    print("=" * 70)
    print("4. SOURCE-MIX EVOLUTION (2017-2023 only — see data_era limitation)")
    print("=" * 70)

    detailed = harmonized[harmonized["data_era"] == "2017-2023 (detailed reporting)"].copy()
    detailed["year_month"] = detailed["date"].dt.to_period("M")

    monthly = detailed.groupby("year_month")[
        ["coal_lignite_mu", "renewables_mu", "hydro_mu", "nuclear_mu", "gas_naptha_diesel_mu"]
    ].sum()
    monthly.index = monthly.index.to_timestamp()

    # Convert to % share for a stacked area chart
    share = monthly.div(monthly.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(share.index, share.T, labels=share.columns)
    ax.set_title("Generation source mix — % share (2017-2023)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("% share")
    plt.tight_layout()
    fname = FIG_DIR / "source_mix_evolution.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")

    print("\nRenewables % share, first vs last year available:")
    print(f"  {share.index[0].strftime('%Y-%m')}: {share['renewables_mu'].iloc[0]:.1f}%")
    print(f"  {share.index[-1].strftime('%Y-%m')}: {share['renewables_mu'].iloc[-1]:.1f}%")
    print()


def demand_temperature_correlation(master):
    print("=" * 70)
    print("5. DEMAND vs TEMPERATURE CORRELATION")
    print("=" * 70)

    df = master.dropna(subset=["Monthly_load", "max_temp"])
    if len(df) < 3:
        print("  Not enough overlapping data to compute correlation.")
        return

    corr = df["Monthly_load"].corr(df["max_temp"])
    print(f"  Correlation (monthly load vs max temp): {corr:.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["max_temp"], df["Monthly_load"], alpha=0.6)
    ax.set_xlabel("Max Temperature (°C)")
    ax.set_ylabel("Monthly Load")
    ax.set_title(f"Demand vs Temperature (r = {corr:.2f})")
    plt.tight_layout()
    fname = FIG_DIR / "demand_vs_temperature.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")
    print()


def peak_vs_average_gap(hourly):
    print("=" * 70)
    print("6. PEAK vs AVERAGE DEMAND GAP")
    print("=" * 70)

    hourly = hourly.copy()
    hourly["gap_pct"] = (
        (hourly["hourly_demand_daily_max"] - hourly["hourly_demand_daily_mean"])
        / hourly["hourly_demand_daily_mean"] * 100
    )
    hourly["year"] = hourly["date"].dt.year
    yearly_gap = hourly.groupby("year")["gap_pct"].mean()
    print("Average daily peak-vs-mean gap (%) by year:")
    print(yearly_gap.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    yearly_gap.plot(kind="bar", ax=ax, title="Avg daily peak-vs-mean demand gap (%) by year")
    ax.set_ylabel("% gap")
    plt.tight_layout()
    fname = FIG_DIR / "peak_vs_average_gap.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")
    print()


def main():
    master, harmonized, hourly = load_sheets()
    sanity_check(master)
    trend_seasonality(master)
    yoy_growth(master)
    source_mix_evolution(harmonized)
    demand_temperature_correlation(master)
    peak_vs_average_gap(hourly)

    print("=" * 70)
    print("EDA COMPLETE — all charts saved in ./figures/")
    print("=" * 70)
    print("Next: look at the YoY growth printouts for 2020 (COVID dip) and")
    print("2022-2025 (recovery/growth), and check whether the 2017-05 vertical")
    print("line in any chart coincides suspiciously with a 'trend change' —")
    print("if so, that's the reporting-era artifact, not a real-world event.")


if __name__ == "__main__":
    main()
