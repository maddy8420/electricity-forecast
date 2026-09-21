"""
India Electricity EDA
------------------------
Reads india_electricity_master.xlsx and produces:
  - Summary stats / missing-value check
  - Trend & seasonality decomposition (generation + demand)
  - Year-over-year growth rates (low-coverage months excluded)
  - A reporting-era boundary check (2017-05-26) so real growth isn't
    confused with a data-collection artifact
  - Source-mix evolution chart (Dec 2018 onward — see note below)
  - Demand vs temperature correlation
  - Peak vs average demand gap over time (partial years excluded)

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
    print("Note: May-Sept 2017 generation figures are built on severely under-")
    print("reported days (as few as 0-8 days/month vs a normal ~30) due to CEA's")
    print("reporting-format transition. Expect a dip/spike artifact there — it's")
    print("not a real production crash.\n")

    for col, label in [
        ("avg_daily_generation", "Generation"),
        ("avg_daily_demand", "Demand (Energy Met)"),
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

    # Exclude low-coverage months from growth calculations — a month with
    # <20 days reported produces a fake spike/crash in YoY comparisons
    # (confirmed cause: e.g. Dec 2022 had only 4 days reported, making
    # Dec 2023 look like +600% growth when nothing unusual happened).
    df = master.set_index("date").sort_index()
    if "low_coverage_flag" in df.columns:
        n_excluded = int(df["low_coverage_flag"].sum())
        print(f"Excluding {n_excluded} low-coverage month(s) from growth calculations "
              f"(see MASTER_monthly 'low_coverage_flag' column).\n")
        for c in df.columns:
            if c != "low_coverage_flag" and pd.api.types.is_numeric_dtype(df[c]):
                df.loc[df["low_coverage_flag"], c] = pd.NA

    for col, label in [
        ("avg_daily_generation", "Generation"),
        ("avg_daily_demand", "Demand"),
        ("avg_daily_consumption", "Consumption"),
    ]:
        if col not in df.columns:
            continue
        yoy = df[col].pct_change(periods=12, fill_method=None) * 100
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
    print("4. SOURCE-MIX EVOLUTION (Dec 2018 onward — see below)")
    print("=" * 70)

    # Start from Dec 2018, not May 2017: Coal/Lignite itemization only
    # becomes reliable from 2018-11-29 onward (confirmed via check_date_ranges).
    # Using May 2017 as a starting point inflates renewables' apparent share,
    # since coal was NaN (treated as 0 in the sum) for nearly that entire month.
    reliable_start = pd.Timestamp("2018-12-01")
    detailed = harmonized[harmonized["date"] >= reliable_start].copy()
    detailed["year_month"] = detailed["date"].dt.to_period("M")

    monthly = detailed.groupby("year_month")[
        ["coal_lignite_mu", "renewables_mu", "hydro_mu", "nuclear_mu", "gas_naptha_diesel_mu"]
    ].mean()  # mean, not sum — consistent with the day-coverage fix elsewhere
    monthly.index = monthly.index.to_timestamp()

    share = monthly.div(monthly.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(share.index, share.T, labels=share.columns)
    ax.set_title("Generation source mix — % share (Dec 2018 onward)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("% share")
    plt.tight_layout()
    fname = FIG_DIR / "source_mix_evolution.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")

    print("\nRenewables % share, first vs last month available:")
    print(f"  {share.index[0].strftime('%Y-%m')}: {share['renewables_mu'].iloc[0]:.1f}%")
    print(f"  {share.index[-1].strftime('%Y-%m')}: {share['renewables_mu'].iloc[-1]:.1f}%")
    print("  (Earlier run showed 23.3% at May 2017 — that was inflated by missing")
    print("   coal data that month, not a real baseline. This range is reliable.)")
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

    # Flag partial years (fewer than ~350 days) so they aren't compared
    # directly against full years — e.g. 2024 may only have data through
    # April/May, which would skew the average toward whichever season
    # happens to be covered.
    days_per_year = hourly.groupby("year").size()
    partial_years = days_per_year[days_per_year < 350].index.tolist()
    if partial_years:
        print(f"  Note: {partial_years} have <350 days of hourly data — excluding "
              f"from year-over-year comparison below (shown separately if present).")

    full_years = hourly[~hourly["year"].isin(partial_years)]
    yearly_gap = full_years.groupby("year")["gap_pct"].mean()
    print("Average daily peak-vs-mean gap (%) by year (full years only):")
    print(yearly_gap.round(2).to_string())

    if partial_years:
        partial_gap = hourly[hourly["year"].isin(partial_years)].groupby("year")["gap_pct"].mean()
        print(f"\nPartial year(s) — not directly comparable, shown for reference only:")
        print(partial_gap.round(2).to_string())

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