"""
India Electricity Data Consolidation (v2)
--------------------------------------------
Fixes a hierarchy problem in Daily_Power_Gen_Source_march_23.csv:
CEA changed its reporting granularity partway through the dataset.
  - Early period: 'Thermal (Coal & Lignite)' reported as one combined row
  - Later period: 'Coal' and 'Lignite' itemized separately
  - Similarly: 'RES (Wind, Solar, Biomass & Others)' combined vs.
    'Wind Gen' / 'Solar Gen' itemized separately in other periods
Summing all non-Total rows blindly double-counts on any date where both
a parent category and its components happen to be present.

Fix: never recompute the national total from parts — trust the reported
'Total' row directly. For source-composition analysis, build harmonized
columns that use the itemized value when available and fall back to the
combined value when not, so you get one continuous series across the
whole reporting-format change instead of holes.
"""

import re
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_FILE = "india_electricity_master.xlsx"

FILES = {
    "gen_by_source": DATA_DIR / "Daily_Power_Gen_Source_march_23.csv",
    "state_demand":  DATA_DIR / "Daily_Power_Gen_States_march_23.csv",
    "hourly_load":   DATA_DIR / "hourlyLoadDataIndia.xlsx",
    "consumption":   DATA_DIR / "Indias_Electricity_Consumption_Dataset.csv",
    "monthly_temp":  DATA_DIR / "monthly_temp.xlsx",
}


def clean_colname(c):
    c = str(c).strip()
    c = re.sub(r"\s+", " ", c)
    c = c.replace("EasternRegion", "Eastern Region")
    return c


# ----------------------------------------------------------------------
# LOADERS
# ----------------------------------------------------------------------

def load_gen_by_source():
    df = pd.read_csv(FILES["gen_by_source"])
    df.columns = [clean_colname(c) for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    df["source"] = (
        df["source"].astype(str).str.strip()
        .str.replace(r"\s*\(MU\)", "", regex=True)
        .str.replace(r"\s*Gen$", "", regex=True)
    )

    totals = df[df["source"] == "Total"].copy()
    by_source = df[df["source"] != "Total"].copy()
    return by_source, totals


def load_state_demand():
    df = pd.read_csv(FILES["state_demand"])
    df.columns = [clean_colname(c) for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.rename(columns={
        "Max.Demand Met during the day(MW)": "max_demand_mw",
        "Shortage during maximum Demand(MW)": "shortage_mw",
        "Energy Met (MU)": "energy_met_mu",
    })
    return df


def load_hourly_load():
    df = pd.read_excel(FILES["hourly_load"])
    df.columns = [clean_colname(c) for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    return df


def load_consumption():
    df = pd.read_csv(FILES["consumption"])
    df.columns = [clean_colname(c) for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns={"Dates": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_monthly_temp():
    df = pd.read_excel(FILES["monthly_temp"])
    df.columns = [clean_colname(c) for c in df.columns]
    df["year_month"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str), format="%Y-%b"
    ).dt.to_period("M")
    return df


# ----------------------------------------------------------------------
# GENERATION: raw pivot + harmonized hierarchy-aware columns
# ----------------------------------------------------------------------

def build_generation_tables(by_source, source_totals):
    # Raw pivot, exactly as reported, NaN where a category wasn't reported that day.
    # Kept for transparency/debugging, not for summing.
    raw_wide = by_source.pivot_table(
        index="date", columns="source", values="All India", aggfunc="sum"
    ).reset_index()

    p = by_source.pivot_table(index="date", columns="source", values="All India", aggfunc="sum")

    def col(name):
        return p[name] if name in p.columns else pd.Series(index=p.index, dtype=float)

    coal = col("Coal")
    lignite = col("Lignite")
    thermal_combined = col("Thermal (Coal & Lignite)")
    wind = col("Wind")
    solar = col("Solar")
    res_combined = col("RES (Wind, Solar, Biomass & Others)")

    harmonized = pd.DataFrame(index=p.index)
    # Coal+Lignite family: prefer itemized sum, fall back to combined 'Thermal' row
    itemized_coal_lignite = coal.fillna(0) + lignite.fillna(0)
    harmonized["coal_lignite_mu"] = itemized_coal_lignite.where(
        coal.notna() | lignite.notna(), thermal_combined
    )

    # Renewables family: prefer combined RES figure (most complete — includes
    # biomass/others); only fall back to Wind+Solar sum on dates RES is missing
    # (this fallback will UNDERSTATE renewables slightly, since it excludes
    # biomass/others — flagged here rather than silently absorbed).
    itemized_wind_solar = wind.fillna(0) + solar.fillna(0)
    harmonized["renewables_mu"] = res_combined.where(
        res_combined.notna(), itemized_wind_solar
    )
    harmonized["renewables_mu_is_wind_solar_only"] = res_combined.isna() & (
        wind.notna() | solar.notna()
    )

    harmonized["hydro_mu"] = col("Hydro")
    harmonized["nuclear_mu"] = col("Nuclear")
    harmonized["gas_naptha_diesel_mu"] = col("Gas, Naptha & Diesel")
    harmonized["wind_mu"] = wind          # only populated where itemized; NaN otherwise
    harmonized["solar_mu"] = solar        # only populated where itemized; NaN otherwise

    harmonized = harmonized.reset_index()

    # DATA ERA FLAG — before 2017-05-26, only Hydro/Wind(/Solar from 2016) are
    # separable; Coal, Lignite, Nuclear, and Gas are all bundled inside 'Total'
    # with no way to split them individually. Don't let downstream charts imply
    # precision that doesn't exist for this period.
    era_cutoff = pd.Timestamp("2017-05-26")
    harmonized["data_era"] = harmonized["date"].apply(
        lambda d: "2013-2017 (aggregated reporting)" if d < era_cutoff
        else "2017-2023 (detailed reporting)"
    )

    # For the aggregated era, coal/nuclear/gas can't be split — null them out
    # explicitly rather than leaving a misleading partial value, and instead
    # provide one honest 'other_mu' residual (Total minus what IS separable).
    early_mask = harmonized["date"] < era_cutoff
    harmonized.loc[early_mask, ["coal_lignite_mu", "nuclear_mu", "gas_naptha_diesel_mu"]] = pd.NA

    # Reported total — trusted as-is, never recomputed from parts.
    total_reported = (
        source_totals.groupby("date")["All India"].sum()
        .rename("national_total_generation_mu")
        .reset_index()
    )

    # 'other_mu': for the aggregated era only, this is Total minus the parts
    # we CAN separate (Hydro + Wind + Solar) — an honest single bucket for
    # Coal+Lignite+Nuclear+Gas+Biomass combined, since the raw data gives no
    # way to split them further in this period. Left as NaN in the detailed
    # era, since every source is already accounted for individually there.
    harmonized = harmonized.merge(total_reported, on="date", how="left")
    known_early = (
        harmonized["hydro_mu"].fillna(0)
        + harmonized["wind_mu"].fillna(0)
        + harmonized["solar_mu"].fillna(0)
    )
    harmonized["other_mu"] = pd.NA
    harmonized.loc[early_mask, "other_mu"] = (
        harmonized.loc[early_mask, "national_total_generation_mu"] - known_early[early_mask]
    )
    harmonized = harmonized.drop(columns=["national_total_generation_mu"])

    return raw_wide, harmonized, total_reported


def build_national_demand_daily(state_demand):
    return (
        state_demand.groupby("date")
        .agg(
            national_max_demand_mw=("max_demand_mw", "sum"),
            national_shortage_mw=("shortage_mw", "sum"),
            national_energy_met_mu=("energy_met_mu", "sum"),
        )
        .reset_index()
    )


def build_hourly_agg_daily(hourly_load):
    agg = (
        hourly_load.groupby("date")["National Hourly Demand"]
        .agg(["mean", "max", "min"])
        .rename(columns={
            "mean": "hourly_demand_daily_mean",
            "max": "hourly_demand_daily_max",
            "min": "hourly_demand_daily_min",
        })
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["date"])
    return agg


def build_monthly_master(total_reported, harmonized, demand_daily, hourly_daily, consumption, temp):
    def to_month(df, date_col="date"):
        out = df.copy()
        out["year_month"] = pd.to_datetime(out[date_col]).dt.to_period("M")
        return out

    gen_m = to_month(total_reported).groupby("year_month")["national_total_generation_mu"].sum()
    coal_m = to_month(harmonized).groupby("year_month")["coal_lignite_mu"].sum().rename("coal_lignite_mu")
    ren_m = to_month(harmonized).groupby("year_month")["renewables_mu"].sum().rename("renewables_mu")
    hydro_m = to_month(harmonized).groupby("year_month")["hydro_mu"].sum().rename("hydro_mu")
    nuclear_m = to_month(harmonized).groupby("year_month")["nuclear_mu"].sum().rename("nuclear_mu")
    dem_m = to_month(demand_daily).groupby("year_month")["national_energy_met_mu"].sum().rename("total_energy_met_mu")
    hr_m = to_month(hourly_daily).groupby("year_month")["hourly_demand_daily_max"].max().rename("monthly_peak_demand_mw")
    con_m = to_month(consumption).groupby("year_month")["Total Consumption"].sum().rename("total_consumption_mu")

    master = pd.concat([gen_m, coal_m, ren_m, hydro_m, nuclear_m, dem_m, hr_m, con_m], axis=1).reset_index()
    master = master.merge(temp[["year_month", "Monthly_load", "max_temp"]], on="year_month", how="left")
    master["year_month"] = master["year_month"].astype(str)
    return master


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print("Loading raw files...")
    by_source, source_totals = load_gen_by_source()
    state_demand = load_state_demand()
    hourly_load = load_hourly_load()
    consumption = load_consumption()
    monthly_temp = load_monthly_temp()

    print("Building generation tables (hierarchy-aware)...")
    raw_wide, harmonized, total_reported = build_generation_tables(by_source, source_totals)
    demand_daily = build_national_demand_daily(state_demand)
    hourly_daily = build_hourly_agg_daily(hourly_load)

    n_fallback_days = int(harmonized["renewables_mu_is_wind_solar_only"].sum())
    print(f"  Note: {n_fallback_days} day(s) had no combined 'RES' figure and used "
          f"Wind+Solar only — all in the 2013-2017 window, before India's solar "
          f"buildout, so this doesn't affect your 2017-2023 renewables trend.")

    era_counts = harmonized["data_era"].value_counts()
    print(f"\n  Data era split:")
    for era, n in era_counts.items():
        print(f"    {era}: {n} days")
    print("  Before 2017-05-26, Coal/Nuclear/Gas cannot be individually separated "
          "in this file — bundled into 'other_mu'. Scope any full source-composition "
          "chart (coal vs renewables share) to 2017-2023 for a fair comparison; use "
          "Hydro + Total + 'other_mu' for the earlier window instead.")

    monthly_master = build_monthly_master(
        total_reported, harmonized, demand_daily, hourly_daily, consumption, monthly_temp
    )

    print(f"Writing {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        raw_wide.to_excel(writer, sheet_name="raw_gen_by_source_wide", index=False)
        state_demand.to_excel(writer, sheet_name="raw_state_demand", index=False)
        consumption.to_excel(writer, sheet_name="raw_consumption", index=False)
        monthly_temp.to_excel(writer, sheet_name="raw_monthly_temp", index=False)

        harmonized.to_excel(writer, sheet_name="gen_by_source_harmonized", index=False)
        total_reported.to_excel(writer, sheet_name="national_gen_total_daily", index=False)
        demand_daily.to_excel(writer, sheet_name="national_demand_daily", index=False)
        hourly_daily.to_excel(writer, sheet_name="hourly_agg_daily", index=False)
        monthly_master.to_excel(writer, sheet_name="MASTER_monthly", index=False)

    print("Done.")
    print("\nKey sheets:")
    print("  - national_gen_total_daily : trusted daily total generation (reported, not recomputed)")
    print("  - gen_by_source_harmonized : continuous per-source series across the reporting-format change")
    print("  - MASTER_monthly           : everything joined at monthly resolution, ready for EDA")


if __name__ == "__main__":
    main()