"""
Feature Engineering for Forecasting
--------------------------------------
Reads MASTER_monthly from india_electricity_master.xlsx and produces a
model-ready dataset with:
  - Low-coverage months handled (interpolated, not dropped, to preserve
    a continuous time index — required by most forecasting models)
  - Lag features (t-1, t-3, t-6, t-12)
  - Rolling mean/std (3, 6, 12 month windows)
  - Calendar features (month, quarter, is_summer_peak)
  - COVID anomaly flag (2020-04 to 2020-06)
  - Exogenous temperature feature (forward/back-filled where missing,
    since only 36 months of temperature data exist)

Output: india_electricity_features.xlsx
"""

import pandas as pd
import numpy as np

INPUT_FILE = "india_electricity_master.xlsx"
OUTPUT_FILE = "india_electricity_features.xlsx"

TARGET_COLS = ["avg_daily_generation", "avg_daily_demand", "avg_daily_consumption"]

# India's summer peak demand season (AC load) — roughly March-June
SUMMER_MONTHS = {3, 4, 5, 6}

# National lockdown period — flagged so models don't treat this as a
# repeating seasonal pattern
COVID_START = pd.Period("2020-04", freq="M")
COVID_END = pd.Period("2020-06", freq="M")


def load_master():
    df = pd.read_excel(INPUT_FILE, sheet_name="MASTER_monthly")
    df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    df = df.sort_values("year_month").reset_index(drop=True)
    df["date"] = df["year_month"].dt.to_timestamp()
    return df


def handle_low_coverage(df):
    """
    Interpolate (don't drop) low-coverage months for the target columns.
    Dropping would break the monthly time index that forecasting models
    (especially seasonal ones) rely on having no gaps. Linear interpolation
    is a reasonable approximation since these are isolated single months,
    not extended gaps.
    """
    df = df.copy()
    n_flagged = int(df["low_coverage_flag"].sum())
    print(f"Interpolating {n_flagged} low-coverage month(s) for target columns "
          f"(index preserved, values estimated from neighbors)...")

    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        df.loc[df["low_coverage_flag"], col] = np.nan
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    return df


def add_lag_features(df):
    df = df.copy()
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        for lag in [1, 3, 6, 12]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df):
    df = df.copy()
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        for window in [3, 6, 12]:
            df[f"{col}_roll_mean{window}"] = df[col].shift(1).rolling(window).mean()
            df[f"{col}_roll_std{window}"] = df[col].shift(1).rolling(window).std()
    return df


def add_calendar_features(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_summer_peak"] = df["month"].isin(SUMMER_MONTHS).astype(int)
    # Cyclical encoding — helps linear/tree models capture month-wraparound
    # (December and January are 1 month apart, not 11)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_covid_flag(df):
    df = df.copy()
    df["is_covid_shock"] = df["year_month"].apply(
        lambda p: 1 if COVID_START <= p <= COVID_END else 0
    )
    return df


def add_temperature_feature(df):
    """
    max_temp only has 36 non-null months. Forward/back-fill using the
    seasonal pattern (same calendar month's average) rather than a naive
    ffill, since temperature is strongly seasonal and a naive fill would
    flatten winter/summer distinction for the missing years.
    """
    df = df.copy()
    if "max_temp" not in df.columns:
        return df

    month_avg_temp = df.groupby(df["date"].dt.month)["max_temp"].transform("mean")
    df["max_temp_filled"] = df["max_temp"].fillna(month_avg_temp)
    df["temp_is_estimated"] = df["max_temp"].isna().astype(int)
    return df


def main():
    df = load_master()
    df = handle_low_coverage(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_covid_flag(df)
    df = add_temperature_feature(df)

    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Models generally can't use the first 12 rows (lag/rolling windows
    # produce NaN there) — flag this clearly rather than silently dropping,
    # so the person doing the forecasting knows the effective usable range.
    usable_from = df["date"].iloc[12] if len(df) > 12 else None
    print(f"\nFirst 12 rows will have NaN in lag12/rolling12 features — "
          f"models should typically start training from {usable_from}.")

    df.to_excel(OUTPUT_FILE, sheet_name="features", index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
