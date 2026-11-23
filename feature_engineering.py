"""Build forecasting features from monthly electricity data. Run with: python feature_engineering.py."""

import pandas as pd
import numpy as np

INPUT_FILE = "india_electricity_master.xlsx"
OUTPUT_FILE = "india_electricity_features.xlsx"

TARGET_COLS = ["avg_daily_generation", "avg_daily_demand", "avg_daily_consumption"]

# India's summer peak demand season (AC load) — roughly March-June
SUMMER_MONTHS = {3, 4, 5, 6}

# Flag lockdown months so models do not treat them as recurring seasonality.
COVID_START = pd.Period("2020-04", freq="M")
COVID_END = pd.Period("2020-06", freq="M")


def load_master():
    df = pd.read_excel(INPUT_FILE, sheet_name="MASTER_monthly")
    df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    df = df.sort_values("year_month").reset_index(drop=True)
    df["date"] = df["year_month"].dt.to_timestamp()
    return df


def handle_low_coverage(df):
    """Interpolate low-coverage months without breaking the monthly time index."""
    df = df.copy()
    n_flagged = int(df["low_coverage_flag"].sum())
    print(f"Interpolating {n_flagged} low-coverage month(s) for target columns "
          f"(index preserved, values estimated from neighbors)...")

    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        df.loc[df["low_coverage_flag"], col] = np.nan
        df[col] = df[col].interpolate(method="linear", limit_area="inside")

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
    # Encode month cyclically because December and January are adjacent.
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
    """Fill missing temperatures using the seasonal pattern."""
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

    # Flag initial lag/rolling rows so users know the effective training range.
    usable_from = df["date"].iloc[12] if len(df) > 12 else None
    print(f"\nFirst 12 rows will have NaN in lag12/rolling12 features — "
          f"models should typically start training from {usable_from}.")

    df.to_excel(OUTPUT_FILE, sheet_name="features", index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
