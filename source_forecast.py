"""Forecast electricity generation by source. Run with: python source_forecast.py."""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

MASTER_FILE = "india_electricity_master.xlsx"
OUTPUT_FILE = "source_forecasts_2040.xlsx"

SOURCES = {
    "coal_lignite_mu": "Coal & Lignite",
    "renewables_mu": "Renewables",
    "hydro_mu": "Hydro",
    "nuclear_mu": "Nuclear",
    "gas_naptha_diesel_mu": "Gas, Naphtha & Diesel",
}
RELIABLE_START = pd.Timestamp("2018-12-01")
FORECAST_END = pd.Timestamp("2040-12-01")
MIN_DAYS = 20
ORDER = (1, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 12)
CHECKPOINT_YEARS = [2030, 2035, 2040]


def load_monthly_sources():
    """Daily harmonized sheet -> monthly average daily value per source."""
    df = pd.read_excel(MASTER_FILE, sheet_name="gen_by_source_harmonized")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= RELIABLE_START].copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = {}
    for col in SOURCES:
        g = df.groupby("month")[col].agg(["mean", "count"])
        series = g["mean"].where(g["count"] >= MIN_DAYS)  # drop thin months
        series = series.asfreq("MS")
        # interpolate gaps inside the data only; never extend past the last real month
        monthly[col] = series.interpolate(method="linear", limit_area="inside")
    return monthly


def forecast_source(series):
    series = series.dropna()
    model = SARIMAX(
        series, order=ORDER, seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    n_steps = (FORECAST_END.to_period("M") - series.index[-1].to_period("M")).n
    fc = model.get_forecast(steps=n_steps).summary_frame(alpha=0.05)
    fc = fc.rename(columns={"mean": "forecast",
                            "mean_ci_lower": "ci_lower_95",
                            "mean_ci_upper": "ci_upper_95"})
    fc = fc[["forecast", "ci_lower_95", "ci_upper_95"]].clip(lower=0)  # generation can't be negative
    fc.index.name = "date"
    return series, fc


def main():
    monthly = load_monthly_sources()
    history_rows, forecast_rows = [], []

    print("=" * 70)
    print("PER-SOURCE FORECASTS (avg daily MU, December of each year)")
    print("=" * 70)
    for col, label in SOURCES.items():
        series, fc = forecast_source(monthly[col])
        print(f"\n{label}: {len(series)} months of history "
              f"({series.index.min().date()} to {series.index.max().date()})")
        for year in CHECKPOINT_YEARS:
            d = pd.Timestamp(f"{year}-12-01")
            if d in fc.index:
                r = fc.loc[d]
                print(f"  {year}: {r['forecast']:,.0f}  "
                      f"[{r['ci_lower_95']:,.0f}, {r['ci_upper_95']:,.0f}]")

        h = series.rename("value").reset_index().rename(columns={"index": "date", "month": "date"})
        h["source"] = label
        history_rows.append(h)
        f = fc.reset_index()
        f["source"] = label
        forecast_rows.append(f)

    history = pd.concat(history_rows, ignore_index=True)
    forecast = pd.concat(forecast_rows, ignore_index=True)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        history.to_excel(writer, sheet_name="history", index=False)
        forecast.to_excel(writer, sheet_name="forecast", index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()