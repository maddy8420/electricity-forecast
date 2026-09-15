import pandas as pd

INPUT = "india_electricity_master.xlsx"

print("="*70)
print("1. CONSUMPTION SPIKE CHECK — raw_consumption sheet, tail end")
print("="*70)
cons = pd.read_excel(INPUT, sheet_name="raw_consumption")
cons["date"] = pd.to_datetime(cons["date"])
cons = cons.sort_values("date")
print(cons[["date", "Total Consumption"]].tail(20).to_string())
print(f"\nRow count by year:")
print(cons.groupby(cons["date"].dt.year).size())
print(f"\nDuplicate dates in consumption file: {cons['date'].duplicated().sum()}")

print()
print("="*70)
print("2. GENERATION CLIFF CHECK — days reported per month, Apr-Aug 2017")
print("="*70)
total_daily = pd.read_excel(INPUT, sheet_name="national_gen_total_daily")
total_daily["date"] = pd.to_datetime(total_daily["date"])
window = total_daily[(total_daily["date"] >= "2017-03-01") & (total_daily["date"] <= "2017-09-01")]
print(window.groupby(window["date"].dt.to_period("M")).agg(
    days_reported=("national_total_generation_mu", "count"),
    monthly_sum=("national_total_generation_mu", "sum"),
    avg_daily=("national_total_generation_mu", "mean"),
))

print()
print("="*70)
print("3. RENEWABLES FIRST MONTH CHECK — May 2017 completeness")
print("="*70)
harm = pd.read_excel(INPUT, sheet_name="gen_by_source_harmonized")
harm["date"] = pd.to_datetime(harm["date"])
may2017 = harm[(harm["date"] >= "2017-05-01") & (harm["date"] <= "2017-05-31")]
print(f"Days present in May 2017: {len(may2017)} (out of 31 possible)")
print(may2017[["date", "renewables_mu", "coal_lignite_mu"]].to_string())

print()
print("="*70)
print("4. 2024 HOURLY DATA COMPLETENESS CHECK")
print("="*70)
hourly = pd.read_excel(INPUT, sheet_name="hourly_agg_daily")
hourly["date"] = pd.to_datetime(hourly["date"])
by_year = hourly.groupby(hourly["date"].dt.year).size()
print("Days present per year in hourly_agg_daily:")
print(by_year)
