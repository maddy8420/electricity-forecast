import pandas as pd

INPUT = "india_electricity_master.xlsx"

cons = pd.read_excel(INPUT, sheet_name="raw_consumption")
cons["date"] = pd.to_datetime(cons["date"])

print("Days reported per month, full range (look for near-zero months):")
monthly = cons.groupby(cons["date"].dt.to_period("M")).agg(
    days_reported=("Total Consumption", "count"),
    monthly_sum=("Total Consumption", "sum"),
)
# Only show months with unusually low day counts (likely culprits)
print(monthly.to_string())

print()
print("Months with fewer than 20 days reported (likely YoY distortion sources):")
print(monthly[monthly["days_reported"] < 20].to_string())
