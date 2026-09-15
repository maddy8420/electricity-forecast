import pandas as pd

df = pd.read_csv("data/Daily_Power_Gen_Source_march_23.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print("Unique dates:", df["date"].nunique())

# Expected: ~10 years -> roughly 3650-3660 unique calendar days
expected_days = (df["date"].max() - df["date"].min()).days + 1
print("Expected calendar days in that span:", expected_days)
print("Missing days:", expected_days - df["date"].nunique())

# Check for suspicious jumps: if dayfirst mis-parsed some rows, you'd often
# see dates snap out of chronological order or cluster oddly by month
by_month = df.groupby(df["date"].dt.to_period("M")).size()
print("\nRow count by month (spot-check for anything zero or wildly off):")
print(by_month.to_string())
