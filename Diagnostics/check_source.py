import pandas as pd
from pathlib import Path

source_file = Path(__file__).resolve().parent / "data" / "Daily_Power_Gen_Source_march_23.csv"
df = pd.read_csv(source_file)
print("Unique values in 'source' column:")
for s in sorted(df["source"].unique()):
    print(f"  - {s}")

print()
print("Row count per source:")
print(df["source"].value_counts())