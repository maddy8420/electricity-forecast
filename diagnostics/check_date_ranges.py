import pandas as pd

df = pd.read_csv("data/Daily_Power_Gen_Source_march_23.csv")
df["date"] = pd.to_datetime(df["date"])
df["source"] = df["source"].str.strip().str.replace(r"\s*\(MU\)", "", regex=True).str.replace(r"\s*Gen$", "", regex=True)

for s in ["RES (Wind, Solar, Biomass & Others)", "Wind", "Solar", "Coal", "Lignite", "Thermal (Coal & Lignite)"]:
    sub = df[df["source"] == s]
    if len(sub):
        print(f"{s:45s} -> {sub['date'].min().date()} to {sub['date'].max().date()}  ({len(sub)} days)")
    else:
        print(f"{s:45s} -> not found")

print()
print("Checking full-range coverage for reconstruction:")
for s in ["Hydro", "Nuclear", "Gas, Naptha & Diesel", "Total"]:
    sub = df[df["source"] == s] if s != "Total" else pd.read_csv("data/Daily_Power_Gen_Source_march_23.csv")
    sub = df[df["source"] == s]
    if len(sub):
        print(f"{s:25s} -> {sub['date'].min().date()} to {sub['date'].max().date()}  ({len(sub)} days)")