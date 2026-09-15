"""
India Electricity Data Pipeline
--------------------------------
Pulls historical electricity data from:
  1. CEA (Central Electricity Authority) open API — no key needed
  2. data.gov.in Open Government Data API — free key needed (see below)

Saves everything into one Excel workbook, one sheet per dataset.

SETUP BEFORE RUNNING
=====================
1. CEA endpoints below are open (no key), but CEA's server is slow/flaky —
   this script retries and times out gracefully rather than hanging forever.

2. For data.gov.in:
   - Go to https://data.gov.in, create a free account
   - Go to "My Account" -> "API Key" to get your key (does not expire)
   - Search the catalog for the specific electricity dataset you want
     (e.g. "per capita electricity consumption", "all india installed
     capacity"). Each dataset's API page gives you its RESOURCE_ID.
   - Paste your key + the resource IDs you find into DATAGOVIN_CONFIG below.
     Leave a resource_id blank / remove the entry to skip it — this script
     will just skip datasets that aren't configured instead of failing.

Run:
    python data_pipeline.py
Output:
    india_electricity_data.xlsx  (in the same folder)
"""

import time
import requests
import pandas as pd
from datetime import datetime

# ----------------------------------------------------------------------
# CONFIG — fill these in
# ----------------------------------------------------------------------

DATAGOVIN_API_KEY = "PASTE_YOUR_DATA_GOV_IN_KEY_HERE"

# Add resource IDs here as you find them on data.gov.in.
# Format: "sheet_name": "resource_id"
DATAGOVIN_CONFIG = {
    # "per_capita_consumption": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    # "state_wise_generation": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
}

# CEA open endpoints — documented at
# https://cea.nic.in/api-for-central-electricity-authority-data/
CEA_ENDPOINTS = {
    "cea_installed_capacity_res": "https://cea.nic.in/api/instcap_allindia_res.php",
    "cea_installed_capacity_allindia": "https://cea.nic.in/api/installed_capacity_allindia.php",
    "cea_installed_capacity_region": "https://cea.nic.in/api/installed_capacity.php",
    "cea_installed_capacity_state": "https://cea.nic.in/api/installed_capacity_statewise.php",
    "cea_power_supply_energy": "https://cea.nic.in/api/psp_energy.php",
    "cea_power_supply_peak": "https://cea.nic.in/api/psp_peak.php",
    "cea_transformation_capacity": "https://cea.nic.in/api/transformation_substations.php",
    "cea_transmission_lines": "https://cea.nic.in/api/transmission_lines.php",
    "cea_power_generation": "https://cea.nic.in/api/power_generation.php",
    "cea_renewable_energy": "https://cea.nic.in/api/renewable_energy.php",
    "cea_installed_capacity_composition": "https://cea.nic.in/api/installed_capacity_composition.php",
    "cea_per_capita_consumption": "https://cea.nic.in/api/percapitalConsumtion.php",
}

OUTPUT_FILE = "india_electricity_data.xlsx"
REQUEST_TIMEOUT = 20   # seconds
MAX_RETRIES = 2
RETRY_DELAY = 3        # seconds between retries


# ----------------------------------------------------------------------
# FETCH HELPERS
# ----------------------------------------------------------------------

def fetch_json(url, params=None, label=""):
    """Fetch a URL and return parsed JSON, with retries. Returns None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  [{label}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    print(f"  [{label}] giving up after {MAX_RETRIES} attempts.")
    return None


def json_to_dataframe(data):
    """Best-effort conversion of whatever JSON shape comes back into a DataFrame."""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        # Common patterns: {"records": [...]}  or a single dict of arrays
        for key in ("records", "data", "result"):
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])
        # Fallback: try flattening whatever's there
        try:
            return pd.json_normalize(data)
        except Exception:
            return pd.DataFrame([data])
    return pd.DataFrame()


def fetch_cea_datasets():
    """Fetch all configured CEA endpoints. Returns dict of sheet_name -> DataFrame."""
    results = {}
    print("Fetching CEA datasets...")
    for sheet_name, url in CEA_ENDPOINTS.items():
        print(f" - {sheet_name}")
        data = fetch_json(url, label=sheet_name)
        df = json_to_dataframe(data)
        if not df.empty:
            results[sheet_name] = df
            print(f"   -> {len(df)} rows")
        else:
            print(f"   -> no data (skipped)")
    return results


def fetch_datagovin_datasets():
    """Fetch all configured data.gov.in resources. Returns dict of sheet_name -> DataFrame."""
    results = {}
    if not DATAGOVIN_CONFIG:
        print("No data.gov.in resource IDs configured yet — skipping.")
        return results
    if DATAGOVIN_API_KEY.startswith("PASTE_"):
        print("data.gov.in API key not set — skipping data.gov.in datasets.")
        return results

    print("Fetching data.gov.in datasets...")
    base_url = "https://api.data.gov.in/resource/{resource_id}"
    for sheet_name, resource_id in DATAGOVIN_CONFIG.items():
        print(f" - {sheet_name}")
        url = base_url.format(resource_id=resource_id)
        params = {
            "api-key": DATAGOVIN_API_KEY,
            "format": "json",
            "limit": 10000,  # pull as many rows as available
        }
        data = fetch_json(url, params=params, label=sheet_name)
        df = json_to_dataframe(data)
        if not df.empty:
            results[sheet_name] = df
            print(f"   -> {len(df)} rows")
        else:
            print(f"   -> no data (skipped)")
    return results


def save_to_excel(all_data, output_file):
    """Write each DataFrame to its own sheet in one Excel workbook."""
    if not all_data:
        print("Nothing fetched — no file written.")
        return

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in all_data.items():
            # Excel sheet names capped at 31 chars
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

        # A small metadata sheet so you know when this was pulled
        meta = pd.DataFrame({
            "pulled_at": [datetime.now().isoformat()],
            "sheets": [", ".join(all_data.keys())],
        })
        meta.to_excel(writer, sheet_name="_metadata", index=False)

    print(f"\nSaved {len(all_data)} sheet(s) to {output_file}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    all_data = {}
    all_data.update(fetch_cea_datasets())
    all_data.update(fetch_datagovin_datasets())
    save_to_excel(all_data, OUTPUT_FILE)


if __name__ == "__main__":
    main()
