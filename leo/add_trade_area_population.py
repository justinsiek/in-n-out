"""
Enrich trade_area_in_n_out_california.csv with Trade Area Population
sourced from U.S. Census ACS 5-Year Estimates (2022) at the Census Tract level.

Steps per row:
  1. Census Geocoder  → resolves (lat, lon) to (state, county, tract) FIPS codes.
  2. ACS 5-Year API   → fetches B01003_001E (Total Population) for that tract.

The resulting value is the resident population of the census tract that contains
each In-N-Out location, which serves as a proxy for the surrounding trade-area
population.
"""

import csv
import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "leo/trade_area_in_n_out_california.csv"
ACS_YEAR = 2022
POPULATION_COL = "trade_area_population"
SLEEP_SEC = 0.35  # be polite to Census APIs; ~560 calls total


# ── helpers ──────────────────────────────────────────────────────────────────

def geocode_to_tract(lat: float, lon: float) -> tuple[str, str, str] | tuple[None, None, None]:
    """Return (state_fips, county_fips, tract_code) for a lat/lon pair."""
    url = (
        "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
        f"?x={lon}&y={lat}"
        "&benchmark=Public_AR_Current&vintage=Current_Current&format=json"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        tracts = data["result"]["geographies"].get("Census Tracts", [])
        if not tracts:
            return None, None, None
        t = tracts[0]
        return t["STATE"], t["COUNTY"], t["TRACT"]
    except Exception as exc:
        print(f"    [geocoder error] ({lat}, {lon}): {exc}")
        return None, None, None


def fetch_tract_population(state: str, county: str, tract: str) -> int | None:
    """Return ACS 5-Year total population estimate for one census tract."""
    url = (
        f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
        f"?get=B01003_001E"
        f"&for=tract:{tract}"
        f"&in=state:{state}%20county:{county}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        # data[0] = header row, data[1] = values row
        if len(data) < 2:
            return None
        value = data[1][0]
        return int(value) if value is not None else None
    except Exception as exc:
        print(f"    [ACS error] state={state} county={county} tract={tract}: {exc}")
        return None


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Add new column header if not already present
    if POPULATION_COL not in fieldnames:
        fieldnames = list(fieldnames) + [POPULATION_COL]

    total = len(rows)

    # Step 1: Geocode all rows in parallel to get tract info
    def geocode_row(idx_row):
        idx, row = idx_row
        if row.get(POPULATION_COL) not in ("", None):
            return idx, None, None, None, True
        lat_str = row.get("lat", "").strip()
        lon_str = row.get("lon", "").strip()
        if not lat_str or not lon_str:
            return idx, None, None, None, False
        return idx, *geocode_to_tract(float(lat_str), float(lon_str)), False

    print("Geocoding all rows (10 threads)...", flush=True)
    tract_results = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(geocode_row, (i, row)): i for i, row in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            done += 1
            idx, state, county, tract, skipped = future.result()
            tract_results[idx] = (state, county, tract, skipped)
            if done % 100 == 0:
                print(f"  Geocoded {done}/{total}", flush=True)

    print(f"  Geocoded {total} rows", flush=True)

    # Step 2: Collect unique tracts and fetch populations in parallel
    unique_tracts = set()
    for state, county, tract, skipped in tract_results.values():
        if state is not None:
            unique_tracts.add((state, county, tract))

    print(f"Fetching population for {len(unique_tracts)} unique tracts (10 threads)...", flush=True)
    pop_cache: dict[tuple[str, str, str], int | None] = {}

    def fetch_pop(key):
        return key, fetch_tract_population(*key)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(fetch_pop, key) for key in unique_tracts]
        for future in as_completed(futures):
            key, pop = future.result()
            pop_cache[key] = pop

    print(f"  Got population for {len(pop_cache)} tracts", flush=True)

    # Step 3: Assign values
    for i, row in enumerate(rows):
        state, county, tract, skipped = tract_results[i]
        if skipped:
            continue
        if state is None:
            row[POPULATION_COL] = ""
            continue
        pop = pop_cache.get((state, county, tract))
        row[POPULATION_COL] = pop if pop is not None else ""

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Wrote {total} rows to {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
