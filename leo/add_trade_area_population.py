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

CSV_PATH = "leo/trade_area_in_n_out_california.csv"
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

    # Cache (state, county, tract) → population to avoid duplicate ACS calls
    pop_cache: dict[tuple[str, str, str], int | None] = {}

    total = len(rows)
    for i, row in enumerate(rows, start=1):
        # Skip if already populated
        if row.get(POPULATION_COL) not in ("", None):
            print(f"[{i:3}/{total}] skip (already set): {row.get('city', '')}", flush=True)
            continue

        lat_str = row.get("lat", "").strip()
        lon_str = row.get("lon", "").strip()
        if not lat_str or not lon_str:
            print(f"[{i:3}/{total}] skip (no lat/lon): {row.get('city', '')}", flush=True)
            row[POPULATION_COL] = ""
            continue

        lat, lon = float(lat_str), float(lon_str)
        city = row.get("city", "") or f"{lat},{lon}"
        print(f"[{i:3}/{total}] {city} ({lat}, {lon})", flush=True)

        state, county, tract = geocode_to_tract(lat, lon)
        time.sleep(SLEEP_SEC)

        if state is None:
            row[POPULATION_COL] = ""
            continue

        key = (state, county, tract)
        if key not in pop_cache:
            pop_cache[key] = fetch_tract_population(state, county, tract)
            time.sleep(SLEEP_SEC)

        pop = pop_cache[key]
        row[POPULATION_COL] = pop if pop is not None else ""
        print(f"         tract {state}-{county}-{tract}  →  pop {pop}", flush=True)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Wrote {total} rows to {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
