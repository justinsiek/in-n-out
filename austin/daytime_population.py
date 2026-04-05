import requests
import csv
import time
import gzip
import io
import os
import censusgeocode as cg

LODES_YEAR = 2021
ACS_YEAR = 2021

WAC_URL = f"https://lehd.ces.census.gov/data/lodes/LODES8/ca/wac/ca_wac_S000_JT00_{LODES_YEAR}.csv.gz"
RAC_URL = f"https://lehd.ces.census.gov/data/lodes/LODES8/ca/rac/ca_rac_S000_JT00_{LODES_YEAR}.csv.gz"

WAC_CACHE = "ca_wac.csv.gz"
RAC_CACHE = "ca_rac.csv.gz"


def download_lodes_file(url, cache_path):
    if os.path.exists(cache_path):
        print(f"  Using cached {cache_path}")
        return
    print(f"  Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(cache_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"  Saved to {cache_path}")


def load_lodes_by_tract(cache_path, geocode_col, count_col="C000"):
    """
    Load a LODES WAC or RAC file and aggregate job counts by census tract.
    Census tract = first 11 digits of the 15-digit block FIPS code.
    """
    tract_totals = {}
    with gzip.open(cache_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_fips = row[geocode_col].zfill(15)
            tract_fips = block_fips[:11]
            count = int(row[count_col])
            tract_totals[tract_fips] = tract_totals.get(tract_fips, 0) + count
    return tract_totals


def get_acs_population(state, county, tract, year=ACS_YEAR):
    """Fetch total resident population (B01003) for a census tract."""
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "B01003_001E",
        "for": f"tract:{tract}",
        "in": f"state:{state} county:{county}",
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        val = int(data[1][0])
        return val if val >= 0 else 0
    return None


def get_tract_info(lat, lon):
    """Reverse geocode a lat/lon to get state, county, tract FIPS codes."""
    result = cg.coordinates(x=lon, y=lat)
    if not result or "Census Tracts" not in result:
        return None
    geo = result["Census Tracts"][0]
    state = geo["STATE"]
    county = geo["COUNTY"]
    tract = geo["TRACT"]
    # Build 11-digit tract FIPS: state(2) + county(3) + tract(6)
    tract_fips = state.zfill(2) + county.zfill(3) + tract.zfill(6)
    return {"state": state, "county": county, "tract": tract, "tract_fips": tract_fips}


def main(input_csv="../shared/in_n_out_california.csv", output_csv="daytime_population.csv"):

    # Step 1: Download LODES data (cached after first run)
    print("Loading LODES data...")
    download_lodes_file(WAC_URL, WAC_CACHE)
    download_lodes_file(RAC_URL, RAC_CACHE)

    print("Aggregating WAC (workers working in tract)...")
    wac_by_tract = load_lodes_by_tract(WAC_CACHE, geocode_col="w_geocode")

    print("Aggregating RAC (workers living in tract)...")
    rac_by_tract = load_lodes_by_tract(RAC_CACHE, geocode_col="h_geocode")

    # Step 2: Load In-N-Out locations
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"\nFetching daytime population for {len(rows)} locations...\n")

    for i, row in enumerate(rows):
        lat = float(row["lat"])
        lon = float(row["lon"])

        geo = get_tract_info(lat, lon)
        if not geo:
            print(f"  [{i+1}/{len(rows)}] {row.get('name', '')} - {row.get('city','')} ERROR: no tract found")
            row["resident_pop"] = ""
            row["workers_in_tract"] = ""
            row["workers_from_tract"] = ""
            row["daytime_pop"] = ""
            time.sleep(0.1)
            continue

        tract_fips = geo["tract_fips"]

        # Resident population from ACS
        resident_pop = get_acs_population(geo["state"], geo["county"], geo["tract"])
        time.sleep(0.1)

        workers_in = wac_by_tract.get(tract_fips, 0)   # people working in this tract
        workers_out = rac_by_tract.get(tract_fips, 0)  # residents who hold jobs (commute out)

        # Daytime pop = residents - those who leave for work + those who come in for work
        if resident_pop is not None:
            daytime_pop = resident_pop - workers_out + workers_in
        else:
            daytime_pop = ""

        row["resident_pop"] = resident_pop if resident_pop is not None else ""
        row["workers_in_tract"] = workers_in
        row["workers_from_tract"] = workers_out
        row["daytime_pop"] = daytime_pop

        print(
            f"  [{i+1}/{len(rows)}] {row.get('name', '')} - {row.get('city','')} | "
            f"resident: {resident_pop:,} | workers_in: {workers_in:,} | "
            f"workers_out: {workers_out:,} | daytime: {daytime_pop:,}"
            if isinstance(daytime_pop, int) else
            f"  [{i+1}/{len(rows)}] {row.get('name', '')} - {row.get('city','')} ERROR: population lookup failed"
        )

    # Step 3: Write output CSV
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target = sys.argv[1]
        main(input_csv=target, output_csv=target)
    else:
        main()
