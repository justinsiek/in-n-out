import requests
import csv
import time
import censusgeocode as cg


def get_median_income_b19013(lat, lon, year=2022):
    """
    Finds the Census Tract for a coordinate and returns the Median Household Income (B19013).
    """
    try:
        # 1. Reverse Geocode to get Census Geography
        # Note: Lon comes before Lat in the censusgeocode library
        result = cg.coordinates(x=lon, y=lat)

        if not result or 'Census Tracts' not in result:
            return None, "Coordinate outside of known Census Tracts."

        geo = result['Census Tracts'][0]
        state = geo['STATE']
        county = geo['COUNTY']
        tract = geo['TRACT']

        # 2. Query ACS 5-Year Data for Variable B19013_001E
        # B19013_001E: Median Household Income in the past 12 months (inflation-adjusted)
        base_url = f"https://api.census.gov/data/{year}/acs/acs5"
        params = {
            "get": "NAME,B19013_001E",
            "for": f"tract:{tract}",
            "in": f"state:{state} county:{county}"
            # "key": "YOUR_API_KEY_HERE" # Recommended for large datasets
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            # Census API returns: [['NAME', 'B19013_001E', ...], ['Area Name', 'Value', ...]]
            income_val = data[1][1]

            # Handle cases where data is missing (represented as negative values by Census)
            income = int(income_val) if int(income_val) > 0 else 0
            return income, None
        else:
            return None, f"API Error: {response.status_code}"

    except Exception as e:
        return None, str(e)


def main(input_csv="../in_n_out_california.csv", output_csv="median_income_text.csv"):
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Fetching median income for {len(rows)} locations...")

    for i, row in enumerate(rows):
        lat = float(row["lat"])
        lon = float(row["lon"])

        income, err = get_median_income_b19013(lat, lon)

        if err:
            print(f"  [{i+1}/{len(rows)}] {row.get('name', '')} - {row.get('city', '')} ERROR: {err}")
            row["median_income"] = ""
        else:
            print(f"  [{i+1}/{len(rows)}] {row.get('name', '')} - {row.get('city', '')} income: ${income:,}")
            row["median_income"] = income

        time.sleep(0.1)  # be polite to the Census API

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
