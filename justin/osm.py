import requests
import csv
import math


def haversine(lat1, lon1, lat2, lon2):
    """Return distance in km between two coordinates."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def add_nearest_prior_distance(input_csv="densitytest.csv", output_csv="densitytest.csv"):
    """
    For each In-N-Out, find the distance (km) to the closest In-N-Out
    that was built before it. Uses start_date if available, else osm_first_seen.
    """
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # Parse a usable date for sorting
    for row in rows:
        date = row.get("start_date", "").strip().split("T")[0].split(" ")[0]
        if not date:
            date = row.get("osm_first_seen", "").strip()
        row["_sort_date"] = date if date else "9999-99-99"

    rows.sort(key=lambda r: r["_sort_date"])

    for i, row in enumerate(rows):
        prior = rows[:i]
        if not prior:
            row["dist_to_nearest_prior_km"] = ""
            continue

        lat1 = float(row["lat"])
        lon1 = float(row["lon"])
        min_dist = min(
            haversine(lat1, lon1, float(p["lat"]), float(p["lon"])) for p in prior
        )
        row["dist_to_nearest_prior_km"] = round(min_dist, 3)

    # Remove helper key and write back
    fieldnames = [k for k in rows[0].keys() if k != "_sort_date"]
    for row in rows:
        del row["_sort_date"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows with dist_to_nearest_prior_km to {output_csv}")


def get_competitor_density(lat, lon, radius=3000, date=None):
    """
    Query OSM Overpass API to count fast food competitors (excluding In-N-Out)
    within a given radius of a coordinate.

    Args:
        lat: Latitude of the coordinate
        lon: Longitude of the coordinate
        radius: Search radius in meters (default 3000m / 3km)
        date: Optional date string (e.g. "2020-01-01") to query historical data.
              Earliest available: 2012-09-12. If None, queries present day.

    Returns:
        dict with 'count' (number of competitors) and 'competitors' (list of names)
    """
    overpass_url = "https://overpass-api.de/api/interpreter"

    date_setting = f'[date:"{date}T00:00:00Z"]' if date else ""

    query = f"""
    [out:json][timeout:25]{date_setting};
    (
      node[amenity=fast_food][brand!~"In.N.Out"](around:{radius},{lat},{lon});
      way[amenity=fast_food][brand!~"In.N.Out"](around:{radius},{lat},{lon});
    );
    out body;
    """

    response = requests.post(overpass_url, data={"data": query})
    response.raise_for_status()
    data = response.json()

    elements = data.get("elements", [])

    competitors = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", tags.get("brand", "Unknown"))
        competitors.append(name)

    return {
        "count": len(competitors),
        "competitors": competitors,
    }


if __name__ == "__main__":
    add_nearest_prior_distance()
