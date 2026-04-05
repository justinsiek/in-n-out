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


def load_competitors(csv_path="../shared/ca_fast_food_competitors.csv"):
    """Load all CA fast food competitors from the shared CSV."""
    with open(csv_path, newline="") as f:
        comps = list(csv.DictReader(f))
    for c in comps:
        c["lat"] = float(c["lat"])
        c["lon"] = float(c["lon"])
    return comps


def add_competitor_distances(input_csv="densitytest.csv", output_csv="densitytest.csv"):
    """Add nearest competitor distance columns, only considering competitors built before each In-N-Out."""
    comps = load_competitors()
    print(f"Loaded {len(comps)} competitors from shared CSV")

    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        # Get In-N-Out build date
        date = row.get("start_date", "").strip().split("T")[0].split(" ")[0]
        if not date or not date[0].isdigit():
            date = row.get("osm_first_seen", "").strip()

        lat = float(row["lat"])
        lon = float(row["lon"])

        # Only consider competitors created before this In-N-Out
        distances = []
        for c in comps:
            if date and c["created"] < date:
                dist = haversine(lat, lon, c["lat"], c["lon"])
                distances.append(dist)

        distances.sort()
        top5 = distances[:5]

        row["nearest_competitor_km"] = round(top5[0], 3) if top5 else ""
        row["avg_nearest_5_competitors_km"] = round(sum(top5) / len(top5), 3) if top5 else ""

        print(f"  [{i+1}/{len(rows)}] {row.get('city', '?')} - nearest: {row['nearest_competitor_km']}")

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows with competitor distances to {output_csv}")


if __name__ == "__main__":
    add_nearest_prior_distance()
    add_competitor_distances()
