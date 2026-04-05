import requests
import csv
import time
import math
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_all_ca_competitors():
    """One single query to get ALL fast food competitors in California."""
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = """
    [out:json][timeout:120];
    area["name"="California"]["admin_level"="4"]->.ca;
    (
      node[amenity=fast_food][brand!~"In.N.Out"](area.ca);
      way[amenity=fast_food][brand!~"In.N.Out"](area.ca);
    );
    out center body;
    """
    print("Fetching all fast food competitors in California (single query)...")
    response = requests.post(overpass_url, data={"data": query})
    response.raise_for_status()
    elements = response.json().get("elements", [])
    print(f"  Got {len(elements)} competitors")
    return elements


def fetch_creation_date(el_type, el_id):
    """Get the creation date of an OSM element from its history."""
    url = f"https://api.openstreetmap.org/api/0.6/{el_type}/{el_id}/history.json"
    response = requests.get(url)
    response.raise_for_status()
    versions = response.json().get("elements", [])
    if versions:
        return versions[0].get("timestamp", "")[:10]
    return ""


def add_9_months(date_str):
    if len(date_str) == 4:
        date_str = f"{date_str}-01-01"
    elif len(date_str) == 7:
        date_str = f"{date_str}-01"
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    month = dt.month + 9
    year = dt.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(dt.day, 28)
    return datetime(year, month, day).strftime("%Y-%m-%d")


def main():
    comp_csv = "../shared/ca_fast_food_competitors.csv"

    if os.path.exists(comp_csv):
        # Load cached competitors
        print(f"Loading competitors from {comp_csv}...")
        with open(comp_csv, newline="") as f:
            competitors = list(csv.DictReader(f))
        for comp in competitors:
            comp["lat"] = float(comp["lat"])
            comp["lon"] = float(comp["lon"])
        print(f"  Loaded {len(competitors)} competitors")
    else:
        # Step 1: One Overpass query for all CA competitors
        elements = fetch_all_ca_competitors()

        # Step 2: Fetch creation dates from OSM API (concurrent)
        # Prepare elements with coordinates
        valid_elements = []
        for el in elements:
            tags = el.get("tags", {})
            comp_lat = el.get("lat") or el.get("center", {}).get("lat")
            comp_lon = el.get("lon") or el.get("center", {}).get("lon")
            if comp_lat is not None and comp_lon is not None:
                valid_elements.append((el, tags, comp_lat, comp_lon))

        def fetch_one(item):
            el, tags, comp_lat, comp_lon = item
            created = fetch_creation_date(el["type"], el["id"])
            if not created:
                return None
            return {
                "name": tags.get("name", tags.get("brand", "Unknown")),
                "lat": comp_lat,
                "lon": comp_lon,
                "city": tags.get("addr:city", ""),
                "street": tags.get("addr:street", ""),
                "housenumber": tags.get("addr:housenumber", ""),
                "postcode": tags.get("addr:postcode", ""),
                "created": created,
            }

        competitors = []
        done = 0
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(fetch_one, item): item for item in valid_elements}
            for future in as_completed(futures):
                done += 1
                result = future.result()
                if result:
                    competitors.append(result)
                if done % 500 == 0:
                    print(f"  Fetched creation dates: {done}/{len(valid_elements)}")

        print(f"  Got creation dates for {len(competitors)} competitors")

        # Save to shared CSV
        with open(comp_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=competitors[0].keys())
            writer.writeheader()
            writer.writerows(competitors)
        print(f"  Saved all competitors to {comp_csv}")

    # Step 3: Read In-N-Outs and match
    with open("densitytest.csv", newline="") as f:
        in_n_outs = list(csv.DictReader(f))

    rejected_rows = []

    for i, ino in enumerate(in_n_outs):
        date = ino.get("start_date", "").strip().split("T")[0].split(" ")[0]
        if not date or not date[0].isdigit():
            date = ino.get("osm_first_seen", "").strip()
        if not date or not date[0].isdigit():
            continue

        date_to = add_9_months(date)
        lat = float(ino["lat"])
        lon = float(ino["lon"])
        city = ino.get("city", "?")

        # Find competitors created in the 9-month window, sorted by distance
        candidates = []
        for comp in competitors:
            if date <= comp["created"] <= date_to:
                dist = haversine(lat, lon, comp["lat"], comp["lon"])
                candidates.append((dist, comp))

        candidates.sort(key=lambda x: x[0])
        picked = candidates[:5]

        for dist, comp in picked:
            rejected_rows.append({
                "source_ino_city": city,
                "source_ino_lat": lat,
                "source_ino_lon": lon,
                "source_ino_date": date,
                "name": comp["name"],
                "lat": comp["lat"],
                "lon": comp["lon"],
                "city": comp["city"],
                "street": comp["street"],
                "housenumber": comp["housenumber"],
                "postcode": comp["postcode"],
                "competitor_created": comp["created"],
                "distance_km": round(dist, 3),
            })

        print(f"[{i+1}/{len(in_n_outs)}] {city} - picked {len(picked)} (window: {date} -> {date_to})")

    output = "rejected_locations.csv"
    if rejected_rows:
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rejected_rows[0].keys())
            writer.writeheader()
            writer.writerows(rejected_rows)

    print(f"\nWrote {len(rejected_rows)} rejected locations to {output}")


if __name__ == "__main__":
    main()
