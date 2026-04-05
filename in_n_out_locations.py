import requests
import csv
import time


def get_in_n_out_california():
    """Query all In-N-Out locations within California from OSM."""
    overpass_url = "https://overpass-api.de/api/interpreter"

    query = """
    [out:json][timeout:60];
    area["name"="California"]["admin_level"="4"]->.ca;
    (
      node[brand~"In.N.Out"](area.ca);
      way[brand~"In.N.Out"](area.ca);
    );
    out center meta;
    """

    response = requests.post(overpass_url, data={"data": query})
    response.raise_for_status()
    return response.json().get("elements", [])


def get_first_seen_date(element_type, element_id):
    """Fetch the creation date of an OSM element via its edit history."""
    url = f"https://api.openstreetmap.org/api/0.6/{element_type}/{element_id}/history.json"
    response = requests.get(url)
    response.raise_for_status()
    versions = response.json().get("elements", [])
    if versions:
        return versions[0].get("timestamp", "")[:10]  # first version date
    return ""


def main():
    print("Querying In-N-Out locations in California...")
    elements = get_in_n_out_california()
    print(f"Found {len(elements)} locations. Fetching creation dates...")

    rows = []
    for i, el in enumerate(elements):
        tags = el.get("tags", {})
        el_type = el["type"]
        el_id = el["id"]

        # Get coordinates (ways use 'center' from 'out center')
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")

        # Check for explicit date tags
        start_date = tags.get("start_date", "")
        opening_date = tags.get("opening_date", "")

        # Fetch when the element was first added to OSM
        first_seen = get_first_seen_date(el_type, el_id)
        time.sleep(0.05)  # respect OSM API rate limits

        rows.append({
            "name": tags.get("name", "In-N-Out Burger"),
            "lat": lat,
            "lon": lon,
            "address": tags.get("addr:full", ""),
            "street": tags.get("addr:street", ""),
            "housenumber": tags.get("addr:housenumber", ""),
            "city": tags.get("addr:city", ""),
            "state": tags.get("addr:state", ""),
            "postcode": tags.get("addr:postcode", ""),
            "start_date": start_date,
            "opening_date": opening_date,
            "osm_first_seen": first_seen,
            "osm_type": el_type,
            "osm_id": el_id,
        })

        print(f"  [{i+1}/{len(elements)}] {rows[-1]['name']} - {rows[-1]['city'] or 'unknown city'} (first seen: {first_seen})")

    # Write CSV
    output = "in_n_out_california.csv"
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} locations to {output}")


if __name__ == "__main__":
    main()
