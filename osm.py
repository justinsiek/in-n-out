import requests


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
    lat, lon = 34.0522, -118.2437

    # Present day
    result = get_competitor_density(lat, lon)
    print(f"Present: {result['count']} competitors within 3km")

    # Historical query
    result_2018 = get_competitor_density(lat, lon, date="2018-01-01")
    print(f"2018:    {result_2018['count']} competitors within 3km")
