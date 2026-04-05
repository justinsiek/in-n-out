import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point

# ---------------------------------------------------------
# Feature 1: Distance to Distribution Centers
# ---------------------------------------------------------

# Approximate coordinates for In-N-Out's distribution centers
# Locations: Baldwin Park CA, Lathrop CA, Lancaster TX, Draper UT, Colorado Springs CO, Goodyear AZ
DC_LOCATIONS = [
    (34.084, -117.961),  # Baldwin Park, CA
    (37.825, -121.288),  # Lathrop, CA
    (32.592, -96.769),   # Lancaster, TX
    (40.524, -111.863),  # Draper, UT
    (38.833, -104.821),  # Colorado Springs, CO
    (33.435, -112.357)   # Goodyear, AZ
]

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in miles.
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    
    # Radius of earth in miles is 3956
    mi = 3956 * c
    return mi

def get_distance_to_nearest_dc(lat, lon):
    """
    Returns the distance (in miles) to the closest In-N-Out Distribution Center.
    """
    distances = [haversine_distance(lat, lon, dc_lat, dc_lon) for dc_lat, dc_lon in DC_LOCATIONS]
    return min(distances)


# ---------------------------------------------------------
# Feature 2: Distance to Freeway Ramp (OSM) — bulk approach
# ---------------------------------------------------------

RAMPS_CACHE = Path(__file__).resolve().parent / "ca_freeway_ramps.geojson"


def fetch_all_ca_ramps():
    """One single Overpass query to get all freeway ramps in California."""
    import requests
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = """
    [out:json][timeout:120];
    area["name"="California"]["admin_level"="4"]->.ca;
    way[highway=motorway_link](area.ca);
    out center;
    """
    print("Fetching all CA freeway ramps from Overpass (single query)...")
    response = requests.post(overpass_url, data={"data": query})
    response.raise_for_status()
    elements = response.json().get("elements", [])

    # Convert to GeoDataFrame using center points
    points = []
    for el in elements:
        c = el.get("center")
        if c:
            points.append(Point(c["lon"], c["lat"]))
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    gdf.to_file(RAMPS_CACHE, driver="GeoJSON")
    print(f"  Cached {len(gdf)} ramp segments to {RAMPS_CACHE}")
    return gdf


def load_ramps():
    if RAMPS_CACHE.exists():
        print(f"Loading cached ramps from {RAMPS_CACHE}...")
        return gpd.read_file(RAMPS_CACHE)
    return fetch_all_ca_ramps()


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

def augment_dataframe_with_features(df, lat_col='latitude', lon_col='longitude'):
    """
    Takes a pandas DataFrame of potential locations and adds the engineered features.
    """
    print("Calculating distances to distribution centers...")
    df['dist_to_nearest_dc_miles'] = df.apply(
        lambda row: get_distance_to_nearest_dc(row[lat_col], row[lon_col]), axis=1
    )

    # Bulk freeway ramp distance
    ramps = load_ramps()
    ramps_proj = ramps.to_crs(epsg=3857)

    print("Calculating proximity to freeway ramps...")
    gdf_stores = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # Drop conflicting columns from prior joins if present
    for col in ['index_right', 'index_left']:
        if col in gdf_stores.columns:
            gdf_stores = gdf_stores.drop(columns=[col])

    joined = gpd.sjoin_nearest(
        gdf_stores, ramps_proj.reset_index(drop=True), how='left', distance_col='dist_to_freeway_ramp_meters'
    )
    # sjoin_nearest can duplicate rows if equidistant; keep first match per original index
    joined = joined[~joined.index.duplicated(keep='first')]
    df['dist_to_freeway_ramp_meters'] = joined['dist_to_freeway_ramp_meters'].values

    return df

# --- Main ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]
        df = pd.read_csv(target)
        df = augment_dataframe_with_features(df, lat_col='lat', lon_col='lon')
        df.to_csv(target, index=False)
        print(f"Wrote {len(df)} rows with DC/freeway features to {target}")
    else:
        test_data = {
            'lat': [34.0628, 38.5724],
            'lon': [-118.4476, -117.1518]
        }
        df = pd.DataFrame(test_data)
        df = augment_dataframe_with_features(df, lat_col='lat', lon_col='lon')
        print(df)