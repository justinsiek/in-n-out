import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np

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
# Feature 2: Distance to Freeway Ramp (OSM)
# ---------------------------------------------------------

def get_distance_to_nearest_motorway_link(lat, lon, search_radius_meters=3000):
    """
    Queries OpenStreetMap for freeway ramps (motorway_link) within a given radius
    and returns the distance in meters to the closest one.
    """
    target_point = (lat, lon)
    
    try:
        # Query OSM for motorway links (on-ramps/off-ramps) within the radius
        tags = {'highway': 'motorway_link'}
        
        # Note: using the modern OSMnx features API
        ramps = ox.features.features_from_point(target_point, tags=tags, dist=search_radius_meters)
        
        if ramps.empty:
            return search_radius_meters # Cap distance at max radius if none found
            
        # Create a GeoSeries of our target point
        point_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
        
        # Project both the point and the ramps to a local UTM CRS to calculate accurate distance in meters
        # OSMnx has a handy built-in projector
        point_proj = ox.projection.project_gdf(gpd.GeoDataFrame(geometry=point_geom))
        ramps_proj = ox.projection.project_gdf(ramps)
        
        # Calculate distance from the point to all ramps and grab the minimum
        # .distance returns a Series of distances, .min() gets the smallest
        min_distance = ramps_proj.distance(point_proj.geometry[0]).min()
        
        return min_distance

    except Exception as e:
        # If no ramps are found in the radius, OSMnx might throw an EmptyOverpassResponse exception
        # We handle this by returning the maximum search radius
        return search_radius_meters


# ---------------------------------------------------------
# Example Usage / Pipeline Integration
# ---------------------------------------------------------

def augment_dataframe_with_features(df, lat_col='latitude', lon_col='longitude'):
    """
    Takes a pandas DataFrame of potential locations and adds the engineered features.
    """
    print("Calculating distances to distribution centers...")
    df['dist_to_nearest_dc_miles'] = df.apply(
        lambda row: get_distance_to_nearest_dc(row[lat_col], row[lon_col]), axis=1
    )
    
    print("Calculating proximity to freeway ramps... (This may take a moment due to API calls)")
    df['dist_to_freeway_ramp_meters'] = df.apply(
        lambda row: get_distance_to_nearest_motorway_link(row[lat_col], row[lon_col]), axis=1
    )
    
    return df

# --- Test the code ---
if __name__ == "__main__":
    # Example coordinates: Let's test a known In-N-Out location (Westwood, LA)
    # and a random remote location (Middle of nowhere, Nevada)
    test_data = {
        'location_name': ['Westwood In-N-Out', 'Remote Desert'],
        'latitude': [34.0628, 38.5724],
        'longitude': [-118.4476, -117.1518]
    }
    
    df = pd.DataFrame(test_data)
    df = augment_dataframe_with_features(df)
    print("\nResults:")
    print(df[['location_name', 'dist_to_nearest_dc_miles', 'dist_to_freeway_ramp_meters']])