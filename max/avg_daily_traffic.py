import argparse
import json
import subprocess
import urllib.parse
from pathlib import Path

import pandas as pd
import geopandas as gpd

CALTRANS_AADT_QUERY = (
    "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/CHhighway/"
    "Traffic_AADT/FeatureServer/0/query"
)


def fetch_caltrans_aadt_geojson(dest: Path, page_size: int = 2000) -> None:
    """Download all features from the public Caltrans AADT FeatureServer as GeoJSON."""
    offset = 0
    features: list[dict] = []
    while True:
        qs = urllib.parse.urlencode(
            {
                "where": "1=1",
                "outFields": "*",
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }
        )
        url = f"{CALTRANS_AADT_QUERY}?{qs}"
        # Use curl so TLS uses the macOS trust store (python.org builds often lack certs).
        proc = subprocess.run(
            ["curl", "-sS", "-L", "--max-time", "120", url],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout)
        batch = payload.get("features") or []
        features.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    print(f"Saved {len(features)} AADT count stations to {dest}")


def _ensure_aadt_column(gdf_traffic: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Caltrans open data uses BACK_AADT / AHEAD_AADT; normalize to AADT for joins."""
    if "AADT" in gdf_traffic.columns:
        return gdf_traffic
    gdf = gdf_traffic.copy()
    if "BACK_AADT" in gdf.columns and "AHEAD_AADT" in gdf.columns:
        gdf["AADT"] = gdf[["BACK_AADT", "AHEAD_AADT"]].max(axis=1)
    elif "BACK_AADT" in gdf.columns:
        gdf["AADT"] = gdf["BACK_AADT"]
    elif "AHEAD_AADT" in gdf.columns:
        gdf["AADT"] = gdf["AHEAD_AADT"]
    else:
        raise ValueError(
            "Traffic layer needs 'AADT' or Caltrans fields 'BACK_AADT' / 'AHEAD_AADT'."
        )
    return gdf


def add_traffic_features(df, aadt_geojson_path):
    """
    df: Your DataFrame with 'lat' and 'lon'
    aadt_geojson_path: Path to the Caltrans AADT GeoJSON file
    """
    print("🚦 Loading Caltrans Traffic Data... (this might take a second)")
    
    # 1. Convert your In-N-Out list into a GeoDataFrame
    gdf_stores = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )

    # 2. Load traffic count stations (Caltrans publishes points, not road centerlines)
    gdf_traffic = gpd.read_file(aadt_geojson_path)
    gdf_traffic = _ensure_aadt_column(gdf_traffic)

    # 3. Project to a coordinate system that uses meters (Mercator) 
    # This makes "nearest" calculations much more accurate than degrees
    gdf_stores = gdf_stores.to_crs(epsg=3857)
    gdf_traffic = gdf_traffic.to_crs(epsg=3857)

    # 4. Spatial Join: Find the nearest traffic segment for each store
    # 'sjoin_nearest' finds the closest line to your point
    print("🔍 Snapping stores to nearest AADT count station...")
    joined_gdf = gpd.sjoin_nearest(
        gdf_stores, 
        gdf_traffic[['AADT', 'geometry']], 
        how='left', 
        distance_col="dist_to_highway_meters"
    )

    # 5. Cleanup
    # Rename 'AADT' to something clearer
    joined_gdf = joined_gdf.rename(columns={'AADT': 'hwy_aadt'})
    
    # Convert back to standard DataFrame and drop the geometry column
    final_df = pd.DataFrame(joined_gdf.drop(columns='geometry'))
    
    return final_df

# --- Example Usage ---
if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Snap store points to nearest Caltrans AADT highway segment."
    )
    parser.add_argument(
        "aadt_geojson",
        nargs="?",
        type=Path,
        default=_here / "caltrans_aadt.geojson",
        help=f"path to Caltrans AADT GeoJSON (default: {_here / 'caltrans_aadt.geojson'})",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download Caltrans AADT GeoJSON from the public FeatureServer into the path above",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="path to CSV to add traffic features to (reads and overwrites in place)",
    )
    args = parser.parse_args()
    aadt_path = args.aadt_geojson.expanduser().resolve()
    if args.fetch:
        fetch_caltrans_aadt_geojson(aadt_path)
    if not aadt_path.is_file():
        raise SystemExit(
            f"File not found: {aadt_path}\n\n"
            "Fetch it automatically:\n"
            "  python3 max/avg_daily_traffic.py --fetch\n"
            "Or pass a file you exported elsewhere:\n"
            "  python3 max/avg_daily_traffic.py /path/to/your_aadt.geojson"
        )

    if args.csv:
        df = pd.read_csv(args.csv)
        result = add_traffic_features(df, aadt_path)
        result.to_csv(args.csv, index=False)
        print(f"Wrote {len(result)} rows with traffic features to {args.csv}")
    else:
        data = {"lat": [33.6489], "lon": [-117.7426]}  # Near Irvine Spectrum
        test_df = pd.DataFrame(data)
        result = add_traffic_features(test_df, aadt_path)
        print(result)