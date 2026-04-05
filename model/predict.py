import math
import csv
import json
import urllib.request
import geopandas as gpd
import pandas as pd
import lightgbm as lgb
from scipy.special import expit
from pathlib import Path
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent

# ── Paths to cached data ────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).resolve().parent / "model_ranker.txt"
INNOUT_CSV = ROOT / "shared" / "in_n_out_california.csv"
COMPETITORS_CSV = ROOT / "shared" / "ca_fast_food_competitors.csv"
RAMPS_GEOJSON = ROOT / "max" / "ca_freeway_ramps.geojson"
AADT_GEOJSON = ROOT / "max" / "caltrans_aadt.geojson"
WAC_CACHE = ROOT / "austin" / "ca_wac.csv.gz"
RAC_CACHE = ROOT / "austin" / "ca_rac.csv.gz"

# ── Distribution centers ────────────────────────────────────────────────────

DC_LOCATIONS = [
    (34.084, -117.961),   # Baldwin Park, CA
    (37.825, -121.288),   # Lathrop, CA
    (32.592, -96.769),    # Lancaster, TX
    (40.524, -111.863),   # Draper, UT
    (38.833, -104.821),   # Colorado Springs, CO
    (33.435, -112.357),   # Goodyear, AZ
]

FEATURES = [
    "dist_to_nearest_prior_km",
    "nearest_competitor_km",
    "avg_nearest_5_competitors_km",
    "median_income",
    "resident_pop",
    "workers_in_tract",
    "workers_from_tract",
    "daytime_pop",
    "hwy_aadt",
    "dist_to_highway_meters",
    "dist_to_nearest_dc_miles",
    "dist_to_freeway_ramp_meters",
    "trade_area_population",
]

FEATURE_LABELS = {
    "dist_to_nearest_prior_km":       "Distance To Nearest In-N-Out (km)",
    "nearest_competitor_km":          "Distance To Nearest Competitor (km)",
    "avg_nearest_5_competitors_km":   "Avg Distance To 5 Nearest Competitors (km)",
    "median_income":                  "Median Area Income ($)",
    "resident_pop":                   "Resident Population",
    "workers_in_tract":               "Workers Employed In Area",
    "workers_from_tract":             "Workers Living In Area",
    "daytime_pop":                    "Daytime Population",
    "hwy_aadt":                       "Annual Average Daily Traffic (Highway)",
    "dist_to_highway_meters":         "Distance To Highway (m)",
    "dist_to_nearest_dc_miles":       "Distance To Nearest Distribution Center (miles)",
    "dist_to_freeway_ramp_meters":    "Distance To Freeway Ramp (m)",
    "trade_area_population":          "Trade Area Population",
}

ACS_YEAR = 2022

# ── Helpers ─────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_miles(lat1, lon1, lat2, lon2):
    return haversine_km(lat1, lon1, lat2, lon2) * 0.621371


# ── Lazy-loaded caches ──────────────────────────────────────────────────────

_cache = {}


def _load_innouts():
    if "innouts" not in _cache:
        with open(INNOUT_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        _cache["innouts"] = [(float(r["lat"]), float(r["lon"])) for r in rows]
    return _cache["innouts"]


def _load_competitors():
    if "competitors" not in _cache:
        with open(COMPETITORS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        _cache["competitors"] = [(float(r["lat"]), float(r["lon"])) for r in rows]
    return _cache["competitors"]


def _load_ramps():
    if "ramps" not in _cache:
        gdf = gpd.read_file(RAMPS_GEOJSON).to_crs(epsg=3857)
        _cache["ramps"] = gdf
    return _cache["ramps"]


def _load_aadt():
    if "aadt" not in _cache:
        gdf = gpd.read_file(AADT_GEOJSON)
        # Normalize AADT column
        if "AADT" not in gdf.columns:
            if "BACK_AADT" in gdf.columns and "AHEAD_AADT" in gdf.columns:
                gdf["AADT"] = gdf[["BACK_AADT", "AHEAD_AADT"]].max(axis=1)
            elif "BACK_AADT" in gdf.columns:
                gdf["AADT"] = gdf["BACK_AADT"]
            elif "AHEAD_AADT" in gdf.columns:
                gdf["AADT"] = gdf["AHEAD_AADT"]
        _cache["aadt"] = gdf[["AADT", "geometry"]].to_crs(epsg=3857)
    return _cache["aadt"]


def _load_lodes():
    if "wac" not in _cache:
        import gzip
        wac = {}
        with gzip.open(WAC_CACHE, "rt") as f:
            for row in csv.DictReader(f):
                tract = row["w_geocode"].zfill(15)[:11]
                wac[tract] = wac.get(tract, 0) + int(row["C000"])
        _cache["wac"] = wac

        rac = {}
        with gzip.open(RAC_CACHE, "rt") as f:
            for row in csv.DictReader(f):
                tract = row["h_geocode"].zfill(15)[:11]
                rac[tract] = rac.get(tract, 0) + int(row["C000"])
        _cache["rac"] = rac
    return _cache["wac"], _cache["rac"]


# ── Feature computation ────────────────────────────────────────────────────

def _census_geocode(lat, lon):
    url = (
        "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
        f"?x={lon}&y={lat}"
        "&benchmark=Public_AR_Current&vintage=Current_Current&format=json"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        tracts = data["result"]["geographies"].get("Census Tracts", [])
        if not tracts:
            return None
        t = tracts[0]
        return {"state": t["STATE"], "county": t["COUNTY"], "tract": t["TRACT"],
                "fips": t["STATE"].zfill(2) + t["COUNTY"].zfill(3) + t["TRACT"].zfill(6)}
    except Exception:
        return None


def _acs_value(state, county, tract, variable):
    url = (
        f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
        f"?get={variable}&for=tract:{tract}&in=state:{state}%20county:{county}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        val = int(data[1][0])
        return val if val >= 0 else 0
    except Exception:
        return 0


def compute_features(lat, lon):
    """Compute all 13 features for a given coordinate. Returns a dict."""
    features = {}

    # 1. dist_to_nearest_prior_km (distance to nearest existing In-N-Out)
    innouts = _load_innouts()
    dists = [haversine_km(lat, lon, ila, ilo) for ila, ilo in innouts]
    features["dist_to_nearest_prior_km"] = min(dists)

    # 2-3. Competitor distances
    competitors = _load_competitors()
    comp_dists = sorted(haversine_km(lat, lon, cla, clo) for cla, clo in competitors)
    top5 = comp_dists[:5]
    features["nearest_competitor_km"] = top5[0] if top5 else 0
    features["avg_nearest_5_competitors_km"] = sum(top5) / len(top5) if top5 else 0

    # 4-8. Census features (median income, population, workers)
    print("  Querying Census API...")
    geo = _census_geocode(lat, lon)
    if geo:
        features["median_income"] = _acs_value(geo["state"], geo["county"], geo["tract"], "B19013_001E")
        resident_pop = _acs_value(geo["state"], geo["county"], geo["tract"], "B01003_001E")
        features["resident_pop"] = resident_pop

        wac, rac = _load_lodes()
        workers_in = wac.get(geo["fips"], 0)
        workers_from = rac.get(geo["fips"], 0)
        features["workers_in_tract"] = workers_in
        features["workers_from_tract"] = workers_from
        features["daytime_pop"] = resident_pop - workers_from + workers_in
        features["trade_area_population"] = resident_pop
    else:
        for key in ["median_income", "resident_pop", "workers_in_tract",
                     "workers_from_tract", "daytime_pop", "trade_area_population"]:
            features[key] = 0

    # 9-10. Highway AADT
    print("  Computing highway distance...")
    aadt_gdf = _load_aadt()
    point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
    joined = gpd.sjoin_nearest(point, aadt_gdf.reset_index(drop=True), how="left", distance_col="dist")
    features["hwy_aadt"] = joined["AADT"].iloc[0] if not pd.isna(joined["AADT"].iloc[0]) else 0
    features["dist_to_highway_meters"] = joined["dist"].iloc[0]

    # 11. Distance to nearest distribution center
    features["dist_to_nearest_dc_miles"] = min(
        haversine_miles(lat, lon, dcla, dclo) for dcla, dclo in DC_LOCATIONS
    )

    # 12. Distance to nearest freeway ramp
    print("  Computing freeway ramp distance...")
    ramps = _load_ramps()
    point_proj = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
    ramp_joined = gpd.sjoin_nearest(point_proj, ramps.reset_index(drop=True), how="left", distance_col="dist")
    features["dist_to_freeway_ramp_meters"] = ramp_joined["dist"].iloc[0]

    return features


# ── Predictor ───────────────────────────────────────────────────────────────

_model = None

def predict(lat, lon):
    """
    Predict the probability that In-N-Out would build at (lat, lon).
    Returns a float between 0 and 1.
    """
    global _model
    if _model is None:
        _model = lgb.Booster(model_file=str(MODEL_PATH))

    print(f"Computing features for ({lat}, {lon})...")
    feats = compute_features(lat, lon)

    # Guard: if no census data, location is uninhabitable / outside US
    census_fields = ["median_income", "resident_pop", "trade_area_population"]
    if all(feats.get(f, 0) == 0 for f in census_fields):
        print("  No census data — location is outside US or uninhabitable.")
        return 0.0, feats

    df = pd.DataFrame([feats])[FEATURES].apply(pd.to_numeric, errors="coerce")
    raw_score = _model.predict(df)[0]
    prob = float(expit(raw_score))

    return prob, feats


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python predict.py <lat> <lon>")
        print("Example: python predict.py 34.0522 -118.2437")
        sys.exit(1)

    lat = float(sys.argv[1])
    lon = float(sys.argv[2])

    prob, feats = predict(lat, lon)

    print(f"\n{'='*50}")
    print(f"  In-N-Out Probability: {prob:.1%}")
    print(f"{'='*50}")
    print(f"\nFeatures:")
    for k, v in feats.items():
        label = FEATURE_LABELS.get(k, k)
        print(f"  {label:50s} {v}")
