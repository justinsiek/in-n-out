import sys
import numpy as np
import pandas as pd
import folium
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add model dir to path so we can import predict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from predict import compute_features, predict, _cache, FEATURES, FEATURE_LABELS

import lightgbm as lgb
from scipy.special import expit
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model_ranker.txt"

# ── Irvine bounding box ──────────────────────────────────────────────────────

LAT_MIN, LAT_MAX = 33.63, 33.77
LON_MIN, LON_MAX = -117.90, -117.72

GRID_SIZE = 40  # 40x40 = 1600 points

# ── Score a single point ─────────────────────────────────────────────────────

_model = None

def score_point(lat, lon):
    global _model
    if _model is None:
        _model = lgb.Booster(model_file=str(MODEL_PATH))

    try:
        feats = compute_features(lat, lon)

        census_fields = ["median_income", "resident_pop", "trade_area_population"]
        if all(feats.get(f, 0) == 0 for f in census_fields):
            return lat, lon, 0.0

        df = pd.DataFrame([feats])[FEATURES].apply(pd.to_numeric, errors="coerce")
        raw = _model.predict(df)[0]
        return lat, lon, float(expit(raw))
    except Exception as e:
        print(f"  Error at ({lat:.4f}, {lon:.4f}): {e}")
        return lat, lon, None


# ── Build grid ───────────────────────────────────────────────────────────────

def main():
    lats = np.linspace(LAT_MIN, LAT_MAX, GRID_SIZE)
    lons = np.linspace(LON_MIN, LON_MAX, GRID_SIZE)
    points = [(lat, lon) for lat in lats for lon in lons]

    print(f"Scoring {len(points)} grid points over Irvine ({GRID_SIZE}x{GRID_SIZE})...")
    print("Preloading cached data...")

    # Preload heavy files once before threading
    from predict import _load_innouts, _load_competitors, _load_ramps, _load_aadt, _load_lodes
    _load_innouts()
    _load_competitors()
    _load_ramps()
    _load_aadt()
    _load_lodes()
    print("  Cached data loaded.\n")

    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(score_point, lat, lon): (lat, lon) for lat, lon in points}
        for future in as_completed(futures):
            lat, lon, score = future.result()
            results[(lat, lon)] = score
            done += 1
            if done % 50 == 0:
                print(f"  Scored {done}/{len(points)}")

    print(f"\nDone. Building map...")

    # ── Build folium map ─────────────────────────────────────────────────────

    center_lat = (LAT_MIN + LAT_MAX) / 2
    center_lon = (LON_MIN + LON_MAX) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    dlat = (LAT_MAX - LAT_MIN) / GRID_SIZE
    dlon = (LON_MAX - LON_MIN) / GRID_SIZE

    def score_to_color(score):
        if score is None:
            return "#aaaaaa", 0.0
        # red (0) -> orange (0.4) -> yellow (0.6) -> green (1.0)
        if score < 0.5:
            r = 255
            g = int(255 * (score / 0.5))
        else:
            r = int(255 * (1 - (score - 0.5) / 0.5))
            g = 255
        return f"#{r:02x}{g:02x}00", 0.5

    for (lat, lon), score in results.items():
        color, opacity = score_to_color(score)
        folium.Rectangle(
            bounds=[[lat - dlat / 2, lon - dlon / 2], [lat + dlat / 2, lon + dlon / 2]],
            color=None,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=f"({lat:.4f}, {lon:.4f})<br>Probability: {score:.1%}" if score is not None else "N/A",
        ).add_to(m)

    # Add existing In-N-Out locations as markers
    import csv
    ino_csv = Path(__file__).resolve().parent.parent / "shared" / "in_n_out_california.csv"
    with open(ino_csv, newline="") as f:
        for row in csv.DictReader(f):
            rlat, rlon = float(row["lat"]), float(row["lon"])
            if LAT_MIN <= rlat <= LAT_MAX and LON_MIN <= rlon <= LON_MAX:
                folium.Marker(
                    location=[rlat, rlon],
                    tooltip=row.get("city", "In-N-Out"),
                    icon=folium.Icon(color="red", icon="cutlery", prefix="fa"),
                ).add_to(m)

    output = "irvine_heatmap.html"
    m.save(output)
    print(f"Map saved to {output}")
    print("Open it in your browser.")


if __name__ == "__main__":
    main()
