"""
Exports three CSVs for Omni:
  1. omni_irvine_heatmap.csv   — lat/lon grid with model scores
  2. omni_feature_importance.csv — feature names and importance values
  3. omni_model_evaluation.csv  — real In-N-Out vs rejected sites with scores
"""

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import (
    compute_features, FEATURES, FEATURE_LABELS,
    _load_innouts, _load_competitors, _load_ramps, _load_aadt, _load_lodes
)

MODEL_PATH = Path(__file__).resolve().parent / "model_ranker.txt"
DATA_PATH = Path(__file__).resolve().parent.parent / "finaldataset" / "final_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "omni_exports"

LAT_MIN, LAT_MAX = 33.63, 33.77
LON_MIN, LON_MAX = -117.90, -117.72
GRID_SIZE = 40


# ── Load model ───────────────────────────────────────────────────────────────

model = lgb.Booster(model_file=str(MODEL_PATH))

def score_point(lat, lon):
    try:
        feats = compute_features(lat, lon)
        census_fields = ["median_income", "resident_pop", "trade_area_population"]
        if all(feats.get(f, 0) == 0 for f in census_fields):
            return lat, lon, 0.0
        df = pd.DataFrame([feats])[FEATURES].apply(pd.to_numeric, errors="coerce")
        return lat, lon, float(expit(model.predict(df)[0]))
    except Exception as e:
        print(f"  Error ({lat:.4f}, {lon:.4f}): {e}")
        return lat, lon, None


# ── 1. Irvine heatmap scores ─────────────────────────────────────────────────

def export_heatmap():
    lats = np.linspace(LAT_MIN, LAT_MAX, GRID_SIZE)
    lons = np.linspace(LON_MIN, LON_MAX, GRID_SIZE)
    points = [(lat, lon) for lat in lats for lon in lons]

    print(f"Scoring {len(points)} Irvine grid points...")
    _load_innouts(); _load_competitors(); _load_ramps(); _load_aadt(); _load_lodes()

    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(score_point, lat, lon): (lat, lon) for lat, lon in points}
        done = 0
        for future in as_completed(futures):
            lat, lon, score = future.result()
            if score is not None:
                rows.append({"latitude": lat, "longitude": lon, "score": round(score, 4)})
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(points)}")

    df = pd.DataFrame(rows).sort_values(["latitude", "longitude"])
    path = OUTPUT_DIR / "omni_irvine_heatmap.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}\n")


# ── 2. Feature importance ────────────────────────────────────────────────────

def export_feature_importance():
    importances = model.feature_importance(importance_type="split")
    total = importances.sum()
    rows = []
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True):
        rows.append({
            "feature": FEATURE_LABELS.get(feat, feat),
            "feature_key": feat,
            "importance": round(float(imp), 2),
            "importance_pct": round(float(imp / total) * 100, 1),
        })
    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / "omni_feature_importance.csv"
    df.to_csv(path, index=False)
    print(f"  Saved feature importance → {path}\n")


# ── 3. Model evaluation (real vs rejected) ────────────────────────────────────

def export_model_evaluation():
    df = pd.read_csv(DATA_PATH)

    # Drop bad rows
    df = df.drop(columns=["index_right"], errors="ignore")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_mask = (df["resident_pop"] == 0) & (df["median_income"] == 0)
    df = df[~bad_mask].reset_index(drop=True)

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    raw_scores = model.predict(X)
    df["predicted_score"] = [round(float(expit(s)), 4) for s in raw_scores]
    df["site_type"] = df["label"].map({1: "Real In-N-Out", 0: "Rejected Site"})

    out = df[["site_type", "lat", "lon", "city", "predicted_score"] +
             [f for f in FEATURES if f in df.columns]]
    path = OUTPUT_DIR / "omni_model_evaluation.csv"
    out.to_csv(path, index=False)
    print(f"  Saved {len(out)} rows → {path}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("=== Exporting for Omni ===\n")

    print("1. Feature importance...")
    export_feature_importance()

    print("2. Model evaluation (real vs rejected)...")
    export_model_evaluation()

    print("3. Irvine heatmap scores (this takes a few minutes)...")
    export_heatmap()

    print(f"\nDone! Upload these 3 files to Omni:")
    for f in sorted(OUTPUT_DIR.glob("omni_*.csv")):
        print(f"  {f.name}")
