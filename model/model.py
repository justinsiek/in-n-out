import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from scipy.special import expit  # sigmoid for normalizing ranker scores

# ── Config ──────────────────────────────────────────────────────────────────

DATA_PATH = "../finaldataset/final_dataset.csv"

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

TARGET = "label"

# ── Load & Prep ─────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop join artifact
    df = df.drop(columns=["index_right"], errors="ignore")

    # Convert features to numeric, coerce errors to NaN
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows in water/special-use census tracts (0 population + 0 income)
    # These are edge cases (ports, airports) that teach the model 0 pop is OK
    bad_mask = (df["resident_pop"] == 0) & (df["median_income"] == 0)
    if bad_mask.sum() > 0:
        print(f"Dropping {bad_mask.sum()} rows with 0 population + 0 income (water/special tracts)")
        df = df[~bad_mask].reset_index(drop=True)

    print(f"Dataset: {len(df)} rows ({df[TARGET].sum():.0f} positive, {(df[TARGET] == 0).sum():.0f} negative)")
    print(f"Missing values per feature:\n{df[FEATURES].isnull().sum().to_string()}\n")

    return df


# ── Train & Evaluate ────────────────────────────────────────────────────────

def run_model(df, features, label="All Features"):
    """Train and evaluate a model with the given feature set."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Features: {len(features)}")
    print(f"{'='*60}\n")

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}\n")

    params = dict(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="auc",
    )

    # Stratified 5-Fold CV
    cv_model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Per fold: {[f'{s:.4f}' for s in cv_scores]}\n")

    # Train final model
    model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    print(f"Test AUC: {auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "In-N-Out"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance:")
    for feat, imp in feat_imp:
        label = FEATURE_LABELS.get(feat, feat)
        bar = "█" * int(imp * 50)
        print(f"  {label:50s} {imp:.4f} {bar}")

    return model, auc


def build_groups(df):
    """
    Reconstruct group IDs from the rejected_locations.csv.
    Each In-N-Out + its rejected alternatives = one group.
    Returns a Series of group_id aligned to df's index.
    """
    rejected = pd.read_csv("../justin/rejected_locations.csv")

    # Round coords to avoid float precision issues
    rejected["rej_lat_r"] = rejected["lat"].round(5)
    rejected["rej_lon_r"] = rejected["lon"].round(5)
    rejected["ino_lat_r"] = rejected["source_ino_lat"].round(5)
    rejected["ino_lon_r"] = rejected["source_ino_lon"].round(5)

    # Map each unique In-N-Out to a group ID
    ino_coords = rejected[["ino_lat_r", "ino_lon_r"]].drop_duplicates().reset_index(drop=True)
    ino_coords["group_id"] = ino_coords.index

    # Map rejected lat/lon -> group_id
    rej_to_group = rejected.merge(ino_coords, on=["ino_lat_r", "ino_lon_r"])[
        ["rej_lat_r", "rej_lon_r", "group_id"]
    ].drop_duplicates(subset=["rej_lat_r", "rej_lon_r"])
    rej_lookup = {(r.rej_lat_r, r.rej_lon_r): r.group_id for r in rej_to_group.itertuples()}

    # Assign group IDs to final dataset
    group_ids = []
    next_new_group = len(ino_coords)
    for _, row in df.iterrows():
        if row[TARGET] == 1:
            # In-N-Out: find its group via lat/lon match to ino_coords
            lat_r = round(row["lat"], 5)
            lon_r = round(row["lon"], 5)
            match = ino_coords[(ino_coords["ino_lat_r"] == lat_r) & (ino_coords["ino_lon_r"] == lon_r)]
            if not match.empty:
                group_ids.append(int(match["group_id"].iloc[0]))
            else:
                group_ids.append(next_new_group)
                next_new_group += 1
        else:
            lat_r = round(row["lat"], 5)
            lon_r = round(row["lon"], 5)
            group_ids.append(rej_lookup.get((lat_r, lon_r), next_new_group))

    return pd.Series(group_ids, index=df.index)


def run_lgbm_ranker(df, features):
    """Train a LightGBM LambdaRank model using group structure."""
    print(f"\n{'='*60}")
    print(f"  LightGBM LambdaRank")
    print(f"  Features: {len(features)}")
    print(f"{'='*60}\n")

    df = df.copy()
    df["group_id"] = build_groups(df)

    # Split groups (not rows) to avoid leaking a group across train/test
    unique_groups = df["group_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    split = int(len(unique_groups) * 0.85)
    train_groups = set(unique_groups[:split])
    test_groups = set(unique_groups[split:])

    train_df = df[df["group_id"].isin(train_groups)]
    test_df = df[df["group_id"].isin(test_groups)]

    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_test = test_df[features]
    y_test = test_df[TARGET]

    # Group sizes for LightGBM ranker (must be sorted by group)
    train_df_sorted = train_df.sort_values("group_id")
    test_df_sorted = test_df.sort_values("group_id")
    train_sizes = train_df_sorted.groupby("group_id").size().values
    test_sizes = test_df_sorted.groupby("group_id").size().values

    X_train_s = train_df_sorted[features]
    y_train_s = train_df_sorted[TARGET]
    X_test_s = test_df_sorted[features]
    y_test_s = test_df_sorted[TARGET]

    print(f"Train groups: {len(train_sizes)} | Test groups: {len(test_sizes)}")
    print(f"Train rows: {len(X_train_s)} | Test rows: {len(X_test_s)}\n")

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5,
        random_state=42,
    )

    model.fit(
        X_train_s, y_train_s,
        group=train_sizes,
        eval_set=[(X_test_s, y_test_s)],
        eval_group=[test_sizes],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    # Score test set — sigmoid to normalize to 0-1
    raw_scores = model.predict(X_test_s)
    y_prob = expit(raw_scores)
    auc = roc_auc_score(y_test_s, y_prob)

    print(f"Test AUC: {auc:.4f}\n")

    # Feature importance
    feat_imp = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print("Feature Importance:")
    max_imp = feat_imp[0][1] if feat_imp else 1
    for feat, imp in feat_imp:
        label = FEATURE_LABELS.get(feat, feat)
        bar = "█" * int((imp / max_imp) * 30)
        print(f"  {label:50s} {imp:6.0f} {bar}")

    model.booster_.save_model("model_ranker.txt")
    print("\nRanker saved to model_ranker.txt")

    return model, auc


def main():
    df = load_data()

    # ── XGBoost classifier (all features) ──────────────────────────────────
    model_xgb, auc_xgb = run_model(df, FEATURES, label="XGBoost Classifier")

    # ── LightGBM LambdaRank ─────────────────────────────────────────────────
    model_lgbm, auc_lgbm = run_lgbm_ranker(df, FEATURES)

    # ── Comparison ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  XGBoost Classifier AUC:   {auc_xgb:.4f}")
    print(f"  LightGBM LambdaRank AUC:  {auc_lgbm:.4f}")
    winner = "LightGBM" if auc_lgbm > auc_xgb else "XGBoost"
    print(f"  Winner: {winner}")
    print()

    # Save XGBoost as default (probability output, easier for API)
    model_xgb.save_model("model.json")
    print("XGBoost model saved to model.json")


if __name__ == "__main__":
    main()
