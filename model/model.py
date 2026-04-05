import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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
        bar = "█" * int(imp * 50)
        print(f"  {feat:35s} {imp:.4f} {bar}")

    return model, auc


def main():
    df = load_data()

    # ── Run 1: All features ─────────────────────────────────────────────────
    model_full, auc_full = run_model(df, FEATURES, label="All Features")

    # ── Run 2: Without dist_to_nearest_prior_km ─────────────────────────────
    features_no_prior = [f for f in FEATURES if f != "dist_to_nearest_prior_km"]
    model_no_prior, auc_no_prior = run_model(df, features_no_prior, label="Without dist_to_nearest_prior_km")

    # ── Comparison ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  All features AUC:              {auc_full:.4f}")
    print(f"  Without nearest_prior AUC:     {auc_no_prior:.4f}")
    print(f"  Difference:                    {auc_full - auc_no_prior:+.4f}")
    print()

    # Save the full model
    model_full.save_model("model.json")
    print("Full model saved to model.json")

    # Example predictions
    X_test = df[FEATURES].iloc[:5]
    scores = model_full.predict_proba(X_test)[:, 1]
    print("\n── Example Predictions (first 5) ──")
    for i, (idx, row) in enumerate(X_test.iterrows()):
        city = df.loc[idx, "city"] or "?"
        actual = df.loc[idx, TARGET]
        print(f"  {city:20s}  score: {scores[i]:.3f}  actual: {actual}")


if __name__ == "__main__":
    main()
