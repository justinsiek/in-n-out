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

    print(f"Dataset: {len(df)} rows ({df[TARGET].sum():.0f} positive, {(df[TARGET] == 0).sum():.0f} negative)")
    print(f"Missing values per feature:\n{df[FEATURES].isnull().sum().to_string()}\n")

    return df


# ── Train & Evaluate ────────────────────────────────────────────────────────

def main():
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    # 85/15 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Scale positive weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}\n")

    # XGBoost params (without early stopping for CV compatibility)
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

    # ── Stratified 5-Fold CV on training set ────────────────────────────────
    cv_model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Per fold: {[f'{s:.4f}' for s in cv_scores]}\n")

    # ── Train final model with early stopping on test set ───────────────────
    model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluate on held-out test set ───────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    print(f"Test AUC: {auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "In-N-Out"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Feature Importance ──────────────────────────────────────────────────
    importance = model.feature_importances_
    feat_imp = sorted(zip(FEATURES, importance), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance:")
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 50)
        print(f"  {feat:35s} {imp:.4f} {bar}")

    # ── Save model ──────────────────────────────────────────────────────────
    model.save_model("model.json")
    print("\nModel saved to model.json")

    # ── Example: score a new location ───────────────────────────────────────
    print("\n── Example Predictions (test set, first 5) ──")
    sample = X_test.head()
    scores = model.predict_proba(sample)[:, 1]
    for i, (idx, row) in enumerate(sample.iterrows()):
        city = df.loc[idx, "city"] or "?"
        actual = y_test.loc[idx]
        print(f"  {city:20s}  score: {scores[i]:.3f}  actual: {actual}")


if __name__ == "__main__":
    main()
