"""
Train script for dropout prediction. Reads CSV, runs cross-validation, trains a RandomForest pipeline and saves model to ml/models/model.joblib.
"""
import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml.pipeline import GPATrendTransformer

def build_pipeline(numeric_cols, categorical_cols):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="passthrough")

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ("gpa_features", GPATrendTransformer()),
        ("preprocess", preprocessor),
        ("clf", model),
    ])
    return pipeline

def main(args):
    df = pd.read_csv(args.data)
    target = "dropout_label"
    sem_cols = [f"sem{i}_gpa" for i in range(1,9)]
    numeric_cols = [
        "cgpa", "attendance_percent", "family_income", "distance_km",
        "extracurricular_count", "outstanding_fees_amount"
    ] + sem_cols
    categorical_cols = ["college", "scholarship_flag", "fees_paid_current", "hostel_resident", "internet_access_home"]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    selected = [c for c in set(numeric_cols + categorical_cols + [target]) if c in df.columns]
    df = df[selected]

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    pipeline = build_pipeline(numeric_cols=[c for c in numeric_cols if c in df.columns],
                              categorical_cols=[c for c in categorical_cols if c in df.columns])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["roc_auc", "accuracy", "f1"]

    print("Running cross-validation (may take a few minutes on large datasets)...")
    try:
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
        print("Cross-validation results:")
        for k, v in scores.items():
            print(f"  {k}: mean={v.mean():.4f} std={v.std():.4f}")
    except Exception as e:
        print("Cross-validation failed:", e)

    pipeline.fit(X, y)
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = Path(args.output_dir) / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved trained pipeline to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dropout prediction model")
    parser.add_argument("--data", type=str, default="ml/sample_data/students.csv", help="Path to CSV data")
    parser.add_argument("--output-dir", type=str, default="ml/models", help="Directory to write model")
    args = parser.parse_args()
    main(args)
