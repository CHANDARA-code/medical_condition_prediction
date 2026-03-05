#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_training_sample(data_dir: Path, rows: int, seed: int) -> pd.DataFrame:
    files = sorted(data_dir.glob("symptom_disease_1m_part_*.csv"))
    if not files:
        raise FileNotFoundError(f"No generated CSV files found in {data_dir}")

    rng = np.random.default_rng(seed)
    per_file = int(np.ceil(rows / len(files)))
    sampled = []

    for f in files:
        df = pd.read_csv(f)
        take = min(per_file, len(df))
        idx = rng.choice(len(df), size=take, replace=False)
        sampled.append(df.iloc[idx])

    out = pd.concat(sampled, ignore_index=True)
    if len(out) > rows:
        out = out.sample(n=rows, random_state=seed)
    return out.reset_index(drop=True)


def inject_missing_values(df: pd.DataFrame, feature_cols: list[str], frac: float, seed: int) -> pd.DataFrame:
    if frac <= 0:
        return df
    rng = np.random.default_rng(seed)
    out = df.copy()

    n_rows = len(out)
    n_feats = len(feature_cols)
    n_missing = int(n_rows * n_feats * frac)
    row_idx = rng.integers(0, n_rows, size=n_missing)
    col_idx = rng.integers(0, n_feats, size=n_missing)

    arr = out[feature_cols].to_numpy(dtype=float).copy()
    arr[row_idx, col_idx] = np.nan
    out[feature_cols] = arr
    return out


def make_pipeline(model_name: str):
    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=300,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_name == "knn":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", MinMaxScaler()),
                ("model", KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)),
            ]
        )
    if model_name == "decision_tree":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=120,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    if model_name == "gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "model",
                    GradientBoostingClassifier(random_state=42, n_estimators=80),
                ),
            ]
        )
    if model_name == "gaussian_nb":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("model", GaussianNB()),
            ]
        )
    if model_name == "adaboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("model", AdaBoostClassifier(random_state=42, n_estimators=80)),
            ]
        )
    raise ValueError(f"Unknown model name: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/generated")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-frac", type=float, default=0.01)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_sample(data_dir, rows=args.rows, seed=args.seed)
    feature_cols = [c for c in df.columns if c != "prognosis"]

    df = inject_missing_values(df, feature_cols, frac=args.missing_frac, seed=args.seed)

    x = df[feature_cols]
    y = df["prognosis"].astype(str)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_enc, test_size=0.2, random_state=args.seed, stratify=y_enc
    )

    model_names = [
        "logistic_regression",
        "knn",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "gaussian_nb",
        "adaboost",
    ]

    results = []
    best_model_name = None
    best_model = None
    best_f1 = -1.0

    for name in model_names:
        print(f"Training: {name}", flush=True)
        pipeline = make_pipeline(name)
        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        train_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = pipeline.predict(x_test)
        predict_seconds = time.perf_counter() - t1

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro", zero_division=0)
        recall = recall_score(y_test, preds, average="macro", zero_division=0)

        row = {
            "model": name,
            "accuracy": acc,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall,
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
        }
        results.append(row)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = pipeline

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    results_df.to_csv(artifacts_dir / "model_comparison.csv", index=False)

    joblib.dump(best_model, artifacts_dir / "best_model.joblib")
    joblib.dump(label_encoder, artifacts_dir / "label_encoder.joblib")

    metadata = {
        "best_model": best_model_name,
        "trained_rows": int(args.rows),
        "features": feature_cols,
        "classes": label_encoder.classes_.tolist(),
        "missing_fraction_used": args.missing_frac,
        "seed": args.seed,
    }
    (artifacts_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Saved artifacts:")
    print("-", artifacts_dir / "model_comparison.csv")
    print("-", artifacts_dir / "best_model.joblib")
    print("-", artifacts_dir / "label_encoder.joblib")
    print("-", artifacts_dir / "model_metadata.json")
    print("Best model:", best_model_name)
    print(results_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
