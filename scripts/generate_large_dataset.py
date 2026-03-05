#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_source(train_path: Path, test_path: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    df = pd.concat([train_df, test_df], ignore_index=True)
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    symptom_cols = [c for c in df.columns if c != "prognosis"]
    df[symptom_cols] = (
        df[symptom_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(np.uint8)
    )
    df["prognosis"] = df["prognosis"].astype(str).str.strip()
    return df


def generate_chunk(
    *,
    rows: int,
    symptom_cols: list[str],
    class_names: np.ndarray,
    class_weights: np.ndarray,
    class_symptom_probs: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    class_idx = rng.choice(len(class_names), size=rows, p=class_weights)
    probs = class_symptom_probs[class_idx]
    x = (rng.random(size=probs.shape) < probs).astype(np.uint8)

    # Avoid all-zero symptom vectors.
    zero_rows = np.where(x.sum(axis=1) == 0)[0]
    if len(zero_rows) > 0:
        symptom_idx = rng.integers(low=0, high=len(symptom_cols), size=len(zero_rows))
        x[zero_rows, symptom_idx] = 1

    out = pd.DataFrame(x, columns=symptom_cols)
    out["prognosis"] = class_names[class_idx]
    return out


def write_symptom_dictionary(symptom_cols: list[str], output_dir: Path) -> None:
    symptom_df = pd.DataFrame(
        {
            "symptom_code": symptom_cols,
            "symptom_display": [s.replace("_", " ").title() for s in symptom_cols],
        }
    )
    symptom_df.to_csv(output_dir / "symptom_dictionary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large synthetic symptom dataset.")
    parser.add_argument("--train", default="data/raw/training_data.csv")
    parser.add_argument("--test", default="data/raw/test_data.csv")
    parser.add_argument("--out-dir", default="data/generated")
    parser.add_argument("--total-rows", type=int, default=1_000_000)
    parser.add_argument("--rows-per-file", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_df = load_source(Path(args.train), Path(args.test))
    symptom_cols = [c for c in source_df.columns if c != "prognosis"]

    counts = source_df["prognosis"].value_counts().sort_index()
    class_names = counts.index.to_numpy()
    class_weights = (counts / counts.sum()).to_numpy()

    # Laplace smoothing keeps probabilities in (0, 1) and adds realism/noise.
    grouped = source_df.groupby("prognosis")[symptom_cols]
    sum_df = grouped.sum().reindex(class_names)
    n_df = grouped.size().reindex(class_names)
    alpha = 0.6
    class_symptom_probs = ((sum_df + alpha).to_numpy()) / ((n_df + 2 * alpha).to_numpy()[:, None])

    rng = np.random.default_rng(args.seed)

    parts = int(np.ceil(args.total_rows / args.rows_per_file))
    rows_written = 0
    generated_files: list[str] = []

    for part_idx in range(parts):
        remaining = args.total_rows - rows_written
        if remaining <= 0:
            break
        chunk_rows = min(args.rows_per_file, remaining)
        chunk_df = generate_chunk(
            rows=chunk_rows,
            symptom_cols=symptom_cols,
            class_names=class_names,
            class_weights=class_weights,
            class_symptom_probs=class_symptom_probs,
            rng=rng,
        )
        out_path = out_dir / f"symptom_disease_1m_part_{part_idx + 1:02d}.csv"
        chunk_df.to_csv(out_path, index=False)
        generated_files.append(out_path.name)
        rows_written += chunk_rows

    source_df.to_csv(out_dir / "source_cleaned.csv", index=False)
    write_symptom_dictionary(symptom_cols, out_dir)

    metadata = pd.DataFrame(
        {
            "key": [
                "total_rows_generated",
                "rows_per_file",
                "num_files",
                "num_symptoms",
                "num_diseases",
                "seed",
            ],
            "value": [
                rows_written,
                args.rows_per_file,
                len(generated_files),
                len(symptom_cols),
                len(class_names),
                args.seed,
            ],
        }
    )
    metadata.to_csv(out_dir / "generation_metadata.csv", index=False)

    file_index = pd.DataFrame({"csv_file": generated_files})
    file_index.to_csv(out_dir / "generated_files.csv", index=False)

    print(f"Generated {rows_written:,} rows across {len(generated_files)} files in {out_dir}")


if __name__ == "__main__":
    main()
