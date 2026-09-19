"""Cross-model error analysis for the organ classification probe.

Reuses each model's cached embeddings.npz (no re-inference needed) to get
out-of-fold predictions via the same StratifiedKFold protocol as
experiment.py, then checks whether misclassifications cluster on
particular organs / magnifications / sources rather than being spread
evenly across models. Run manually after the model-key jobs have
completed:

    python experiments/0002_20260918_eval_pathology_fms_organ_probe/analyze_errors.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.yml"
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def resolve_viper_parquet(dataset_dir: Path) -> Path:
    matches = sorted(dataset_dir.glob("datasets--MahmoodLab--viper/snapshots/*/viper.parquet"))
    if not matches:
        raise FileNotFoundError(f"viper.parquet not found under {dataset_dir}")
    return matches[0]


def oof_predict(embeddings: np.ndarray, labels: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Out-of-fold linear-probe predictions, one per sample, using the same
    StratifiedKFold protocol as experiment.py's run_cv (so error rates here
    are directly comparable to the reported CV accuracy)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.empty(len(labels), dtype=object)
    for train_idx, test_idx in skf.split(embeddings, labels):
        clf = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
        clf.fit(embeddings[train_idx], labels[train_idx])
        preds[test_idx] = clf.predict(embeddings[test_idx])
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    project_root = exp_dir.parents[1]
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else project_root / "outputs" / exp_dir.name
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else project_root / "data"

    config = load_config(exp_dir)
    n_splits = config.get("n_splits", 5)
    seed = config.get("seed", 42)

    model_dirs = sorted(p.parent for p in outputs_dir.glob("*/embeddings.npz"))
    if not model_dirs:
        print(f"No embeddings.npz found under {outputs_dir}")
        return

    image_ids_ref = None
    labels_ref = None
    per_model_wrong = {}
    for model_dir in model_dirs:
        model_key = model_dir.name
        data = np.load(model_dir / "embeddings.npz", allow_pickle=True)
        embeddings, labels, image_ids = data["embeddings"], data["labels"], list(data["image_ids"])
        if image_ids_ref is None:
            image_ids_ref = image_ids
            labels_ref = labels
        elif image_ids != image_ids_ref:
            raise ValueError(f"image_id order mismatch for {model_key}; cannot compare across models")

        preds = oof_predict(embeddings, labels, n_splits, seed)
        per_model_wrong[model_key] = preds != labels

    error_df = pd.DataFrame(per_model_wrong)
    error_df.insert(0, "image_id", image_ids_ref)
    error_df.insert(1, "organ", labels_ref)
    model_keys = list(per_model_wrong)
    error_df["n_models_wrong"] = error_df[model_keys].sum(axis=1)

    parquet_path = resolve_viper_parquet(dataset_dir)
    meta = pq.read_table(parquet_path, columns=["image_id", "category", "magnification", "source"]).to_pandas()
    meta = meta.drop_duplicates(subset="image_id")
    error_df = error_df.merge(meta, on="image_id", how="left")

    n_models = len(model_keys)
    print(f"Models compared: {model_keys}\n")

    print("=== images wrong by N models (0 = all models correct) ===")
    print(error_df["n_models_wrong"].value_counts().sort_index().to_string())

    for col in ["organ", "category", "magnification", "source"]:
        print(f"\n=== mean fraction of models wrong, by {col} ===")
        rate = (error_df.groupby(col)["n_models_wrong"].mean() / n_models).sort_values(ascending=False)
        print(rate.to_string())

    hard_images = error_df[error_df["n_models_wrong"] >= n_models - 1].sort_values(
        "n_models_wrong", ascending=False
    )
    print(f"\n=== images wrong by >= {n_models - 1}/{n_models} models ({len(hard_images)}) ===")
    print(hard_images[["image_id", "organ", "category", "magnification", "source", "n_models_wrong"]].to_string(index=False))

    out_csv = outputs_dir / "error_analysis.csv"
    error_df.to_csv(out_csv, index=False)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
