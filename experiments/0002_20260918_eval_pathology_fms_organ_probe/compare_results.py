"""Aggregate the per-model results.json files from this experiment into one
comparison table. Run manually after all GRID_VALUES model-keys have
completed (not part of the SLURM job itself):

    python experiments/0002_20260918_eval_pathology_fms_organ_probe/compare_results.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default=None, help="Defaults to outputs/<this exp name>/")
    args = parser.parse_args()

    exp_name = Path(__file__).parent.name
    project_root = Path(__file__).resolve().parents[2]
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else project_root / "outputs" / exp_name

    rows = []
    for results_path in sorted(outputs_dir.glob("*/results.json")):
        results = json.loads(results_path.read_text())
        model_key = results["model_key"]
        for probe_name, metrics in results["cv"].items():
            rows.append(
                {
                    "model_key": model_key,
                    "probe": probe_name,
                    "embedding_dim": results["embedding_dim"],
                    "accuracy_mean": metrics["accuracy_mean"],
                    "accuracy_std": metrics["accuracy_std"],
                    "macro_f1_mean": metrics["macro_f1_mean"],
                    "macro_f1_std": metrics["macro_f1_std"],
                }
            )

    if not rows:
        print(f"No results.json found under {outputs_dir}")
        return

    df = pd.DataFrame(rows).sort_values(["probe", "macro_f1_mean"], ascending=[True, False])
    out_csv = outputs_dir / "comparison.csv"
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
