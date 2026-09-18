import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_registry(exp_dir: Path) -> dict:
    """Load models.yml (feature extractor registry) from the experiment directory."""
    registry_path = exp_dir / "models.yml"
    with open(registry_path) as f:
        return (yaml.safe_load(f) or {}).get("models", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    return parser.parse_args()


def resolve_viper_parquet(dataset_dir: Path) -> Path:
    """Locate the VIPER parquet under an HF-hub-cache-shaped directory tree.

    The snapshot directory name is a content hash that changes if the
    dataset revision changes, so it must be resolved via glob rather than
    hardcoded (same pattern as 0001's viper_output_dir.glob(...)).
    """
    matches = sorted(dataset_dir.glob("datasets--MahmoodLab--viper/snapshots/*/viper.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"viper.parquet not found under {dataset_dir}/datasets--MahmoodLab--viper/snapshots/*/"
        )
    return matches[0]


def load_organ_dataset(parquet_path: Path) -> tuple[list[Image.Image], np.ndarray, list[str]]:
    """Decode the VIPER parquet into (unique images, organ labels, image_ids).

    VIPER has 3 questions per image (mcq/free_text/kprim), so rows are
    deduplicated by image_id to get one image per organ-labeled sample.
    """
    table = pq.read_table(parquet_path, columns=["image_id", "image", "organ"])
    df = table.to_pandas()
    df = df.drop_duplicates(subset="image_id", keep="first").reset_index(drop=True)

    images = [Image.open(io.BytesIO(row["bytes"])).convert("RGB") for row in df["image"]]
    labels = df["organ"].to_numpy()
    image_ids = df["image_id"].tolist()
    return images, labels, image_ids


def _resolve_weights_path(hf_cache_dir: str, weights_glob: str) -> Path:
    matches = sorted(Path(hf_cache_dir, "hub").glob(weights_glob))
    if not matches:
        raise FileNotFoundError(f"No weights found under {hf_cache_dir}/hub/{weights_glob}")
    return matches[0]


def _load_uni(spec: dict, device: str, hf_cache_dir: str):
    import timm

    weights_path = _resolve_weights_path(hf_cache_dir, spec["weights_glob"])
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    def embed_fn(m, x):
        return m(x)

    return model, transform, embed_fn


def _load_conch(spec: dict, device: str, hf_cache_dir: str):
    from conch.open_clip_custom import create_model_from_pretrained

    weights_path = _resolve_weights_path(hf_cache_dir, spec["weights_glob"])
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", str(weights_path))
    model.eval().to(device)

    def embed_fn(m, x):
        return m.encode_image(x, proj_contrast=False, normalize=False)

    return model, preprocess, embed_fn


def _load_virchow2(spec: dict, device: str, hf_cache_dir: str):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        f"hf-hub:{spec['hf_hub_id']}",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    def embed_fn(m, x):
        out = m(x)  # [B, 261, 1280] = CLS + 4 register tokens + 256 patch tokens
        cls_token = out[:, 0]
        patch_tokens = out[:, 5:]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)  # [B, 2560]

    return model, transform, embed_fn


def _load_h_optimus_0(spec: dict, device: str, hf_cache_dir: str):
    import timm

    model = timm.create_model(
        f"hf-hub:{spec['hf_hub_id']}",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
            ),
        ]
    )

    def embed_fn(m, x):
        return m(x)

    return model, transform, embed_fn


LOADERS: dict[str, Callable] = {
    "uni": _load_uni,
    "conch": _load_conch,
    "virchow2": _load_virchow2,
    "h_optimus_0": _load_h_optimus_0,
}


def extract_embeddings(
    images: list[Image.Image],
    model,
    transform,
    embed_fn: Callable,
    device: str,
    batch_size: int,
    logger: logging.Logger,
) -> np.ndarray:
    chunks = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        tensor = torch.stack([transform(img) for img in batch]).to(device)
        with torch.inference_mode():
            out = embed_fn(model, tensor)
        chunks.append(out.float().cpu().numpy())
        logger.info(f"embedded {min(i + batch_size, len(images))}/{len(images)}")
    return np.concatenate(chunks, axis=0)


def run_cv(embeddings: np.ndarray, labels: np.ndarray, n_splits: int, knn_k: int, seed: int) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_scores: dict[str, dict[str, list[float]]] = {
        "linear_probe": {"accuracy": [], "macro_f1": []},
        "knn": {"accuracy": [], "macro_f1": []},
    }

    for train_idx, test_idx in skf.split(embeddings, labels):
        x_train, x_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        classifiers = {
            "linear_probe": Pipeline(
                [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
            ),
            "knn": Pipeline(
                [("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=knn_k))]
            ),
        }
        for probe_name, clf in classifiers.items():
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            fold_scores[probe_name]["accuracy"].append(accuracy_score(y_test, pred))
            fold_scores[probe_name]["macro_f1"].append(f1_score(y_test, pred, average="macro"))

    summary = {}
    for probe_name, metrics in fold_scores.items():
        summary[probe_name] = {
            f"{metric_name}_mean": float(np.mean(values))
            for metric_name, values in metrics.items()
        } | {
            f"{metric_name}_std": float(np.std(values))
            for metric_name, values in metrics.items()
        }
    return summary


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    exp_dir = Path(__file__).parent
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    variant_key = args.model_key

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(exp_dir)
    registry = load_registry(exp_dir)
    seed = config.get("seed", 42)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, model_key=args.model_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")

    if args.model_key not in registry:
        raise KeyError(f"'{args.model_key}' not found in models.yml (available: {list(registry)})")
    spec = registry[args.model_key]
    loader_key = spec["loader"].replace("-", "_")
    if loader_key not in LOADERS:
        raise KeyError(f"No loader implemented for '{loader_key}' (models.yml: {args.model_key})")

    hf_cache_dir = config.get("hf_cache_dir")
    if hf_cache_dir:
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HF_HUB_OFFLINE"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    parquet_path = resolve_viper_parquet(dataset_dir)
    images, labels, image_ids = load_organ_dataset(parquet_path)
    logger.info(f"loaded {len(images)} unique images, {len(set(labels))} organ classes")

    model, transform, embed_fn = LOADERS[loader_key](spec, device, hf_cache_dir)
    embeddings = extract_embeddings(
        images, model, transform, embed_fn, device, config.get("batch_size", 16), logger
    )
    logger.info(f"embeddings shape: {embeddings.shape}")

    # Flush embeddings before running CV so a crash during evaluation doesn't
    # lose the (expensive-to-recompute) extraction step.
    np.savez(run_dir / "embeddings.npz", embeddings=embeddings, labels=labels, image_ids=image_ids)

    cv_summary = run_cv(embeddings, labels, config.get("n_splits", 5), config.get("knn_k", 5), seed)

    results = {
        "model_key": args.model_key,
        "n_images": len(images),
        "embedding_dim": int(embeddings.shape[1]),
        "n_classes": int(len(set(labels))),
        "cv": cv_summary,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
