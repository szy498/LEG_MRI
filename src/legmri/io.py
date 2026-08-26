"""Small shared IO helpers; importing the package never starts an analysis."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_table(path, id_col="Patient_ID"):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        frame = pd.read_excel(path, dtype={id_col: str})
    else:
        frame = pd.read_csv(path, dtype={id_col: str})
    if id_col in frame:
        if frame[id_col].isna().any() or frame[id_col].duplicated().any():
            raise ValueError(f"{path}: missing or duplicate {id_col}")
        frame[id_col] = frame[id_col].str.strip()
        if frame[id_col].eq("").any() or frame[id_col].duplicated().any():
            raise ValueError(f"{path}: empty or duplicate normalized {id_col}")
    return frame


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, (np.integer, np.floating, np.bool_)):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, Path):
            return str(item)
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_run(output, config, inputs=()):
    packages = [
        "leg-mri",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "torch",
        "nnunetv2",
        "SimpleITK",
        "pyradiomics",
        "xgboost",
        "shap",
    ]
    versions = {}
    source_installs = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
            direct_url = importlib.metadata.distribution(name).read_text("direct_url.json")
            if direct_url and name == "pyradiomics":
                source_installs[name] = json.loads(direct_url)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(
        Path(output) / "run.json",
        {
            "config": config,
            "versions": versions,
            "source_installs": source_installs,
            "inputs_sha256": {str(Path(p).resolve()): sha256(p) for p in inputs},
        },
    )


def resolve_path(base, path):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path
