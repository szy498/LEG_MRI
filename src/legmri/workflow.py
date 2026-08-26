"""Connections between the independently usable pipeline stages."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .io import load_config, read_table, record_run, write_json


def combine_features(folders, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    extraction_settings = []
    inputs = []
    for folder in map(Path, folders):
        record = folder / "run.json"
        if record.exists():
            saved = json.loads(record.read_text(encoding="utf-8"))["config"]
            extraction_settings.append({k: saved.get(k) for k in ("extraction", "labels")})
    if extraction_settings and any(x != extraction_settings[0] for x in extraction_settings[1:]):
        raise ValueError("Feature folders were extracted with different settings")
    for domain in ("quantitative", "radiomics"):
        tables = []
        for folder in map(Path, folders):
            path = folder / f"{domain}.csv"
            tables.append(read_table(path))
            inputs.append(path)
        frame = pd.concat(tables, ignore_index=True, sort=False)
        if frame.Patient_ID.duplicated().any():
            raise ValueError("Feature folders contain duplicate participants")
        frames[domain] = frame.sort_values("Patient_ID").reset_index(drop=True)
    if not frames["quantitative"].Patient_ID.equals(frames["radiomics"].Patient_ID):
        raise ValueError("Quantitative and radiomics participant sets differ")
    for domain, frame in frames.items():
        frame.to_csv(out / f"{domain}.csv", index=False)
    provenance = []
    for folder in map(Path, folders):
        path = folder / "feature_provenance.json"
        if path.exists():
            provenance.extend(json.loads(path.read_text(encoding="utf-8")))
    if provenance:
        write_json(out / "feature_provenance.json", provenance)
    record_run(out, extraction_settings[0] if extraction_settings else {}, inputs)
    return out


def run_study(
    subjects,
    clinical,
    config_path,
    work,
    device="cuda",
    processes=4,
    start_at="prepare",
    split_file=None,
):
    """Execute a prespecified configuration, with resumable stage boundaries.

    Fresh subprocesses isolate nnU-Net's import-time path configuration and
    multiprocessing workers. No hyperparameters are selected on the segmentation test set.
    """
    work = Path(work).resolve()
    config_path = Path(config_path).resolve()
    subjects, clinical = Path(subjects).resolve(), Path(clinical).resolve()
    config = load_config(config_path)
    work.mkdir(parents=True, exist_ok=True)
    seg = work / "segmentation"
    stages = ["prepare", "plan", "train", "predict", "extract", "model"]
    if start_at != "prepare":
        previous = json.loads((seg / "study.json").read_text())
        if previous != config:
            raise ValueError("Resume configuration differs from the saved study configuration")

    def run(*arguments):
        subprocess.run([sys.executable, "-m", "legmri", *map(str, arguments)], check=True)

    for stage in stages[stages.index(start_at) :]:
        if stage == "prepare":
            args = ["seg-prepare", subjects, "--config", config_path, "--work", seg]
            if split_file:
                args.extend(["--splits", Path(split_file).resolve()])
            run(*args)
        elif stage == "plan":
            run("seg-plan", "--work", seg, "--processes", processes)
        elif stage == "train":
            for fold in range(config["segmentation"]["folds"]):
                cfg = config["segmentation"]
                dataset = f"Dataset{cfg['dataset_id']:03d}_{cfg['dataset_name']}"
                folder = (
                    seg
                    / "nnUNet_results"
                    / dataset
                    / "LegMRITrainer__LegMRIPlans__3d_fullres"
                    / f"fold_{fold}"
                )
                if (folder / "checkpoint_final.pth").exists():
                    continue
                args = ["seg-train", "--work", seg, "--fold", fold, "--device", device]
                if (folder / "checkpoint_latest.pth").exists():
                    args.append("--resume")
                run(*args)
        elif stage == "predict":
            for cohort in ("development", "test"):
                run(
                    "seg-predict",
                    "--work",
                    seg,
                    "--cohort",
                    cohort,
                    "--device",
                    device,
                    "--processes",
                    processes,
                )
                run(
                    "seg-evaluate",
                    seg / "predictions" / cohort / "manifest.csv",
                    "--config",
                    config_path,
                    "--output",
                    work / "segmentation_metrics" / cohort,
                )
        elif stage == "extract":
            for cohort in ("development", "test"):
                run(
                    "extract",
                    seg / "predictions" / cohort / "manifest.csv",
                    "--config",
                    config_path,
                    "--output",
                    work / f"features_{cohort}",
                )
            combine_features(
                [work / "features_development", work / "features_test"], work / "features"
            )
        elif stage == "model":
            run(
                "model-cv",
                "--clinical",
                clinical,
                "--quantitative",
                work / "features" / "quantitative.csv",
                "--radiomics",
                work / "features" / "radiomics.csv",
                "--config",
                config_path,
                "--output",
                work / "machine_learning",
            )
    record_run(work, config, [subjects, clinical, config_path])
