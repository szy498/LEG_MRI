"""Dataset preparation, subject splits, training and held-out mask generation."""

import json
import os
import random
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .imaging import same_geometry
from .io import read_table, record_run, resolve_path, sha256, write_json


def configure_paths(work):
    work = Path(work).resolve()
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        (work / key).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(work / key)
    os.environ.setdefault("nnUNet_compile", "false")
    return work


def prepare_dataset(manifest, config, work, split_file=None):
    import SimpleITK as sitk

    work = configure_paths(work)
    cfg, seed = config["segmentation"], config["seed"]
    subjects = read_table(manifest).sort_values("Patient_ID").reset_index(drop=True)
    ids = subjects.Patient_ID.tolist()
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", s) for s in ids):
        raise ValueError("Use pseudonymous case IDs containing only letters, numbers, '-' or '_'")
    if split_file:
        split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    else:
        development, test = train_test_split(ids, test_size=cfg["test_fraction"], random_state=seed)
        development = sorted(development)
        folds = [
            {"train": [development[i] for i in a], "val": [development[i] for i in b]}
            for a, b in KFold(cfg["folds"], shuffle=True, random_state=seed).split(development)
        ]
        split = {"development": development, "test": sorted(test), "folds": folds, "seed": seed}
    validate_split(split, ids, cfg["folds"])
    name = f"Dataset{cfg['dataset_id']:03d}_{cfg['dataset_name']}"
    raw, prep = work / "nnUNet_raw" / name, work / "nnUNet_preprocessed" / name
    if raw.exists():
        raise FileExistsError(f"{raw} already exists; use a new work directory")
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    prep.mkdir(parents=True, exist_ok=True)
    base, hashes = Path(manifest).parent, []
    for row in subjects.to_dict("records"):
        sid = row["Patient_ID"]
        images = [sitk.ReadImage(str(resolve_path(base, row[c]))) for c in cfg["channels"]]
        mask = sitk.ReadImage(str(resolve_path(base, row["manual_mask"])))
        same_geometry(mask, *images)
        if not np.isin(sitk.GetArrayFromImage(mask), list(config["labels"].values())).all():
            raise ValueError(f"{sid}: unexpected manual mask label")
        development = sid in split["development"]
        for i, (channel, image) in enumerate(zip(cfg["channels"], images)):
            dest = raw / ("imagesTr" if development else "imagesTs") / f"{sid}_{i:04d}.nii.gz"
            sitk.WriteImage(image, str(dest))
            hashes.append({"Patient_ID": sid, "channel": channel, "sha256": sha256(dest)})
        # Independent test references stay outside nnU-Net training/fingerprint directories.
        dest = raw / "labelsTr" if development else work / "test_references"
        dest.mkdir(exist_ok=True)
        sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(dest / f"{sid}.nii.gz"))
    write_json(
        raw / "dataset.json",
        {
            "channel_names": {str(i): c for i, c in enumerate(cfg["channels"])},
            "labels": config["labels"],
            "numTraining": len(split["development"]),
            "file_ending": ".nii.gz",
        },
    )
    write_json(prep / "splits_final.json", split["folds"])
    write_json(work / "split.json", split)
    write_json(work / "study.json", config)
    write_json(work / "image_hashes.json", hashes)
    subjects.to_csv(work / "subjects.csv", index=False)
    # Preserve a resolved local manifest so subsequent commands do not depend on cwd.
    for col in cfg["channels"] + ["manual_mask"]:
        subjects[col] = subjects[col].map(lambda p: str(resolve_path(base, p).resolve()))
    subjects.to_csv(work / "subjects_resolved.csv", index=False)
    record_run(work, config, [manifest])
    return split


def validate_split(split, ids, folds):
    development, test = set(split["development"]), set(split["test"])
    if development & test or development | test != set(ids):
        raise ValueError("Development/test must be a disjoint partition of participants")
    if len(development) != len(split["development"]) or len(test) != len(split["test"]):
        raise ValueError("Duplicate IDs in split")
    if len(split["folds"]) != folds:
        raise ValueError("Unexpected number of segmentation folds")
    validations = []
    for fold in split["folds"]:
        a, b = set(fold["train"]), set(fold["val"])
        if a & b or a | b != development or not a or not b:
            raise ValueError("Each fold must partition the development cohort")
        validations += fold["val"]
    if sorted(validations) != sorted(development):
        raise ValueError("Every development participant must be held out exactly once")


def customize_plans(plans, config):
    plans = deepcopy(plans)
    cfg = config["segmentation"]
    plans["plans_name"] = "LegMRIPlans"
    plans["legmri"] = config
    full = plans["configurations"]["3d_fullres"]
    full.update(
        patch_size=cfg["patch_size"],
        batch_size=cfg["batch_size"],
        data_identifier="LegMRIPlans_3d_fullres",
    )
    full.pop("next_stage", None)
    full.pop("previous_stage", None)
    arch = full["architecture"]["arch_kwargs"]
    arch.update(
        n_stages=6,
        features_per_stage=cfg["features_per_stage"],
        strides=cfg["strides"],
        kernel_sizes=[[3, 3, 3]] * 6,
        n_conv_per_stage=[2] * 6,
        n_conv_per_stage_decoder=[2] * 5,
    )
    plans["configurations"] = {"3d_fullres": full}
    return plans


def plan_and_preprocess(work, processes=4):
    work = configure_paths(work)
    config = json.loads((work / "study.json").read_text())
    cfg = config["segmentation"]
    name = f"Dataset{cfg['dataset_id']:03d}_{cfg['dataset_name']}"

    def run(entry, *args):
        executable = Path(sys.executable).parent / (entry + (".exe" if os.name == "nt" else ""))
        env = os.environ.copy()
        # The planner also uses this augmentation setting as a CPU thread count.
        # Zero is valid for training's single-process augmenter, not for the planner.
        key = "NNUNET_N_PROC_DA" if os.name == "nt" else "nnUNet_n_proc_DA"
        if env.get(key) == "0":
            env[key] = "1"
        subprocess.run([str(executable), *map(str, args)], check=True, env=env)

    run(
        "nnUNetv2_extract_fingerprint",
        "-d",
        cfg["dataset_id"],
        "-np",
        processes,
        "--verify_dataset_integrity",
    )
    run("nnUNetv2_plan_experiment", "-d", cfg["dataset_id"])
    prep = work / "nnUNet_preprocessed" / name
    plans = customize_plans(json.loads((prep / "nnUNetPlans.json").read_text()), config)
    write_json(prep / "LegMRIPlans.json", plans)
    run(
        "nnUNetv2_preprocess",
        "-d",
        cfg["dataset_id"],
        "-p",
        "LegMRIPlans",
        "-c",
        "3d_fullres",
        "-np",
        processes,
    )


def train_fold(work, fold, device="cuda", resume=False):
    work = configure_paths(work)
    import torch
    from .trainer import LegMRITrainer

    config = json.loads((work / "study.json").read_text())
    cfg, seed = config["segmentation"], config["seed"] + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    name = f"Dataset{cfg['dataset_id']:03d}_{cfg['dataset_name']}"
    prep = work / "nnUNet_preprocessed" / name
    plans = json.loads((prep / "LegMRIPlans.json").read_text())
    dataset = json.loads((work / "nnUNet_raw" / name / "dataset.json").read_text())
    split = json.loads((work / "split.json").read_text())
    if fold not in range(cfg["folds"]):
        raise ValueError("Fold index is outside the configured range")
    trainer = LegMRITrainer(plans, "3d_fullres", fold, dataset, torch.device(device))
    folder = Path(trainer.output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "training_subjects.json", split["folds"][fold])
    if resume:
        trainer.load_checkpoint(str(folder / "checkpoint_latest.pth"))
    elif (folder / "checkpoint_final.pth").exists():
        raise FileExistsError("Fold is already trained; use a new work directory")
    trainer.run_training()
    return folder


def predict_heldout(work, cohort="development", device="cuda", processes=2):
    work = configure_paths(work)
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from .trainer import LegMRITrainer

    config = json.loads((work / "study.json").read_text())
    cfg = config["segmentation"]
    name = f"Dataset{cfg['dataset_id']:03d}_{cfg['dataset_name']}"
    root = work / "nnUNet_results" / name / "LegMRITrainer__LegMRIPlans__3d_fullres"
    plans = json.loads((root / "plans.json").read_text())
    dataset = json.loads((root / "dataset.json").read_text())
    manager = PlansManager(plans)
    configuration = manager.get_configuration("3d_fullres")
    split = json.loads((work / "split.json").read_text())
    subjects = read_table(work / "subjects_resolved.csv").set_index("Patient_ID")
    out = work / "predictions" / cohort
    out.mkdir(parents=True, exist_ok=True)
    groups = (
        [(split["folds"][f]["val"], [f]) for f in range(cfg["folds"])]
        if cohort == "development"
        else [(split["test"], list(range(cfg["folds"])))]
    )
    rows = []
    for ids, folds in groups:
        weights, models, checkpoint = [], [], None
        for fold in folds:
            path = root / f"fold_{fold}" / cfg["checkpoint"]
            # Only load checkpoints produced by this trusted local training workflow.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            weights.append(checkpoint["network_weights"])
            training = json.loads((path.parent / "training_subjects.json").read_text())["train"]
            if set(ids) & set(training):
                raise ValueError("Requested predictions include model training participants")
            models.append(
                {"fold": fold, "checkpoint_sha256": sha256(path), "training_subjects": training}
            )
        arch = configuration.network_arch_init_kwargs
        network = LegMRITrainer.build_network_architecture(
            "", arch, [], len(cfg["channels"]), len(config["labels"]), False
        )
        network.load_state_dict(weights[0])
        predictor = nnUNetPredictor(
            device=torch.device(device),
            perform_everything_on_device=device.startswith("cuda"),
            use_mirroring=True,
            tile_step_size=0.5,
            use_gaussian=True,
        )
        predictor.manual_initialization(
            network,
            manager,
            configuration,
            weights,
            dataset,
            "LegMRITrainer",
            checkpoint["inference_allowed_mirroring_axes"],
        )
        inputs = [[str(subjects.loc[sid, c]) for c in cfg["channels"]] for sid in ids]
        outputs = [str(out / sid) for sid in ids]
        predictor.predict_from_files(
            inputs,
            outputs,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=processes,
            num_processes_segmentation_export=processes,
        )
        for sid in ids:
            mask = out / f"{sid}.nii.gz"
            provenance = out / f"{sid}.json"
            write_json(
                provenance,
                {
                    "Patient_ID": sid,
                    "cohort": cohort,
                    "models": models,
                    "mask_sha256": sha256(mask),
                },
            )
            rows.append(
                {
                    "Patient_ID": sid,
                    **subjects.loc[sid].to_dict(),
                    "predicted_mask": str(mask),
                    "provenance": str(provenance),
                }
            )
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    shutil.copy2(work / "study.json", out / "study.json")
    return out / "manifest.csv"
