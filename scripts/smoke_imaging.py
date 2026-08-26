"""Small synthetic NIfTI -> segmentation -> radiomics -> ML software test.

This deliberately uses a tiny network, two segmentation folds and reduced ML
grids. No output is a study result. Requires all optional dependencies.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

from legmri.io import load_config, read_table, write_json
from legmri.imaging import same_geometry, verify_mask_provenance
from legmri.workflow import run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--start-at",
        default="prepare",
        choices=["prepare", "plan", "train", "predict", "extract", "model"],
    )
    args = parser.parse_args()
    root = Path(args.output).resolve()
    data = root / "synthetic_data"
    data.mkdir(parents=True, exist_ok=True)
    config_path = root / "smoke.yaml"
    if args.start_at == "prepare":
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "study.yaml")
        config["segmentation"].update(
            folds=2,
            patch_size=[17, 33, 33],
            batch_size=1,
            features_per_stage=[2, 4, 4, 8, 8, 8],
            strides=[[1, 1, 1], [1, 2, 2]] + [[2, 2, 2]] * 4,
            epochs=1,
            iterations_per_epoch=1,
            validation_iterations=1,
        )
        config["selection"].update(texture_stability_splits=2, integration_folds=2, lasso_c=[1.0])
        config["cv"].update(
            outer_folds=3,
            outer_repeats=1,
            inner_folds=3,
            inner_repeats=1,
            bootstrap=20,
            shap=False,
            grids={
                "LR": {"C": [1.0]},
                "RF": {"n_estimators": [10], "max_depth": [2]},
                "XGB": {"n_estimators": [10], "max_depth": [2]},
            },
        )
        rng = np.random.default_rng(19)
        z, y, x = np.indices((16, 24, 24))
        radius = ((x - 12) ** 2 + (y - 12) ** 2) ** 0.5
        mask = np.zeros(radius.shape, np.uint8)
        mask[(radius < 11) & (z > 1) & (z < 14)] = 2
        mask[(radius < 8) & (z > 1) & (z < 14)] = 3
        mask[(radius < 3) & (z > 1) & (z < 14)] = 4
        mask[(x > 16) & (x < 20) & (y > 8) & (y < 16) & (z > 1) & (z < 14)] = 1
        subjects, clinical = [], []
        for i in range(24):
            sid = f"synthetic-{i:03d}"
            row = {"Patient_ID": sid}
            ff = np.clip(
                np.choose(mask, [0, 70, 90, 8 + i % 2 * 6, 25]) + rng.normal(0, 2, mask.shape),
                0,
                100,
            ).astype(np.float32)
            total = (600 + rng.normal(0, 20, mask.shape)).astype(np.float32)
            for channel, array in {
                "water": total * (1 - ff / 100),
                "fat": total * ff / 100,
                "ff": ff,
                "manual_mask": mask,
            }.items():
                image = sitk.GetImageFromArray(array)
                image.SetSpacing((1.0, 1.0, 2.0))
                image.SetOrigin((11.0, -8.0, 30.0))
                path = data / f"{sid}_{channel}.nii.gz"
                sitk.WriteImage(image, str(path))
                row[channel] = str(path)
            subjects.append(row)
            clinical.append(
                {
                    "Patient_ID": sid,
                    "Diabetes_Status": i % 2,
                    "Age": 30 + i,
                    "Sex": (i // 2) % 2,
                    "BMI": 22 + rng.normal(),
                }
            )
        pd.DataFrame(subjects).to_csv(data / "subjects.csv", index=False)
        pd.DataFrame(clinical).to_csv(data / "clinical.csv", index=False)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.environ["nnUNet_n_proc_DA"] = "0"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    run_study(
        data / "subjects.csv",
        data / "clinical.csv",
        config_path,
        root / "run",
        device="cpu",
        processes=1,
        start_at=args.start_at,
    )
    validate_outputs(root)


def validate_outputs(root):
    root = Path(root)
    work = root / "run"
    split = json.loads((work / "segmentation" / "split.json").read_text())
    total = 0
    for cohort in ("development", "test"):
        manifest = read_table(work / "segmentation" / "predictions" / cohort / "manifest.csv")
        assert set(manifest.Patient_ID) == set(split[cohort])
        for row in manifest.to_dict("records"):
            record = verify_mask_provenance(
                row["Patient_ID"], row["provenance"], row["predicted_mask"]
            )
            assert len(record["models"]) == (1 if cohort == "development" else 2)
            same_geometry(sitk.ReadImage(row["water"]), sitk.ReadImage(row["predicted_mask"]))
        total += len(manifest)
    assert total == 24
    quant = read_table(work / "features" / "quantitative.csv")
    radio = read_table(work / "features" / "radiomics.csv")
    assert len(quant) == len(radio) == total
    assert any("_original_glcm_" in c for c in radio.columns)
    predictions = pd.read_csv(work / "machine_learning" / "outer_predictions.csv")
    assert len(predictions) == 24 * 7 * 3
    assert np.isfinite(predictions.probability).all()
    assert predictions.probability.between(0, 1).all()
    splits = json.loads((work / "machine_learning" / "outer_splits.json").read_text())
    for fold in splits:
        assert not set(fold["train"]) & set(fold["test"])
    write_json(
        root / "validation.json",
        {
            "synthetic_only": True,
            "participants": total,
            "segmentation_folds": 2,
            "outer_prediction_rows": len(predictions),
            "native_geometry": "passed",
            "mask_provenance": "passed",
        },
    )
    print("Synthetic image-to-ML validation passed.")


if __name__ == "__main__":
    main()
