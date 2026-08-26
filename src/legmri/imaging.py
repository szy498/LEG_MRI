"""DICOM conversion and native-space tissue measurements."""

import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_table, record_run, resolve_path, sha256, write_json


def same_geometry(*images, atol=1e-4):
    ref = images[0]
    for img in images[1:]:
        if img.GetDimension() != 3 or img.GetSize() != ref.GetSize():
            raise ValueError("Images and masks must share a 3D native-space grid")
        for getter in ("GetSpacing", "GetOrigin", "GetDirection"):
            if not np.allclose(getattr(img, getter)(), getattr(ref, getter)(), atol=atol, rtol=0):
                raise ValueError(f"Geometry mismatch: {getter}; align images explicitly first")


def dicom_inventory(folder):
    """Group classic single-frame DICOMs by series, component and echo, not file order."""
    import pydicom
    from pydicom.errors import InvalidDicomError

    groups = defaultdict(list)
    descriptions = {}
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except InvalidDicomError:
            continue
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("Enhanced multiframe DICOM: convert with dcm2niix, then use NIfTI")
        if not hasattr(ds, "ImagePositionPatient"):
            continue
        key = (
            str(ds.SeriesInstanceUID),
            tuple(getattr(ds, "ImageType", [])),
            str(getattr(ds, "EchoNumbers", "")),
            str(getattr(ds, "EchoTime", "")),
            str(getattr(ds, "TemporalPositionIdentifier", "")),
        )
        groups[key].append(str(path.resolve()))
        descriptions[key] = str(getattr(ds, "SeriesDescription", ""))
    return {
        hashlib.sha256(repr(k).encode()).hexdigest()[:16]: {
            "description": descriptions[k],
            "image_type": list(k[1]),
            "echo": k[2:4],
            "files": v,
            "slices": len(v),
        }
        for k, v in groups.items()
    }


def convert_dicom_group(files, output):
    import pydicom
    import SimpleITK as sitk

    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in files]
    if len(headers) < 2:
        raise ValueError("At least two spatial slices are needed for a 3D series")
    orientation = np.asarray(headers[0].ImageOrientationPatient, float)
    normal = np.cross(orientation[:3], orientation[3:])
    positions = []
    for ds in headers:
        if not np.allclose(ds.ImageOrientationPatient, orientation, atol=1e-4):
            raise ValueError("Inconsistent DICOM orientation within selected component")
        positions.append(np.asarray(ds.ImagePositionPatient, float))
    order = np.argsort(np.asarray(positions) @ normal)
    ordered_positions = np.asarray(positions)[order]
    step = np.diff(ordered_positions @ normal)
    if np.any(step < 1e-4) or not np.allclose(step, np.median(step), atol=1e-3):
        raise ValueError("Duplicate or irregular slice positions; select one Dixon component")
    # Reject in-plane drift (shear) rather than silently constructing a wrong affine.
    if not np.allclose(np.diff(ordered_positions, axis=0), step[:, None] * normal, atol=1e-3):
        raise ValueError("Non-orthogonal slice stack: use an appropriate DICOM converter")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([files[i] for i in order])
    image = reader.Execute()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output))


def ff_fraction(array, divisor):
    if divisor <= 0:
        raise ValueError("ff_divisor must be positive (1 for fractions, 100 for percentages)")
    values = np.asarray(array, float) / divisor
    if not np.isfinite(values).all() or values.min() < -1e-4 or values.max() > 1.0001:
        raise ValueError("FF outside [0,1]: confirm scanner scaling and ff_divisor")
    return np.clip(values, 0, 1)


def quantitative_features(mask, ff, spacing, labels, water=None, fat=None):
    """Volumes in mL, dimensionless ratios and native scanner FF in [0,1]."""
    if mask.shape != ff.shape or not np.isin(mask, list(labels.values())).all():
        raise ValueError("Mask shape or semantic labels are invalid")
    voxel_ml = float(np.prod(spacing)) / 1000
    volumes = {k: float((mask == v).sum()) * voxel_ml for k, v in labels.items() if v}
    m, i, s = (volumes[k] for k in ("Muscle", "IMAT", "Subcutaneous_Fat"))
    soft, fat_total = m + i + s, i + s

    def ratio(a, b):
        return a / b if b > 0 else np.nan

    features = {f"{k}_volume_ml": v for k, v in volumes.items()}
    features.update(
        {
            "IMAT_to_Muscle": ratio(i, m),
            "Subcutaneous_to_Muscle": ratio(s, m),
            "IMAT_to_IMAT_Muscle": ratio(i, i + m),
            "Muscle_to_SoftTissue": ratio(m, soft),
            "Fat_to_SoftTissue": ratio(fat_total, soft),
            "IMAT_to_SoftTissue": ratio(i, soft),
            "SubFat_to_SoftTissue": ratio(s, soft),
            "Fat_to_Muscle": ratio(fat_total, m),
        }
    )
    roi = mask == labels["Muscle"]
    v = ff[roi]
    features.update(
        {
            "SMFF_mean": np.mean(v) if len(v) else np.nan,
            "SMFF_std": np.std(v, ddof=0) if len(v) else np.nan,
            "SMFF_median": np.median(v) if len(v) else np.nan,
            "SMFF_25perc": np.percentile(v, 25) if len(v) else np.nan,
            "SMFF_75perc": np.percentile(v, 75) if len(v) else np.nan,
        }
    )
    if water is not None and fat is not None:
        # Ratio of summed signals, not mean of voxel-wise ratios.
        denom = np.sum(water[roi] + fat[roi])
        features["SMFF_wf"] = ratio(np.sum(fat[roi]), denom)
    return features


def verify_mask_provenance(subject_id, provenance_path, mask_path):
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    if provenance["Patient_ID"] != str(subject_id):
        raise ValueError("Mask provenance participant mismatch")
    if provenance["mask_sha256"] != sha256(mask_path):
        raise ValueError("Mask checksum does not match its prediction record")
    for model in provenance["models"]:
        if str(subject_id) in set(map(str, model["training_subjects"])):
            raise ValueError(f"In-sample segmentation for {subject_id}; use held-out predictions")
    if not provenance["models"]:
        raise ValueError("Mask provenance has no source model")
    return provenance


def extract_features(manifest, config, output, quantitative_only=False):
    import SimpleITK as sitk

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = read_table(manifest)
    base = Path(manifest).parent
    cfg, labels = config["extraction"], config["labels"]
    rows_q, rows_r, quality, input_records = [], [], [], []
    extractor = None
    if not quantitative_only:
        from radiomics import featureextractor

        extractor = featureextractor.RadiomicsFeatureExtractor(
            binWidth=cfg["bin_width"],
            resampledPixelSpacing=None,
            normalize=False,
            force2D=False,
            distances=[1],
            weightingNorm=None,
            symmetricalGLCM=True,
            correctMask=False,
            geometryTolerance=1e-4,
            minimumROIDimensions=2,
        )
        extractor.disableAllFeatures()
        for family in cfg["feature_families"]:
            extractor.enableFeatureClassByName(family)
    for row in frame.to_dict("records"):
        sid = str(row["Patient_ID"])
        mask_path = resolve_path(base, row["predicted_mask"])
        provenance = verify_mask_provenance(sid, resolve_path(base, row["provenance"]), mask_path)
        input_records.append(
            {
                "Patient_ID": sid,
                "mask_sha256": sha256(mask_path),
                "images_sha256": {
                    k: sha256(resolve_path(base, row[k])) for k in ("water", "fat", "ff")
                },
                "prediction_provenance": provenance,
            }
        )
        images = {
            k: sitk.ReadImage(str(resolve_path(base, row[k]))) for k in ("water", "fat", "ff")
        }
        mask_img = sitk.ReadImage(str(mask_path))
        same_geometry(mask_img, *images.values())
        mask = sitk.GetArrayFromImage(mask_img)
        arrays = {k: sitk.GetArrayFromImage(v).astype(float) for k, v in images.items()}
        ff = ff_fraction(arrays["ff"], cfg["ff_divisor"])
        rows_q.append(
            {
                "Patient_ID": sid,
                **quantitative_features(
                    mask, ff, mask_img.GetSpacing(), labels, arrays["water"], arrays["fat"]
                ),
            }
        )
        radiomic = {"Patient_ID": sid}
        if extractor is not None:
            for modality in cfg["modalities"]:
                arr = arrays[modality].copy()
                normalization = cfg.get(f"{modality}_normalization", "none")
                if modality == "ff":
                    arr = ff
                elif normalization == "zscore":
                    std = arr.std(ddof=1)
                    if std <= 0:
                        raise ValueError(f"{sid}: constant {modality} image")
                    arr = (arr - arr.mean()) / std * cfg.get(f"{modality}_normalize_scale", 1)
                elif normalization != "none":
                    raise ValueError(f"Unknown normalization: {normalization}")
                normalized = sitk.GetImageFromArray(arr)
                normalized.CopyInformation(images[modality])
                for tissue in cfg["tissues"]:
                    roi = mask == labels[tissue]
                    n = int(roi.sum())
                    if n < cfg["minimum_roi_voxels"]:
                        quality.append(
                            {
                                "Patient_ID": sid,
                                "modality": modality,
                                "tissue": tissue,
                                "status": "ROI_below_minimum",
                                "voxels": n,
                            }
                        )
                        continue
                    bins = np.unique(np.floor(arr[roi] / cfg["bin_width"])).size
                    quality.append(
                        {
                            "Patient_ID": sid,
                            "modality": modality,
                            "tissue": tissue,
                            "status": "ok",
                            "voxels": n,
                            "occupied_bins": bins,
                        }
                    )
                    if bins < 2:
                        warnings.warn(
                            f"{sid}/{modality}/{tissue}: single gray-level bin", stacklevel=2
                        )
                    values = extractor.execute(normalized, mask_img, label=labels[tissue])
                    radiomic.update(
                        {
                            f"{tissue}_{modality.title()}_{k}": float(v)
                            for k, v in values.items()
                            if k.startswith("original_")
                        }
                    )
        rows_r.append(radiomic)
    pd.DataFrame(rows_q).to_csv(out / "quantitative.csv", index=False)
    pd.DataFrame(rows_r).to_csv(out / "radiomics.csv", index=False)
    pd.DataFrame(quality).to_csv(out / "extraction_qc.csv", index=False)
    write_json(out / "feature_provenance.json", input_records)
    record_run(out, config, [manifest])


def dice_per_class(prediction, reference, labels):
    if prediction.shape != reference.shape:
        raise ValueError("Prediction/reference shape mismatch")
    result = {}
    for name, label in labels.items():
        if label == 0:
            continue
        p, r = prediction == label, reference == label
        denom = p.sum() + r.sum()
        result[name] = float(2 * (p & r).sum() / denom) if denom else np.nan
    result["Overall"] = float(np.nanmean(list(result.values())))
    return result


def evaluate_segmentation(manifest, labels, output):
    import SimpleITK as sitk

    base, rows = Path(manifest).parent, []
    for row in read_table(manifest).to_dict("records"):
        pred = sitk.ReadImage(str(resolve_path(base, row["predicted_mask"])))
        ref = sitk.ReadImage(str(resolve_path(base, row["manual_mask"])))
        same_geometry(pred, ref)
        pa, ra = sitk.GetArrayFromImage(pred), sitk.GetArrayFromImage(ref)
        if (
            not np.isin(pa, list(labels.values())).all()
            or not np.isin(ra, list(labels.values())).all()
        ):
            raise ValueError("Unexpected segmentation label")
        rows.append({"Patient_ID": row["Patient_ID"], **dice_per_class(pa, ra, labels)})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "subject_dice.csv", index=False)
    result.drop(columns="Patient_ID").agg(["mean", "std", "count"]).to_csv(out / "dice_summary.csv")
    write_json(
        out / "definitions.json",
        {
            "unit": "participant",
            "std_ddof": 1,
            "both_empty": "undefined; excluded from mean",
            "overall": "mean foreground Dice",
        },
    )
