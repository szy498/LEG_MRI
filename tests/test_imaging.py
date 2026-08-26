import json

import numpy as np
import pytest

from legmri.imaging import (
    dice_per_class,
    ff_fraction,
    quantitative_features,
    same_geometry,
    verify_mask_provenance,
)
from legmri.io import sha256


def test_quantitative_ratios_and_units(config):
    mask = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4]).reshape(1, 2, 5)
    result = quantitative_features(
        mask,
        np.full(mask.shape, 0.2),
        [2, 2, 5],
        config["labels"],
        np.full(mask.shape, 80),
        np.full(mask.shape, 20),
    )
    assert result["Muscle_volume_ml"] == pytest.approx(0.08)
    assert result["IMAT_to_Muscle"] == 0.5
    assert result["IMAT_to_SoftTissue"] == pytest.approx(2 / 9)
    assert result["SMFF_wf"] == 0.2
    assert result["SMFF_mean"] == 0.2


def test_ff_scaling_requires_declared_unit():
    assert ff_fraction(np.array([0, 50, 100]), 100).tolist() == [0, 0.5, 1]
    with pytest.raises(ValueError):
        ff_fraction(np.array([50]), 1)


def test_geometry_checks_origin():
    sitk = pytest.importorskip("SimpleITK")
    a, b = sitk.Image([3, 3, 3], sitk.sitkFloat32), sitk.Image([3, 3, 3], sitk.sitkFloat32)
    same_geometry(a, b)
    b.SetOrigin((0, 0, 5))
    with pytest.raises(ValueError, match="Geometry"):
        same_geometry(a, b)


def test_provenance_rejects_in_sample_and_modified_mask(tmp_path):
    mask = tmp_path / "mask.nii.gz"
    mask.write_bytes(b"test")
    path = tmp_path / "provenance.json"
    record = {
        "Patient_ID": "p1",
        "mask_sha256": sha256(mask),
        "models": [{"training_subjects": ["p2"]}],
    }
    path.write_text(json.dumps(record))
    verify_mask_provenance("p1", path, mask)
    record["models"][0]["training_subjects"].append("p1")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="In-sample"):
        verify_mask_provenance("p1", path, mask)
    mask.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_mask_provenance("p1", path, mask)


def test_dice_empty_and_one_sided(config):
    prediction, reference = np.array([0, 1, 1, 3]), np.array([0, 1, 3, 3])
    result = dice_per_class(prediction, reference, config["labels"])
    assert result["IMAT"] == pytest.approx(2 / 3)
    assert np.isnan(result["Bone"])
    assert result["Overall"] == pytest.approx(2 / 3)


def test_dicom_components_and_physical_slice_order(tmp_path):
    pytest.importorskip("pydicom")
    sitk = pytest.importorskip("SimpleITK")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    from legmri.imaging import convert_dicom_group, dicom_inventory

    series, study = generate_uid(), generate_uid()
    for component in ("WATER", "FAT"):
        for index in (2, 0, 1):
            path = tmp_path / f"{component}-{2 - index}.dcm"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID, ds.StudyInstanceUID = series, study
            ds.Modality, ds.PatientID = "MR", "SYNTHETIC"
            ds.ImageType = ["DERIVED", "PRIMARY", component]
            ds.SeriesDescription = "synthetic Dixon"
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [11, -8, 30 + index * 2]
            ds.PixelSpacing, ds.SliceThickness = [0.8, 0.9], 2
            ds.Rows, ds.Columns = 8, 9
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
            ds.PixelData = np.full((8, 9), index + 10, np.uint16).tobytes()
            ds.save_as(path, enforce_file_format=True)
    inventory = dicom_inventory(tmp_path)
    assert len(inventory) == 2
    files = next(iter(inventory.values()))["files"]
    output = tmp_path / "converted.nii.gz"
    convert_dicom_group(files, output)
    image = sitk.ReadImage(str(output))
    assert image.GetSpacing() == pytest.approx((0.9, 0.8, 2))
    assert image.GetOrigin() == pytest.approx((11, -8, 30))
    assert sitk.GetArrayFromImage(image)[:, 0, 0].tolist() == [10, 11, 12]
    with pytest.raises(ValueError, match="Duplicate"):
        convert_dicom_group(files + [files[0]], tmp_path / "invalid.nii.gz")


def test_native_radiomics_extraction_and_combination(tmp_path, config):
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("radiomics")
    import pandas as pd
    from legmri.imaging import extract_features
    from legmri.workflow import combine_features
    from legmri.io import write_json

    rng = np.random.default_rng(9)
    mask = np.zeros((12, 12, 12), np.uint8)
    mask[2:10, 2:10, 2:10] = 3
    mask[2:5, 2:5, 2:5] = 1
    mask[7:10, 7:10, 7:10] = 2
    mask[5:7, 5:7, 5:7] = 4  # below the ten-voxel threshold
    folders = []
    for index in range(2):
        sid = f"synthetic-{index}"
        row = {"Patient_ID": sid}
        for channel, array in {
            "water": rng.normal(100, 30, mask.shape),
            "fat": rng.uniform(0, 200, mask.shape),
            "ff": np.full(mask.shape, 20.0),
            "predicted_mask": mask,
        }.items():
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((0.8, 0.8, 2))
            image.SetOrigin((11, -8, 30))
            path = tmp_path / f"{sid}_{channel}.nii.gz"
            sitk.WriteImage(image, str(path))
            row[channel] = str(path)
        record = tmp_path / f"{sid}.json"
        write_json(
            record,
            {
                "Patient_ID": sid,
                "mask_sha256": sha256(row["predicted_mask"]),
                "models": [{"training_subjects": ["synthetic-other"]}],
            },
        )
        row["provenance"] = str(record)
        manifest = tmp_path / f"{sid}.csv"
        pd.DataFrame([row]).to_csv(manifest, index=False)
        out = tmp_path / f"features-{index}"
        extract_features(manifest, config, out)
        extracted = pd.read_csv(out / "radiomics.csv").iloc[0]
        water = sitk.GetArrayFromImage(sitk.ReadImage(row["water"]))
        fat = sitk.GetArrayFromImage(sitk.ReadImage(row["fat"]))
        normalized_water = 100 * (water - water.mean()) / water.std(ddof=1)
        assert extracted.IMAT_Water_original_firstorder_Mean == pytest.approx(
            normalized_water[mask == 1].mean()
        )
        assert extracted.IMAT_Fat_original_firstorder_Mean == pytest.approx(fat[mask == 1].mean())
        folders.append(out)
    combined = combine_features(folders, tmp_path / "combined")
    quant = pd.read_csv(combined / "quantitative.csv")
    radio = pd.read_csv(combined / "radiomics.csv")
    assert len(quant) == len(radio) == 2
    assert quant.SMFF_mean.to_numpy() == pytest.approx([0.2, 0.2])
    assert quant.Muscle_volume_ml.iloc[0] == pytest.approx(
        (512 - 27 - 27 - 8) * 0.8 * 0.8 * 2 / 1000
    )
    assert "IMAT_Fat_original_glcm_Contrast" in radio
    assert "Bone_Fat_original_glcm_Contrast" not in radio
    assert radio.IMAT_Water_original_shape_VoxelVolume.iloc[0] == pytest.approx(27 * 0.8 * 0.8 * 2)
    provenance = json.loads((combined / "feature_provenance.json").read_text())
    assert len(provenance) == 2 and "images_sha256" in provenance[0]
    with pytest.raises(ValueError, match="duplicate"):
        combine_features([folders[0], folders[0]], tmp_path / "duplicated")
    run = json.loads((folders[1] / "run.json").read_text())
    run["config"]["extraction"]["bin_width"] = 10
    write_json(folders[1] / "run.json", run)
    with pytest.raises(ValueError, match="different settings"):
        combine_features(folders, tmp_path / "mixed-settings")
