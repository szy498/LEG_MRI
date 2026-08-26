import json

import numpy as np
import pandas as pd
import pytest

from legmri.segmentation import prepare_dataset, validate_split


def test_subject_partition_validation():
    split = {
        "development": ["a", "b"],
        "test": ["c"],
        "folds": [{"train": ["a"], "val": ["b"]}, {"train": ["b"], "val": ["a"]}],
    }
    validate_split(split, ["a", "b", "c"], 2)
    split["test"].append("a")
    with pytest.raises(ValueError):
        validate_split(split, ["a", "b", "c"], 2)


def test_dataset_preparation_excludes_test_from_fingerprint(tmp_path, config):
    sitk = pytest.importorskip("SimpleITK")
    config["segmentation"].update(folds=2)
    array = np.arange(125).reshape(5, 5, 5) % 5
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.SetSpacing((0.82, 0.82, 1.0))
    path = tmp_path / "source.nii.gz"
    sitk.WriteImage(image, str(path))
    subjects = pd.DataFrame(
        [
            {
                "Patient_ID": f"sub-{i:03d}",
                "water": str(path),
                "fat": str(path),
                "ff": str(path),
                "manual_mask": str(path),
            }
            for i in range(10)
        ]
    )
    manifest = tmp_path / "subjects.csv"
    subjects.to_csv(manifest, index=False)
    work = tmp_path / "work"
    split = prepare_dataset(manifest, config, work)
    raw = work / "nnUNet_raw" / "Dataset501_ThighDixon"
    assert len(list((raw / "labelsTr").glob("*"))) == 8
    assert len(list((raw / "imagesTr").glob("*"))) == 24
    assert len(list((raw / "imagesTs").glob("*"))) == 6
    assert len(split["test"]) == 2
    assert json.loads((raw / "dataset.json").read_text())["numTraining"] == 8


def test_weighted_ce_has_voxel_denominator():
    torch = pytest.importorskip("torch")
    from legmri.network import IMATDiceCELoss

    logits = torch.zeros(1, 5, 1, 1, 2, requires_grad=True)
    target = torch.tensor([[[[1, 0]]]])
    loss = IMATDiceCELoss()(logits, target)
    p, smooth = 0.2, 1e-5
    expected_dice = np.mean(
        [(2 * p + smooth) / (2 * p + 1 + smooth)] + [smooth / (2 * p + smooth)] * 3
    )
    expected_ce = np.log(5) * (1.5 + 1) / 2
    assert loss.item() == pytest.approx(expected_ce + 1 - expected_dice, rel=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_odd_shape_network_forward_backward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("dynamic_network_architectures")
    from legmri.network import PaddedPlainConvUNet

    torch.set_num_threads(2)
    net = PaddedPlainConvUNet(features=(2, 4, 8))
    x = torch.randn(1, 3, 9, 17, 19)
    output = net(x)
    assert output.shape == (1, 5, 9, 17, 19)
    output.mean().backward()
    # Upstream retains unused auxiliary heads even when deep supervision is off.
    active = [p for n, p in net.named_parameters() if "decoder.seg_layers.0." not in n]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
