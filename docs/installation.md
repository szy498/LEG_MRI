# Installation

Use Python 3.10–3.12 in a dedicated environment. Local checks used Python 3.12.13
on Windows. For all stages:

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev,models,imaging,segmentation]" -c constraints.txt
python scripts/install_radiomics.py
python -m pytest -q
python -m pip check
```

Extras: `imaging` adds SimpleITK/pydicom; `segmentation` adds PyTorch/nnU-Net;
`models` adds XGBoost/SHAP; `dev` adds tests, linting and package building.
`constraints.txt` pins the main scientific dependencies; compatible transitive
dependencies are resolved by pip.

## PyRadiomics v3.1.0 source

The upstream v3.1.0 tag points to commit
[`6a761c4eb01905cfcb1139b0b158547c55edd4df`](https://github.com/AIM-Harvard/PyRadiomics/tree/6a761c4eb01905cfcb1139b0b158547c55edd4df).
Its package metadata can report `3.0.1a1`, so the installer uses that exact source
commit rather than accepting an unrelated version to satisfy `==3.1.0`. The
installer provides NumPy 1.26.4 and build tools before compiling the upstream C
extensions without build isolation. It does not patch scientific source code.
Both the installed version and direct source URL/hash are recorded in `run.json`.

A C compiler is required: GCC/build-essential on Linux, or Visual Studio C++
Build Tools on Windows. If Windows compiler discovery fails, run the installer
inside an x64 developer shell. A project environment prevents these build-tool
pins from changing other projects.

## GPU training

Full-resolution segmentation needs a suitable GPU and a CUDA-compatible PyTorch
build. Install a matching PyTorch/torchvision pair using the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/),
keeping the tested versions torch 2.6.0 / torchvision 0.21.0 where practical.
Check `python -c "import torch; print(torch.cuda.is_available())"` before training.
The repository does not install a GPU driver or change system CUDA.

GPU memory requirements depend on patch size, batch size and internal padding.
The CPU smoke test uses a reduced network. The `--processes` argument controls preprocessing
and export workers; nnU-Net augmentation workers are controlled separately by
`nnUNet_n_proc_DA`.

## Existing feature tables only

```bash
python -m pip install -e ".[dev,models]" -c constraints.txt
leg-mri demo --config configs/study.yaml --output outputs/demo
```

No GPU, MRI libraries or external model downloads are required for this route.
For LR/RF without SHAP, install the base package and configure `cv.models: [LR, RF]`
and `cv.shap: false`. Statsmodels is used for the calibration plot's smoother,
not for mixed-effects models. To build distributable packages:

```bash
python -m build
```

Keep study inputs and outputs outside the tracked source tree or under the ignored
`data/` and `outputs/` folders. The example files contain synthetic records only.
