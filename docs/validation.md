# Software tests

Tests use synthetic data to check numerical operations, data flow and model
serialization. Install the development and optional dependencies as described in
[installation](installation.md).

## Unit tests

```bash
python -m pytest -q
python -m ruff check src tests scripts
python -m pip check
```

Coverage includes:

- DICOM component grouping, physical slice ordering and native image geometry;
- tissue volumes, ratios and FF units;
- Water z-score scaling by 100 and unchanged Fat intensities in PyRadiomics;
- ROI thresholds, feature-table joins and mask provenance;
- subject-level segmentation partitions, weighted CE and odd-size network gradients;
- training-fold imputation, correlation pruning, stability and L1 selection;
- seven feature strategies, empty feature sets and saved-model prediction;
- participant-level repeated predictions, bootstrap metrics and DCA arithmetic;
- SHAP additivity for LR, RF and XGBoost.

## Image-to-ML smoke test

```bash
python scripts/smoke_imaging.py --output outputs/imaging_smoke
```

This test generates 24 synthetic three-channel volumes, performs nnU-Net planning
and preprocessing, trains two small six-level networks, predicts OOF/test masks,
extracts native-space features, and evaluates all seven strategies with all three
classifiers. The test uses one training step per segmentation model, three outer
and inner ML folds, one repetition, and compact parameter grids.

Assertions check participant coverage, mask/model provenance, native geometry,
radiomic outputs, disjoint ML splits and finite probabilities. Outputs include
`validation.json`, model files and 504 held-out prediction rows.

## Feature-table demo

```bash
leg-mri demo --config configs/study.yaml --output outputs/demo
```

The demo generates 80 synthetic participant records and runs all model strategies
with reduced grids. The study configuration is maintained separately in
`configs/study.yaml`.

## Continuous integration

The GitHub Actions workflow installs dependencies, runs unit tests and the
image-to-ML smoke test, checks style and dependency compatibility, and builds the
Python source and wheel distributions.
