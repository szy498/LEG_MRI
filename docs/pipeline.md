# Image-to-model workflow

## 1. Image input and reference labels

Use one row per participant in `subjects.csv`:

```csv
Patient_ID,water,fat,ff,manual_mask
sub-001,images/sub-001_water.nii.gz,images/sub-001_fat.nii.gz,images/sub-001_ff.nii.gz,labels/sub-001.nii.gz
```

All four volumes must have the same size, spacing, origin and direction. No
implicit registration or label renumbering is performed. Default labels are
background=0, IMAT=1, subcutaneous fat=2, muscle=3 and bone=4.
FF percentages use `ff_divisor: 100`; fractions use `1`.
Features are measured from original volumes, not nnU-Net's preprocessed arrays.

For reconstructed classic single-frame DICOM series:

```bash
leg-mri dicom-index data/sub-001/dicom --output data/sub-001/inventory.json
leg-mri dicom-convert data/sub-001/inventory.json GROUP_ID --output data/sub-001/water.nii.gz
```

Repeat conversion with the selected Fat and FF groups. Inventory groups preserve
series, component and echo distinctions. Select the correct component using the
scanner's series description and ImageType; the code does not guess that label.
Slices are ordered by patient-space position. Duplicate/irregular positions and
sheared stacks are rejected. Enhanced multiframe acquisitions should first be
converted with a suitable tool such as dcm2niix. Scanner raw k-space reconstruction
and correction of misregistered components are not part of this repository.

## 2. Segmentation

```bash
leg-mri seg-prepare data/subjects.csv --config configs/study.yaml --work outputs/seg
leg-mri seg-plan --work outputs/seg --processes 4
leg-mri seg-train --work outputs/seg --fold 0 --device cuda
```

Train folds 0–4, changing `--fold` for each command. Or use `study-run` to execute
the sequence. `--resume` resumes a fold from its latest checkpoint.

The default split is 80:20 by participant; the development set has five folds.
For 114 participants, rounding yields 91 development and 23 test participants.
No test references enter the dataset fingerprint or training directories.
To supply predefined allocations, use `seg-prepare --splits splits.json`:

```json
{
  "development": ["sub-001", "sub-002"],
  "test": ["sub-003"],
  "folds": [
    {"train": ["sub-002"], "val": ["sub-001"]},
    {"train": ["sub-001"], "val": ["sub-002"]}
  ]
}
```

This illustrates the schema with two folds; the study config requires five.

nnU-Net estimates preprocessing spacing and normalization from the development
data. The custom trainer supplies the manuscript's six-level 3D convolutional
network, Adam optimizer, and Dice plus IMAT-weighted voxel-mean cross entropy.
It uses nnU-Net augmentation and sliding-window inference. Use `leg-mri` commands
to load the custom trainer and network.

Default patch `[71,432,432]` is in nnU-Net's transposed spatial order. With the
declared stride pattern it is internally zero-padded to `[96,448,448]`; output
logits are cropped back before loss/evaluation. Memory demand is higher than the
unpadded shape. Store alternative patch/batch settings in a separate experiment
configuration.

## 3. Held-out masks and segmentation metrics

```bash
leg-mri seg-predict --work outputs/seg --cohort development --device cuda
leg-mri seg-predict --work outputs/seg --cohort test --device cuda
leg-mri seg-evaluate outputs/seg/predictions/test/manifest.csv --config configs/study.yaml --output outputs/seg_metrics/test
```

Development subjects use their own held-out fold model; test subjects use all
five models averaged in logit space by nnU-Net. The default uses the fixed final
epoch, so an OOF participant's reference mask does not select the checkpoint.
All predictions are exported onto the original image grid.

Each mask has a JSON record with participant ID, source model training IDs,
checkpoint hashes and mask hash. Feature extraction verifies the record.
`manifest.csv` connects masks, source images and manual references. Dice is
computed separately per participant/tissue, then summarized; both-empty classes
are undefined and excluded, one-sided empty classes receive zero. Overall Dice
is the mean of available foreground-class Dice values, not pooled voxel Dice.

## 4. Native-space features

```bash
leg-mri extract outputs/seg/predictions/development/manifest.csv --config configs/study.yaml --output outputs/features_development
leg-mri extract outputs/seg/predictions/test/manifest.csv --config configs/study.yaml --output outputs/features_test
leg-mri features-combine outputs/features_development outputs/features_test --output outputs/features
```

Outputs include quantitative/radiomics CSVs, per-ROI extraction QC, image/mask
hashes, prediction provenance and extraction configuration. ROIs below ten voxels are omitted
from radiomics; resulting missing values are handled within ML training folds.
`--quantitative-only` skips PyRadiomics when only tissue measurements are needed.

The default uses Water and Fat radiomics, original-image first-order/shape/GLCM
features, native spacing, fixed bin width 25, distance 1 and 13 undirected 3D GLCM
directions. Water is z-scored over each image volume (sample SD) and multiplied
by 100 before discretization; Fat is unnormalized. The QC table reports occupied gray-level
bins. FF maps support segmentation and direct quantitative measurements, not
radiomic extraction in the study configuration.

Volumes are in mL; soft tissue means muscle + IMAT + subcutaneous fat, excluding
bone. The 14 ML quantitative features are ratios and muscle FF summaries.
`SMFF_wf = sum(Fat in muscle) / sum(Water + Fat in muscle)`; scanner FF statistics
remain a separate measurement. Diagnostic absolute volumes are not automatically
added to the Table S6 predictor panel.

## 5. Machine learning

```bash
leg-mri model-cv --clinical data/clinical.csv --quantitative outputs/features/quantitative.csv --radiomics outputs/features/radiomics.csv --config configs/study.yaml --output outputs/ml
```

The clinical file defines the modeling cohort (107 participants in the study).
The feature tables may include all 114 segmentation participants. Image records
are matched by ID; the seven participants without clinical-modeling eligibility
are not included merely because their images were segmented.

Segmentation development/test and downstream ML cross-validation are distinct
partitions. The segmentation test set is not automatically an independent ML
test set. ML reports repeated nested CV over the clinical cohort, using fold-local
preprocessing and selection. See the README for seven strategies and outputs.

`study-run` executes the configuration supplied at startup. Use `model-fit` to
refit the selected strategy and classifier after the cross-validation comparison.
