# Methods and implementation

Parameters are specified in `configs/study.yaml`. This reference describes the
segmentation, feature extraction and machine-learning procedures implemented by
the package.

## Segmentation

| Component | Implementation |
|---|---|
| Inputs | Water, Fat and FF images on the same 3D grid |
| Label IDs | Background=0, IMAT=1, subcutaneous fat=2, muscle=3, bone=4 |
| Subject split | 80:20 development/test; fivefold development CV |
| Network | Six-level 3D U-Net; channels 32/64/128/256/320/320; two 3×3×3 convolutions per stage |
| Normalization/activation | InstanceNorm3d and LeakyReLU |
| Patch and batch | 71×432×432 voxels; batch size 2 |
| Spatial padding | Internal stride-compatible padding to 96×448×448; logits cropped to the input patch |
| Optimizer | Adam, initial learning rate 0.001, weight decay 3e-5 |
| Schedule | Polynomial decay, power 0.9, 120 epochs |
| Epoch length | 250 training and 50 validation iterations |
| Objective | Foreground soft Dice loss plus weighted cross entropy, applied to the full-resolution output |
| CE weights | IMAT=1.5; other classes=1.0; weighted losses averaged over voxels |
| Downstream checkpoints | Fixed final epoch |
| Downstream predictions | Development OOF masks; test ensemble of five fold models |

`network.IMATDiceCELoss` averages soft Dice across foreground classes and
participants. Cross entropy uses the voxel count as its denominator.
`trainer.LegMRITrainer` extends nnU-Net 2.6.2, with one full-resolution output and
deep supervision disabled. Prediction uses nnU-Net sliding windows, mirroring
and logit averaging, followed by export to the native image grid.

`segmentation.prepare_dataset` stores participant allocations and separates test
references from training/fingerprint inputs. `segmentation.predict_heldout`
records the training IDs and checkpoint hashes for each predicted mask.
Dice is calculated per participant and foreground class. Both-empty classes are
undefined; one-sided empty classes have Dice zero. Overall Dice is the mean of
available foreground-class values.

## Native-space feature extraction

| Component | Setting |
|---|---|
| Radiomic images | Water and Fat |
| Water normalization | Per-volume z-score using sample SD, multiplied by 100 |
| Fat normalization | None |
| Discretization | Fixed bin width 25 after intensity preprocessing |
| Spatial resampling | None |
| Features | Original-image first-order statistics, 3D shape and GLCM |
| GLCM | Distance 1; symmetric matrices; feature averaging over 13 undirected 3D directions |
| Minimum ROI | 10 voxels |
| Implementation | PyRadiomics v3.1.0 source; source pin documented in installation instructions |

`imaging.extract_features` operates on predicted masks and native input images.
Mask provenance is checked before extraction, and `extraction_qc.csv` records ROI
sizes and occupied gray-level bins. Missing tissue measurements are retained as
missing values for training-fold imputation.

Volumes are in mL. Soft tissue comprises muscle, IMAT and subcutaneous fat;
total fat comprises IMAT and subcutaneous fat. FF percentages are divided by 100
for quantitative measurements. Muscle FF summaries use the scanner FF map, with
population SD for `SMFF_std`. The signal-based quantity is
`SMFF_wf = sum(Fat within muscle) / sum(Water + Fat within muscle)`.
The quantitative ML panel contains the fourteen variables listed in the README.

## Machine learning

`modeling.load_model_data` joins tables by participant ID. The clinical table
defines the modeling cohort; Age, Sex and BMI are the clinical predictors.
Seven feature strategies are evaluated with LR, RF and XGBoost on identical
outer partitions.

| Component | Setting |
|---|---|
| Outer CV | Stratified fivefold CV, two repetitions, seed 42 |
| Inner tuning | Stratified fivefold CV, two repetitions, seed 7 |
| Missing values/scaling | Training-fold medians, means and SDs |
| Correlation filtering | Absolute Spearman correlation >0.9; declared priorities, then column order |
| Shape exclusions | Boundary/diameter-related suffixes listed in the config |
| Texture stability | Retention across ten stratified 70% training subsets; frequency >0.8 |
| L1 selection | Class-balanced logistic regression, liblinear, 5,000 iterations |
| L1 grid | Twenty log-spaced C values from 0.001 to 10, jointly tuned with classifier parameters |
| Integrated imaging | Per-domain selection in five training-only subfolds; candidates retained at frequency >0.8, then combined, re-pruned and L1-selected |
| Class weighting | Balanced LR/RF; XGBoost positive-class weight recomputed on each fit partition |
| Classification threshold | 0.5 |

All learned preprocessing and selection steps are fitted inside the corresponding
training partition using `selection.StudySelector`. Clinical covariates are
retained. Strict frequency >0.8 requires retention in all five integration
subfolds. A fold with no selected imaging feature uses an intercept-only model
or the clinical covariates, according to its strategy.

Repeated outer-test probabilities are averaged per participant before performance
inference. Confidence intervals use 2,000 stratified participant bootstrap
samples and condition on the fitted CV models. Inner AUC is a tuning summary;
outer-test predictions provide the CV performance estimates.

SHAP values use the corresponding outer-fold model and held-out participants.
LR backgrounds contain training rows. Output scales are recorded: probability
for RF and log odds for LR/XGBoost. Global mean-absolute SHAP includes zero
contributions for folds in which a feature was not selected.

## Reproducibility outputs

Runs retain configurations, participant splits, input/checkpoint hashes,
dependency versions, selected features and fitted pipelines. Seeds are set for
data splitting and model fitting. Asynchronous augmentation and GPU kernels can
produce numerical differences across hardware; `nnUNet_n_proc_DA=0` selects
synchronous augmentation.
