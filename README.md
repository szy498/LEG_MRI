# LEG_MRI
Function and details of the code:

Preprocess stage:

split-new.py:
The core purpose of this script is: To automatically split raw Dixon MRI DICOM data into multiple independent NIfTI (.nii.gz) image series.

Overall workflow :

Read raw DICOM images
-> Convert MRI data into 3D arrays
-> Split the volume into fixed slice blocks
-> Separate different Dixon MRI series
-> Preserve medical spatial information
-> Save as .nii.gz files

Its main objective is: To convert raw hospital MRI data into standardized nnUNet-compatible training data.

final_output:
Five different MRI image series, including:Water image，Fat image，Fat Fraction image，Other Dixon-related series

output format : .nii.gz

output directory structure: dixon_water, dixon_fat, dixon_ff, series_3, series_4 (series_3,series_4 is other Dixon-related series)

Training stage:

trainer.py:

The core purpose of this script is: To train a medical image segmentation model.

It's built upon nnU-Net and forms a complete deep learning training framework for medical imaging.

Over workflow:

Load training data
-> Apply data augmentation
-> Build neural network
-> Forward propagation
-> Compute loss function
-> Backpropagation and parameter update
-> Validate model performance
-> Save the best-performing model

The pipeline includes:
  
  MRI image loading
  
  Data augmentation
  
  Dice Loss + Cross Entropy Loss
  
  Mixed precision training
  
  Validation Dice evaluation
  
  Checkpoint saving
  
  Multi-GPU training support

The final output includes a trained medical image segmentation model as well as Best model weight files, Checkpoints, Validation Dice metrics, Training logs, Model performance results.

Typical output files include: checkpoint_best.pth, checkpoint_final.pth, training_log.txt.

Predicting stage:

The main purpose of this code is constructing glucose metabolism disorder predicting meodel based on image feature(Quantitative + Radiomics), evaluating the performance of three classifiers (LR, RF, XGBoost) under seven feature strategies within a nested cross‑validation framework, and conduct feature interpretability analysis using SHAP.

input:

  1.Radiomics feature matrix (one subject per row, including Patient_ID)  radiomics.xlsx ,etc.
 
  2.Quantitative feature metrix (rate of volume, SMFF statistics, etc. 
  
  3.label file with (Patient_ID,Diabetes_Status,BMI,Age,Sex)

output:

  1.NestedCV_Results:Detailed performance metrics for each outer fold, each model, and each feature strategy
  
  2.Paper-level summary table (Bootstrap 95% CI)
  
  3.outer_test_predictions.npz – Outer-fold pooled predictions of the optimal model (for DeLong test)
  
  4.Feature_Stability.xlsx – LASSO feature selection stability (frequency of each feature being selected)
  
  5.SHAP.xlsx – SHAP feature importance ranking of the optimal model
  
  6._ROC.png – ROC curves of the optimal model for each strategy
  
  7.Univariate_LR_Results.xlsx – Univariate logistic regression results for direct quantitative features (OR, CI, p-value)

Overall Code Structure and Writing Logic

Module1 : Radiomics Feature Modeling (Lines 1–250)

  1.Configuration and Data Loading (Lines 1–70)
  
  2. Custom ROC Plotting Function
  
  3. Nested Cross‑Validation Main Loop (Lines 75–175)
  
  4. SHAP Feature Importance Calculation (Lines 80–120, embedded in main loop)
  
  5. Post‑processing and Summarization (Lines 180–250)

Module 2: Direct Quantitative Feature Modeling (Lines 255–500)

Module 3: All‑Feature Modeling (Lines 503–670, commented out)

radiocom_rongyu.py

Core Purpose:
Evaluate the redundancy and stability of radiomics features – through multiple random data splits, identify which features can be stably retained across different training set splits (i.e., not filtered out due to collinearity).

Inputs

  1.Radiomics feature matrix
  
  2.Label file (used only for stratified splitting, not for feature filtering)

Outputs

  1.Stratified 70%/30% split for each run
  
  2.List of features retained in each run
  
  3.Summary: frequency each feature is retained across n_splits runs
  
  4.Bar chart of top 30 most stable features

Overall Code Structure and Writing Logic

  Step 1 (Lines 1–25): Configuration
  
  Step 2 (Line 30): Main loop
  
  Step 3 (Lines 65–90): Summarize stability
  
  Step 4 (Lines 92–110): Visualization
