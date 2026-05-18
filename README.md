# LEG_MRI
To develop an automated thigh Dixon MRI phenotyping pipeline for early dysglycemia screening and evaluate its clinical interpretability via static metabolic profiling and dynamic physiological mapping.An automated 3D convolutional neural network based on the nnU-Net framework was implemented to delineate skeletal muscle, IMAT, subcutaneous fat, and bone from the thigh Dixon MRI. Inputs included water, fat, and fatfraction (FF) images(27)

The function of the code:

1.split-new.py:
The core purpose of this script is: To automatically split raw Dixon MRI DICOM data into multiple independent NIfTI (.nii.gz) image series.

Overall workflow:
Read raw DICOM images
-> Convert MRI data into 3D arrays
-> Split the volume into fixed slice blocks
-> Separate different Dixon MRI series
-> Preserve medical spatial information
-> Save as .nii.gz files

Its main objective is: To convert raw hospital MRI data into standardized nnUNet-compatible training data.

final_output:
Five different MRI image series, including:Water image，Fat image，Fat Fraction image，Other Dixon-related series

output format: .nii.gz
output directory structure: dixon_water, dixon_fat, dixon_ff, series_3, series_4 (series_3,series_4 is other Dixon-related series)

2.trainer:
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

The final output includes a trained medical image segmentation model as well as Best model weight files, Checkpoints, Validation Dice metrics, Training logs, Model performance results

Typical output files include: checkpoint_best.pth, checkpoint_final.pth, training_log.txt


