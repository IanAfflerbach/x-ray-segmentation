# MIMIC CXR X-Ray Segmentation
Codebase for creating an x-ray image lung segmentation dataset and training a U-Net based model for predicting these segmentation masks
## Initializing Environment
Use the `requirements.txt` file to create an anaconda environment for this project
## Downloading Dataset
NOTE: You will need a Physionet Account and with access to the datasets required
- Within the `data` folder, use `gather_xray_seg_data.py` to collect and organize the images
- Similar actions can be taken in the `lateral_data` folder  but this data can not be used for training
## Training Model
Run `train.py` after data is downloaded to begin training the model, takes about ~20 min with default settings
