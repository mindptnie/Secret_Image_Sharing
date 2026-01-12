# Secret Image Sharing using VQ-VAE

This project implements **Secret Image Sharing (SIS)** by integrating **Vector Quantized Variational Autoencoder (VQ-VAE)** with **Shamir's Secret Sharing (SSS)**. Images are encoded and compressed via the Deep Learning model into discrete codebook indices, which are then distributed into multiple "shares." To reconstruct the original image, a specified threshold of these shares must be combined.

## Prerequisites

* **OS:** Linux (Recommended) or Windows
* **Python:** 3.10 (Required)
* **Conda:** You must have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed.
* **GPU:** NVIDIA GPU with CUDA support is highly recommended for training.

## Installation

**1. Clone the Repository**
Download the project to your local machine.

**2. Install the Environment**
Create the environment using the provided YAML file to install all required libraries (PyTorch, OpenCV, etc.).
```bash
conda env create -f environment.yml
```
**3. Activate the Environment**
```bash
conda activate sis
```
**4. Configuration Copy the example configuration file. You can edit config.yml to change model parameters, data paths, or batch sizes.**
```bash
cp config.example.yml config.yml
```
## Dataset

You can download the dataset from [here](https://www.kaggle.com/datasets/ashwingupta3012/human-faces). or Use other dataset that you want.
Note: The preprocessing script is designed to extract and process only .png, .jpg, .jpeg, .bmp, .tiff images from the source directory. All other file formats will be ignored to ensure model compatibility.

## Training & Automatic Testing

Run the main pipeline to train the model and perform secret sharing tests.
```bash
python main.py
```
Outputs (logs, model checkpoints, reconstructed images) are saved in the logs/version_X directory.

### Evaluation (Optional)
- Automated Experiments: Run grid search experiments for different codebook sizes/dimensions.
```bash
python automation.py
```
- Test Pre-trained Models: To test a specific pre-trained model version:
```bash
python testmodel.py
```

## Project Structure
```
├── main.py             # The main script to run the program
├── automation.py       # Script for running automated experiments (Grid Search)
├── testmodel.py        # Script for testing pre-trained models
├── testmodel_avg.py    # Script for calculating average metrics over the dataset
├── testmodel_n.py      # Script for testing pre-trained models (multiple images)
├── testmodel_v.py      # Script for testing pre-trained models (multiple versions)
├── config/             # Configuration files for specific models
├── config.yml          # Global configuration file (Batch size, LR, Paths, etc.)
├── environment.yml     # Conda environment dependency file
├── model/              # Model architectures (VQ-VAE, VAE, ResNet, GAN)
├── experiment.py       # Handles the training/validation loops and logging
├── dataset.py          # Data loading (CustomDataset, VAEDataModule)
├── shamir.py           # Functions for Shamir's Secret Sharing (SSS)
├── graph.py            # Utility functions for plotting graphs (Loss, LR, Comparison)
├── utility.py          # Helper functions (load_config, save_image, get_next_version_dir)
├── Data/               # Folder for training images
└── logs/               # Output folder for experiments (logs/version_X)
```





