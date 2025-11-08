### Installation & Setup

This project uses a Conda environment to manage dependencies.

**1. Prerequisites**
You must have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed.

**2. Install the Environment**
```bash
conda env create -f environment.yml
```
**3. Activate the Environment**
```bash
conda activate sis
```
**4. Copy The Configuration file**
```bash
cp config.example.yml config.yml
```
**5. Running the Program**
```bash
python main.py
```
**Project Structure**
```
├── main.py             # The main script to run the program
├── config.yml          # Configuration file (Batch size, LR, Paths, etc.)
├── environment.yml     # Conda environment dependency file
|
├── vae.py              # VAE model architecture (Encoder, Decoder, loss_function)
├── experiment.py       # Handles the training/validation loops and logging
├── dataset.py          # Data loading (CustomDataset, VAEDataModule)
|
├── shamir.py           # Functions for Shamir's Secret Sharing (SSS)
├── graph.py            # Utility functions for plotting graphs (Loss, LR, Comparison)
├── utility.py          # Helper functions (load_config, save_image, get_next_version_dir)
|
├── Data/               # Folder for training images (path defined in config.yml)
└── logs/               # Default output folder for experiment versions (logs/version_X)
```
