### Secret Image Sharing (SIS)
This project demonstrates a secure method for image sharing by combining a Variational Autoencoder (VAE) with Shamir's Secret Sharing (SSS).

The core concept is to avoid sharing the raw image file. Instead, we first compress the image into its most essential features (a Latent Vector) using a VAE. This Latent Vector is then treated as the "secret" and securely split into parts using SSS.
___
### How it Works
The program follows these main steps:

1. Encoding: The original image is fed into the VAE's Encoder.

2. Latent Space: The Encoder compresses the image into a Latent Vector (a small vector representing the image's key features). This vector is our "secret".

3. Splitting (SSS): This latent_dim is processed by SSS to split the secret into n shares.

4. Reconstruction (SSS): A minimum of r shares (the defined threshold) are collected.

5. Combining: SSS combines the r shares to reconstruct the original Latent Vector.

6. Decoding: The reconstructed Latent Vector is fed into the VAE's Decoder.

7. Output: The Decoder attempts to generate the original image from this vector.

As long as fewer than r shares are collected, the correct Latent Vector cannot be reconstructed, keeping the image secure.

### Installation & Setup
This project uses a Conda environment to manage dependencies.

1. Prerequisites
You must have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed.

2. Install the Environment 
```
conda env create -f environment.yml
```
3. Once the installation is complete, activate the new environment (named sis):
```
conda activate sis
```
### Running the Program
```
python main.py
```
### Project Structure
.
├── main.py # The main script to run the program
├── vae.py # VAE model code (Encoder, Decoder) and training functions
├── sharmir.py # (Assumed) Functions for Shamir's Secret Sharing
├── graph.py # (Assumed) Utility functions for plotting images
├── environment.yml # Conda environment dependency file
├── Pic # Image for use to train
└── ...
