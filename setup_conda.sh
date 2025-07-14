#!/bin/bash

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment with Python 3.10
conda create -n b2txt25 python=3.10 -y

# Activate the new environment
conda activate b2txt25

# Install PyTorch with conda (CPU version for Apple Silicon)
conda install pytorch torchvision torchaudio -c pytorch -y

# Install packages using conda
conda install -c conda-forge \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    tqdm \
    h5py \
    omegaconf \
    editdistance \
    jupyter \
    -y

# Install packages that might not be available in conda-forge using pip
pip install \
    redis==5.2.1 \
    g2p_en==2.1.0 \
    huggingface-hub==0.33.1 \
    transformers==4.53.0 \
    tokenizers==0.21.2 \
    accelerate==1.8.1 \
    -e .

# Install bitsandbytes if available (skip if not available)
pip install bitsandbytes==0.42.0 || echo "bitsandbytes not available for this platform, skipping..."

echo
echo "Setup complete! Verify it worked by activating the conda environment with the command 'conda activate b2txt25'."
echo 