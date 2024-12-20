# Image Colorisation with CNNs

Deep learning-based image colorisation project using different CNN architectures.

## Overview
This project implements various approaches for automatic image colorisation:
- Base CNN model
- VGG-based architecture
- Lab color space conversion


# Project Structure
├── lab_model
│   ├── PREDICTIONS
│   ├── base_model.py
│   ├── help_functions.py
│   ├── load_data.py
│   ├── main.py
│   ├── model_with_vgg16.py
│   └── train.py
├── Makefile
├── MODELS
├── README.md
├── rgb_model
│   ├── base_model.py
│   ├── help_functions.py
│   ├── load_data.py
│   ├── main.py
│   ├── model_with_mask.py
│   ├── PREDICTIONS
│   ├── train.py

## Installation
```bash
# Clone the repository
git clone git@github.com:Abderahim02/ImageColorisationCNNs.git
cd ImageColorisationCNNs

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


# Run inference with base model
python main.py base /path/to/data --model_path models/base_model.h5 --out_dir PREDICTIONS

# Run inference with VGG model
python main.py vgg_based /path/to/data --model_path models/vgg_model.h5 --out_dir predictions
```

