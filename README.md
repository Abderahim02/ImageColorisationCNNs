# Image Colorisation with CNNs

Deep learning-based image colorisation project using different CNN architectures.

## Overview
This project implements various approaches for automatic image colorisation:
- Base CNN model
- VGG-based architecture
- Lab color space conversion


# Project Structure
```  
├── lab_model
│   ├── PREDICTIONS
│   ├── base_model.py
│   ├── help_functions.py
│   ├── load_data.py
│   ├── main.py
│   ├── model_with_vgg16.py
│   └── train.py
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
```

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
python lab_model/main.py base /path/to/data --model_path models/base_model.h5 --out_dir PREDICTIONS

# Run inference with VGG model
python lab_model/main.py vgg_based /path/to/data --model_path models/vgg_model.h5 --out_dir predictions
```

##
Models are availables here :

```bash
wget https://www.dropbox.com/scl/fi/pxl1qh7ysgb01u62unbe0/lab_model_256x256_75_epochs_mse_10000imgs_benchmark.keras?rlkey=zwoxw6w2dn4f48rqy82xnj80d&st=zlbyk6ca&dl=0

wget https://www.dropbox.com/scl/fi/plsuk8hbhvngd06z2cc3j/lab_vgg16_model_256x256_75_epochs_mse_10000imgs_benchmark.keras?rlkey=0tivkxpiry71r684qbbfyb3ur&st=6m3leb2q&dl=0

wget https://www.dropbox.com/scl/fi/gb2qnbogpzdpfnjgcbwk0/rgb_model_256x256_75_epochs_mse_10000imgs_benchmark.keras?rlkey=v3sbku3k4zh69vwj82oqpj1ho&st=iui336k0&dl=0

wget https://www.dropbox.com/scl/fi/n1ma8m4w4fs5zhugqf85e/rgb_model_256x256_75_epochs_perceptual_10000imgs_benchmark.keras?rlkey=fxup0cd1pi1jzln4y1gnlduid&st=xsn5jpdy&dl=0

```
