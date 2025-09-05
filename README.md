# AI-Driven Flood Prediction Using Sentinel-1 SAR Images

![Project: FLOOD\_PREDICTION](https://img.shields.io/badge/project-FLOOD_PREDICTION-blue)
![Language: Python](https://img.shields.io/badge/language-Python-3776AB)
![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-EA4C2E)

> Predict per-pixel flood probability maps from Sentinel-1 SAR (VV & VH) images using a CNN. Hyperparameters are optimized with a Genetic Algorithm (GA).

## Table of Contents

* [Project Overview](#project-overview)
* [Features / Highlights](#features--highlights)
* [Repository Structure](#repository-structure)
* [Dataset](#dataset)
* [Getting Started](#getting-started)

  * [Requirements](#requirements)
  * [Install](#install)
  * [Environment notes](#environment-notes)
* [Usage](#usage)

  * [Preprocessing](#preprocessing)
  * [Train baseline CNN](#train-baseline-cnn)
  * [Run GA optimization](#run-ga-optimization)
  * [Evaluation & Visualizations](#evaluation--visualizations)
* [Scripts (what they do)](#scripts-what-they-do)
* [Model & Genetic Algorithm (brief)](#model--genetic-algorithm-brief)
* [Results & Artifacts](#results--artifacts)
* [Reproducibility](#reproducibility)
* [License](#license)
* [Citing / References](#citing--references)
* [Contact / Maintainership](#contact--maintainership)
* [Tags / GitHub topics](#tags--github-topics)

## Project Overview

This repository implements a pipeline to predict floods from Sentinel-1 SAR imagery (VV and VH polarizations). The workflow includes pairing SAR images with label masks, preprocessing into NumPy arrays for fast I/O, training a pixel-wise CNN that outputs flood probability maps, and optimizing hyperparameters using a Genetic Algorithm (GA).

## Features / Highlights

* End-to-end pipeline: pairing → preprocessing → training → optimization → evaluation
* Uses Sentinel-1 SAR VV & VH inputs to produce per-pixel flood probability maps
* CNN baseline implemented in PyTorch (flexible, modular)
* Genetic Algorithm for hyperparameter search (learning rate, dropout, batch size, etc.)
* Saved model weights, training visualizations and evaluation metrics

## Repository Structure

```
FLOOD_PREDICTION/
├── sen12flood/                     # Raw dataset (DO NOT push to GitHub)
│   ├── sen12floods_s1_source/
│   └── sen12floods_s1_labels/
├── processed_dataset/              # .npy files created by preprocessing
├── models/                         # Saved trained model weights (.pth)
├── visualizations/                 # Training plots, confusion matrices, ROC etc.
├── 01_match_label_sar_pairs.py     # Pair SAR images and flood labels
├── 02_preprocess_sar_label_pairs.py# Preprocess and save .npy arrays
├── 03_train_model.py               # Train baseline CNN model
├── 04_CNN_GA.py                    # Genetic Algorithm hyperparameter tuner
├── requirements.txt                # Python deps (pip install -r requirements.txt)
├── valid_label_sar_pairs.txt       # Matched file pairs list
└── README.md
```

## Dataset

This project uses a local copy of the **SEN12-FLOOD** dataset (Sentinel-1 SAR images and flood labels). The repository expects the dataset root at `./sen12flood/` with `sen12floods_s1_source/` and `sen12floods_s1_labels/` subfolders. **Do not** include raw dataset files or large GeoTIFF/GeoJSON files in the GitHub repo. Use `.gitignore` to exclude `sen12flood/`.

If you do not have the dataset locally, please obtain it from the original dataset distribution (see [Citing / References](#citing--references)).

## Getting Started

### Requirements

* Python 3.8+
* PyTorch (tested with 1.10+ but newer versions should work)
* numpy, rasterio, geopandas, scikit-learn, opencv-python, tqdm
* deap (or any GA library used in `04_CNN_GA.py`) or a custom GA implementation

Example `requirements.txt` (already included in repo):

```text
numpy
torch
torchvision
rasterio
geopandas
scikit-learn
opencv-python
matplotlib
tqdm
deap
pandas
```

### Install

```bash
# create a venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows PowerShell

pip install -r requirements.txt
```

### Environment notes

* GPU: training benefits greatly from a CUDA-capable GPU. Ensure CUDA and the correct PyTorch build are installed.
* Memory: preprocessing uses rasterio and converts GeoTIFFs into NumPy arrays. Have enough disk space for `processed_dataset/`.

## Usage

Run the pipeline in the following order.

### Preprocessing

1. Match SAR and label files (generates `valid_label_sar_pairs.txt`):

```bash
python 01_match_label_sar_pairs.py
```

2. Convert matched pairs to `.npy` arrays for fast loading (saves to `processed_dataset/`):

```bash
python 02_preprocess_sar_label_pairs.py
```

> Both scripts accept CLI args (path overrides, verbosity) — see their header docstrings or `--help` for details.

### Train baseline CNN

Train the baseline model using preprocessed `.npy` files:

```bash
python 03_train_model.py --data_dir processed_dataset --epochs 50 --batch_size 16
```

Common flags (examples):

* `--data_dir`: path to processed `.npy` dataset
* `--epochs`: number of training epochs
* `--batch_size`: training batch size
* `--lr`: learning rate
* `--save_dir`: directory to save model weights and checkpoints

### Run GA optimization

Run the GA hyperparameter search to find better hyperparameters:

```bash
python 04_CNN_GA.py --data_dir processed_dataset --population 20 --generations 10 --use_ga
```

Common flags (examples):

* `--population`: GA population size
* `--generations`: number of GA generations
* `--use_ga`: toggles GA mode (some scripts may accept the flag to switch behavior)

### Evaluation & Visualizations

After training, run the evaluation script (if provided) or use the helper functions in `03_train_model.py` to:

* Produce training/validation loss and accuracy plots
* Save confusion matrices & ROC curves per-class (flood / non-flood)
* Export per-pixel probability maps for qualitative inspection

Typical workflow:

```bash
# run an evaluation routine (example)
python 03_train_model.py --evaluate --checkpoint models/best_model.pth --data_dir processed_dataset
```

## Scripts (what they do)

* `01_match_label_sar_pairs.py`: traverses the `sen12flood/` dataset, matches SAR source images with label GeoJSON/GeoTIFFs and writes `valid_label_sar_pairs.txt`.
* `02_preprocess_sar_label_pairs.py`: reads `valid_label_sar_pairs.txt`, loads SAR VV/VH bands, normalizes, crops/resamples as needed, and saves NumPy arrays into `processed_dataset/` (images and masks).
* `03_train_model.py`: defines the CNN model, data loaders, training loop, validation, checkpointing, and basic evaluation plotting.
* `04_CNN_GA.py`: defines a GA loop that encodes CNN hyperparameters as chromosomes, evaluates fitness (validation metric), and evolves candidate hyperparameters across generations.

## Model & Genetic Algorithm (brief)

**Model**: A lightweight Convolutional Neural Network that accepts a 2-channel input (VV and VH), uses encoder-decoder or UNet-like blocks (configurable in `03_train_model.py`), and outputs per-pixel sigmoid probabilities for binary flood detection.

**GA**: Typical hyperparameters encoded for GA include learning rate, dropout rate, weight decay, optimizer type, batch size, scheduler parameters, and some architecture choices (e.g., number of filters). Fitness is computed as a validation metric (e.g., F1 score or IoU). `deap` is recommended for GA operators (selection, crossover, mutation).

## License

This project is released under the **MIT License** — see `LICENSE` for details.

