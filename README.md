# Deep Learning with PyTorch

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/AfreenInnovates/deep-learning)

A comprehensive learning journey through PyTorch fundamentals, regression/classification, and computer vision applications. This repository contains structured notebooks, exercises, and detailed documentation covering essential deep learning concepts with hands-on implementations.

## Table of Contents

- [Repo Overview](#repo-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Key Features](#key-features)

## Repo Overview

This repository serves as a complete educational resource for learning PyTorch and deep learning concepts through practical implementations. It covers everything from basic tensor operations to advanced computer vision applications using transfer learning.

**Learning Objectives:**
- Master PyTorch fundamentals and tensor operations
- Understand neural network architectures and training loops
- Implement regression and classification models
- Apply computer vision techniques with CNNs
- Learn transfer learning for image classification
- Practice with real-world datasets and exercises

## Repository Structure

**Deep Learning with PyTorch/**
- **01_basics/** [![GitHub](https://img.shields.io/badge/-black?style=flat-round&logo=github&logoWidth=8&labelColor=black&color=black)](https://github.com/AfreenInnovates/deep-learning/tree/main/01_basics)
  - `01_tensors-pytorch.ipynb`
- **02_regression-classification/** [![GitHub](https://img.shields.io/badge/-black?style=flat-round&logo=github&logoWidth=8&labelColor=black&color=black)](https://github.com/AfreenInnovates/deep-learning/tree/main/02_regression-classification)
  - **learning/**
    - `02_autograd-fundamentals-and-linear-reg.ipynb`
    - `02_neural-network-classification.ipynb`
    - `02-multiclass-classification.ipynb`
    - `PyTorchTraining.md`
  - **practice/**
    - `02_01_linear_reg_exercise.ipynb`
    - `02_02-exercise-student-placement-classification.ipynb`
- **03_computer-vision/** [![GitHub](https://img.shields.io/badge/-black?style=flat-round&logo=github&logoWidth=8&labelColor=black&color=black)](https://github.com/AfreenInnovates/deep-learning/tree/main/03_computer-vision)
  - **learning/**
    - `03_computer_vision_cnns.ipynb`
    - **03_food-vision-transfer-learning/**
      - `03_food_vision_transfer_learning.ipynb`
      - `03_food-vision-transfer-learning.md`
      - `pizza_steak_sushi.zip`
  - **practice/**
    - `03_01_brain_tumor_classification.ipynb`
- `reusable_codes.py` [![GitHub](https://img.shields.io/badge/-black?style=flat-round&logo=github&logoWidth=8&labelColor=black&color=black)](https://github.com/AfreenInnovates/deep-learning/blob/main/reusable_codes.py)
- `clean_nb.py`
- `README.md`

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch 1.9+
- Jupyter Notebook or JupyterLab
- Basic understanding of machine learning concepts

### Installation
```bash
# Clone the repository
git clone https://github.com/AfreenInnovates/deep-learning
cd deep-learning

```

### Running the Notebooks
1. Start Jupyter Notebook: `jupyter notebook`
2. Navigate to the desired section
3. Open notebooks in order for structured learning
4. Run cells sequentially


## Key Features

### Training Utilities (`reusable_codes.py`)
A comprehensive collection of reusable functions for PyTorch training:

**Core Functions:**
- `train()`: Complete training loop with history tracking
- `plot_history()`: Visualize training/validation curves
- `visualize_predictions()`: Grid visualization of model predictions
- `vis_preds_side_by_side()`: Side-by-side actual vs predicted comparison
- `predict_image()`: Single image prediction from URL or local path

**Helper Functions:**
- `get_device()`: Automatic device detection (CUDA/CPU)
- `denormalize_batch()`: Image denormalization for visualization
- Multiprocessing safety for Jupyter/Colab environments

**Usage Example:**
```python
from reusable_codes import train, plot_history, visualize_predictions

# to train model
history = train(model, train_loader, test_loader, optimizer, loss_fn, epochs=5)

# visualize results
plot_history(history)
visualize_predictions(model, test_loader, class_names)
```

### Utility Scripts

#### clean_nb.py
A utility script to clean Jupyter notebooks by removing widget metadata:
```python
# removes widget metadata from all .ipynb files
# helps reduce file size and clean up notebooks
```


**Repository**: [https://github.com/AfreenInnovates/deep-learning](https://github.com/AfreenInnovates/deep-learning)