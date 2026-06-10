# HPS-AlSi-property-prediction

A deep learning-based framework for predicting mechanical properties of high pressure solidified Al-Si alloys from microstructure images using Convolutional Neural Networks.

## Overview

This project uses a CNN-based model to predict mechanical properties (Ultimate Tensile Strength, Yield Strength, and Hardness) of Al-Si alloys from VTK format microstructure images. The model takes phase and composition field data as inputs and outputs multiple material properties simultaneously.

## Features

- **Multi-property prediction**: Predicts multiple mechanical properties (UTS, YS, HV) simultaneously
- **Dual-channel input**: Processes both phase and composition information from VTK files
- **CNN-based architecture**: Efficient feature extraction with convolutional layers
- **Model persistence**: Save and load model components (extractor and predictor) separately
- **Comprehensive evaluation**: Provides R², MAE, and RMSE metrics with visualization

## Project Structure

```
HPS-AlSi-property-prediction/
├── dataloader.py           # Data loading and preprocessing
├── property-predictor.py   # Model definition and training pipeline
├── readvtk.py             # VTK file reading utility
├── train-data/            # Sample data directory with VTK files and CSV labels
├── model-results/         # Output directory for trained models and results
└── README.md              # This file
```

## Installation

### Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- torchvision

### Setup

```bash
# Clone the repository
git clone https://github.com/Gaotc1011/HPS-AlSi-property-prediction.git
cd HPS-AlSi-property-prediction

# Install dependencies
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## Data Format

### VTK Files
- **Phase files**: Named with "phas" (e.g., `sample_phas_001.vtk`)
- **Composition files**: Named with "comp" (e.g., `sample_comp_001.vtk`)
- Expected image size: 600×600 pixels
- Must be paired (same index/ID in filenames)

### CSV File
Properties CSV file should contain:
- Column `id`: Sample identifier (extracted from VTK filenames)
- Column `HV`: Hardness values
- Column `UTS`: Ultimate Tensile Strength (optional)
- Column `YS`: Yield Strength (optional)

Example:
```
id,HV,UTS,YS
1,150.5,450.2,350.1
2,155.3,460.5,360.2
...
```

## Usage

### Training

```bash
# Start training with default parameters
python property-predictor.py -s

# Start training with custom learning rate
python property-predictor.py -s --lr 0.001

# Resume training from checkpoint
python property-predictor.py -s -r --lr 0.001
```

### Validation

```bash
# Run inference on validation set
python property-predictor.py -v
```

### Arguments

- `-s, --start`: Start training
- `-v, --validate`: Run validation/inference
- `-r, --restart`: Resume training from saved checkpoint
- `--lr`: Learning rate (default: 0.01)
- `-d, --debug`: Enable debug logging

## Model Architecture

### ModelCNN

```
Input: (2, 600, 600)  # 2 channels: phase and composition

Extractor (Feature Extraction):
- Conv2d(2→8, k=4, s=2) + BatchNorm + LeakyReLU  → (8, 300, 300)
- Conv2d(8→16, k=4, s=2) + BatchNorm + LeakyReLU  → (16, 150, 150)
- Conv2d(16→16, k=4, s=2) + BatchNorm + LeakyReLU  → (16, 75, 75)
- Conv2d(16→8, k=4, s=2) + BatchNorm + LeakyReLU  → (8, 37, 37)
- Conv2d(8→2, k=4, s=2) + BatchNorm + LeakyReLU   → (2, 18, 18)
- Flatten + Linear(648→256)

Predictor (Property Prediction):
- Linear(256→128) + ReLU
- Linear(128→64) + ReLU
- Linear(64→output_dim)  # output_dim = number of properties

Output: (batch_size, 3)  # 3 properties: UTS, YS, HV
```

## Output Files

After training/inference, the following files are generated in `model-results/`:

- `extractor.pth`: Trained feature extractor weights
- `predictor.pth`: Trained property predictor weights
- `optimizer.pth`: Optimizer state for resuming training
- `loss_history-hardness.csv`: Training loss per epoch
- `loss_history.jpg`: Loss curve visualization
- `multi_property_predictions.csv`: Predictions and true values
- `scatter_UTS.jpg`, `scatter_YS.jpg`, `scatter_HV.jpg`: Property prediction scatter plots with R², MAE, RMSE metrics

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `./train-data` | Path to data directory |
| `model_dir` | `./model-results/` | Path to save models |
| `batch_size` | 2 | Training batch size |
| `lr` | 0.01 | Learning rate |
| `maxiter` | 100 | Number of training epochs |
| `seed` | 70 | Random seed for reproducibility |

## Performance Metrics

The model evaluates predictions using:
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

## Citation

If you use this code, please cite the corresponding paper.

## Contact

For questions or issues, please contact the repository maintainer.

---

**Last Updated**: 2026-06-10
