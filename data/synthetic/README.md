# data/synthetic/README.md

# Synthetic Data Directory

This directory contains synthetic datasets generated for training and experimentation.

## Directory Structure

- `vision/` - Generated vision datasets (images, masks)
- `nlp/` - Generated text datasets (sequences, labels)
- `tabular/` - Generated tabular datasets (features, targets)
- `timeseries/` - Generated time series datasets (sequences)

## Data Generation

Synthetic data is generated using the modules in `src/lmpro/data/`:

- `synth_vision.py` - Image classification and segmentation data
- `synth_nlp.py` - Text sequences and sentiment data
- `synth_tabular.py` - Tabular regression and classification data
- `synth_timeseries.py` - Time series forecasting data

## Usage

Data is automatically generated when running notebooks or training scripts.
Files are gitignored to avoid committing large datasets to version control.
