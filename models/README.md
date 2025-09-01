# models/README.md

# Model Checkpoints Directory

This directory stores saved model checkpoints during training and demonstrations.

## Directory Structure

- `vision/` - Vision model checkpoints
- `nlp/` - NLP model checkpoints
- `tabular/` - Tabular model checkpoints
- `timeseries/` - Time series model checkpoints

## Checkpoint Naming Convention

```
{domain}_{model}_{dataset}_{timestamp}.ckpt
```

Examples:

- `vision_classifier_synthetic_20240101_120000.ckpt`
- `nlp_sentiment_synthetic_20240101_120000.ckpt`

## Usage

Checkpoints are automatically saved during training with Lightning's ModelCheckpoint callback.
Best models are saved based on validation metrics defined in each module.
