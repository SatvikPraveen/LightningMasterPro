# LightningMasterPro

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?style=for-the-badge&logo=pytorch-lightning&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Hydra](https://img.shields.io/badge/Config-Hydra-89b4f5?style=for-the-badge)
![ONNX](https://img.shields.io/badge/Export-ONNX-005CED?style=for-the-badge&logo=onnx)

A comprehensive PyTorch Lightning framework showcasing advanced machine learning engineering patterns, distributed training strategies, and production-ready model development workflows.

## Overview

This project demonstrates mastery of PyTorch Lightning through a complete ML engineering ecosystem featuring 20+ educational notebooks, modular architecture, and comprehensive testing. The framework spans multiple domains (vision, NLP, tabular, time series) while emphasizing scalability, reproducibility, and best practices.

## Key Features

- **Advanced Training Patterns**: Manual optimization, curriculum learning, k-fold validation, and custom training loops
- **Production Architecture**: Modular design with clean separation of concerns and comprehensive configuration management
- **Multi-Domain Support**: Computer vision, NLP, tabular data, and time series forecasting implementations
- **Distributed Training**: DDP strategies, multi-GPU optimization, and scaling patterns
- **Performance Engineering**: Mixed precision training, gradient accumulation, profiling, and optimization techniques
- **Export & Deployment**: ONNX and TorchScript export pipelines with cross-platform compatibility
- **Comprehensive Testing**: Unit tests, smoke tests, and integration testing across all components

## Quick Start

```bash
git clone https://github.com/SatvikPraveen/LightningMasterPro.git
cd LightningMasterPro
pip install -e .
```

### Basic Training

```bash
# Vision classifier with synthetic data
python scripts/train.py --config configs/vision/classifier.yaml

# NLP sentiment analysis
python scripts/train.py --config configs/nlp/sentiment.yaml

# Time series forecasting
python scripts/train.py --config configs/timeseries/forecaster.yaml
```

### Advanced Workflows

```bash
# Learning rate finder
python scripts/tune_lr.py --config configs/tuning/lr_finder.yaml

# Batch size scaling
python scripts/scale_batch.py --config configs/tuning/batch_scaler.yaml

# Ablation study
python scripts/run_ablation.py --config configs/tuning/ablation_study.yaml

# Model export to ONNX
python scripts/export_onnx.py --checkpoint models/best_model.ckpt
```

## Architecture

### Core Structure

```
src/lmpro/
├── modules/           # Lightning modules by domain
│   ├── vision/        # Image classification, segmentation
│   ├── nlp/           # Language modeling, sentiment analysis
│   ├── tabular/       # Regression and classification MLPs
│   └── timeseries/    # Forecasting models
├── datamodules/       # Data loading and preprocessing
├── callbacks/         # Custom callbacks (SWA, EMA, checkpointing)
├── loops/             # Custom training loops (k-fold, curriculum)
├── data/              # Synthetic data generators
└── utils/             # Utilities and metrics
```

### Configuration System

All experiments are driven by YAML configurations using Hydra and LightningCLI:

```yaml
# configs/vision/classifier.yaml
model:
  _target_: lmpro.modules.vision.VisionClassifier
  num_classes: 10
  backbone: resnet18

data:
  _target_: lmpro.datamodules.VisionDataModule
  batch_size: 64
  num_workers: 4

trainer:
  max_epochs: 50
  precision: 16-mixed
  devices: auto
```

## Educational Content

### Structured Learning Path (20 Notebooks)

1. **Lightning Fundamentals** (3 notebooks)

   - PyTorch Lightning architecture and core concepts
   - Trainer configuration and debugging workflows
   - LightningCLI and configuration-driven experiments

2. **Data & Metrics** (2 notebooks)

   - Building robust DataModules for different data types
   - TorchMetrics integration and custom metric development

3. **Callbacks & Checkpointing** (2 notebooks)

   - Model checkpointing and early stopping strategies
   - Advanced callbacks: SWA, EMA, custom monitoring

4. **Performance & Scaling** (3 notebooks)

   - Mixed precision training and automatic optimization
   - Gradient accumulation, clipping, and model compilation
   - Performance profiling and bottleneck analysis

5. **Multi-GPU Strategies** (2 notebooks)

   - Device management and precision strategies
   - Distributed Data Parallel (DDP) implementation

6. **Advanced Mechanics** (3 notebooks)

   - Manual optimization for complex training scenarios
   - K-fold cross-validation with proper data handling
   - Curriculum learning and custom batch processing

7. **Evaluation & Export** (2 notebooks)

   - Comprehensive testing and prediction workflows
   - Model export to ONNX and TorchScript for deployment

8. **Complete Projects** (3 notebooks)
   - End-to-end vision project with ablation studies
   - Comprehensive NLP project comparing generative vs discriminative models
   - Capstone project demonstrating full ML pipeline

## Domain Implementations

### Computer Vision

- **Models**: ResNet, EfficientNet classifiers; U-Net segmentation
- **Features**: Data augmentation, transfer learning, multi-scale training
- **Synthetic Data**: Configurable image generation with realistic variations

### Natural Language Processing

- **Models**: Character-level LSTM language models, bidirectional sentiment analysis
- **Features**: Custom tokenization, sequence-to-sequence architectures
- **Synthetic Data**: Text generation with controllable complexity and domain

### Tabular Data

- **Models**: Multi-layer perceptrons for regression and classification
- **Features**: Feature engineering, categorical encoding, normalization
- **Synthetic Data**: Realistic tabular datasets with configurable correlations

### Time Series

- **Models**: LSTM and Transformer-based forecasting models
- **Features**: Multi-step prediction, seasonality handling, uncertainty quantification
- **Synthetic Data**: Time series with trends, seasonality, and noise patterns

## Advanced Features

### Custom Training Loops

- **K-Fold Validation**: Automated cross-validation with proper data splitting
- **Curriculum Learning**: Progressive difficulty scheduling with custom batch samplers
- **Manual Optimization**: Multi-optimizer scenarios for adversarial training

### Performance Optimization

- **Mixed Precision**: Automatic and manual AMP implementation
- **Gradient Accumulation**: Large effective batch sizes on resource-constrained hardware
- **Model Compilation**: Integration with PyTorch 2.0 compilation features
- **Profiling**: Built-in performance analysis and optimization recommendations

### Production Features

- **Checkpointing**: Advanced save/load strategies with custom state management
- **Callbacks**: Professional-grade monitoring and intervention callbacks
- **Export Pipeline**: Complete model serialization for deployment environments
- **Testing Framework**: Comprehensive test coverage including smoke tests and integration tests

## Synthetic Data Generation

All examples utilize sophisticated synthetic data generators, eliminating external dataset dependencies:

```python
# Vision data with realistic augmentations
vision_data = SyntheticImageDataset(
    num_samples=10000,
    image_size=(224, 224),
    num_classes=10,
    complexity="medium"
)

# NLP data with controlled vocabulary
nlp_data = SyntheticTextDataset(
    num_samples=5000,
    vocab_size=10000,
    sequence_lengths=(10, 100),
    task_type="classification"
)

# Time series with configurable patterns
ts_data = SyntheticTimeSeriesDataset(
    num_series=1000,
    length=365,
    seasonality=True,
    trend=True,
    noise_level=0.1
)
```

## Testing & Quality Assurance

### Comprehensive Test Suite

```bash
# Full test suite
pytest

# Smoke tests for rapid validation
pytest tests/test_step_cpu_smoke.py

# Component-specific testing
pytest tests/test_datamodules.py -v
pytest tests/test_modules_shapes.py -v

# Configuration validation
pytest tests/test_configs.py
```

### Continuous Integration

- Automated testing across multiple Python versions
- GPU and CPU compatibility validation
- Configuration file validation
- Documentation deployment pipeline

## Performance Benchmarks

The framework demonstrates measurable improvements through advanced techniques:

- **Mixed Precision Training**: 1.5-2x speedup with maintained accuracy
- **Stochastic Weight Averaging**: 2-3% accuracy improvements across domains
- **Gradient Accumulation**: Effective large batch training on single GPU
- **DDP Scaling**: Near-linear scaling efficiency across multiple GPUs

## Docker Support

Complete containerized development environment:

```bash
# Development environment
docker-compose up --build

# Interactive shell
docker-compose run --rm lmpro bash

# Jupyter notebook server
docker-compose run --service-ports lmpro jupyter notebook --ip=0.0.0.0
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- TorchMetrics
- Hydra-core
- Additional dependencies in `requirements.txt`

## Project Structure

The codebase demonstrates professional ML engineering practices with clear separation of concerns, comprehensive documentation, and maintainable architecture patterns. Each component is designed for extensibility while maintaining simplicity and performance.

## Contributing

The project follows standard open-source contribution patterns:

- Feature branches with comprehensive testing
- Code review processes
- Documentation updates for new features
- Backward compatibility considerations

## License

MIT License - see [LICENSE](LICENSE) for complete details.

---

**LightningMasterPro** represents a complete PyTorch Lightning mastery framework, suitable for education, research, and production model development workflows.
