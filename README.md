# LightningMasterPro

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?style=for-the-badge&logo=pytorch-lightning&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

A comprehensive **PyTorch Lightning syntax refresher** featuring 20 educational notebooks covering all core concepts, from fundamentals to advanced patterns.

## Overview

LightningMasterPro is a one-stop learning resource for PyTorch Lightning. It provides hands-on implementations of every major Lightning concept through a structured notebook series, synthetic data examples, and practical code patterns. The project is designed as a refresher guide for developers who want to master Lightning syntax and best practices without unnecessary complexity.

## Key Features

- **20 Comprehensive Notebooks**: Structured learning path from fundamentals to advanced patterns
- **Core Lightning Concepts**: LightningModule, LightningDataModule, Trainer, callbacks, and configuration
- **Advanced Patterns**: Manual optimization, custom training loops, curriculum learning, k-fold validation
- **Multi-Domain Examples**: Computer vision, NLP, and tabular data implementations
- **Distributed Training**: DDP strategies, multi-GPU optimization, and device management
- **Performance Techniques**: Mixed precision, gradient accumulation, profiling, and compilation
- **Production-Ready Code**: Modular architecture, proper logging, checkpointing, and validation patterns

## Quick Start

```bash
git clone https://github.com/SatvikPraveen/LightningMasterPro.git
cd LightningMasterPro
pip install -e .
```

### Running the Notebooks

1. **Start with fundamentals** - Open `notebooks/01_lightning_fundamentals/` to learn Lightning core concepts
2. **Progress through domains** - Follow the numbered notebooks in sequence for structured learning
3. **Explore implementations** - Each notebook includes working code examples with synthetic data
4. **Reference guide** - Use notebooks as a quick syntax reference for Lightning patterns

### Training Examples

```bash
# Vision classifier
python scripts/train.py --config configs/vision/classifier.yaml

# NLP sentiment analysis
python scripts/train.py --config configs/nlp/sentiment.yaml

# Learning rate finder
python scripts/tune_lr.py --config configs/tuning/lr_finder.yaml
```

## Project Structure

### Core Components

```
src/lmpro/
├── modules/           # Lightning modules by domain
│   ├── vision/        # Image classification
│   ├── nlp/           # NLP tasks (sentiment, language modeling)
│   └── tabular/       # Regression and classification
├── datamodules/       # LightningDataModule implementations
├── callbacks/         # Custom callbacks (EarlyStopping, SWA, EMA)
├── loops/             # Custom training loops (k-fold, curriculum)
└── utils/             # Utilities, metrics, and visualization
```

### Notebooks Organization

```
notebooks/
├── 01_lightning_fundamentals/      # Core Lightning concepts
├── 02_datamodules_and_metrics/     # Data and metric handling
├── 03_callbacks_and_checkpointing/ # Model persistence
├── 04_performance_and_scaling/     # Optimization techniques
├── 05_strategies_and_ddp/          # Multi-GPU and distributed training
├── 06_advanced_mechanics/          # Custom loops and optimization
├── 07_evaluation_export_predict/   # Testing and model export
└── 08_projects_and_capstone/       # End-to-end projects
```

## Learning Path (20 Notebooks)

### **Module 1: Lightning Fundamentals** (Notebooks 1-3)
- PyTorch Lightning architecture and core concepts
- Building and configuring LightningModules
- Using Trainer and LightningCLI for configuration-driven experiments

### **Module 2: Data & Metrics** (Notebooks 4-5)
- Building LightningDataModules for efficient data loading
- Integrating TorchMetrics for proper metric tracking
- Logging and monitoring training progress

### **Module 3: Callbacks & Checkpointing** (Notebooks 6-7)
- Model checkpointing strategies
- Early stopping and performance monitoring
- Custom callbacks: SWA, EMA, and custom interventions

### **Module 4: Performance & Scaling** (Notebooks 8-10)
- Mixed precision training (AMP)
- Gradient accumulation and clipping
- PyTorch 2.0 model compilation
- Profiling and performance optimization

### **Module 5: Distributed Training** (Notebooks 11-12)
- Device management and precision strategies
- Distributed Data Parallel (DDP) single-node
- Multi-GPU scaling and optimization

### **Module 6: Advanced Mechanics** (Notebooks 13-15)
- Manual optimization for complex scenarios
- K-fold cross-validation workflows
- Curriculum learning and progressive training

### **Module 7: Evaluation & Export** (Notebooks 16-17)
- Comprehensive testing and prediction loops
- Model export to TorchScript and ONNX
- Cross-platform deployment considerations

### **Module 8: Projects & Capstone** (Notebooks 18-20)
- End-to-end vision project with ablation studies
- NLP project demonstrating complete workflows
- Capstone combining all Lightning concepts

## Domain Coverage

### Computer Vision
- Image classification with CNNs
- Data augmentation and preprocessing
- Configurable synthetic image generation

### Natural Language Processing
- Sentiment analysis and text classification
- Character-level language modeling
- Custom tokenization and embeddings

### Tabular Data
- Classification and regression MLPs
- Feature engineering patterns
- Data normalization and handling categorical features

## Key Lightning Patterns Covered

### Training Patterns
- Standard supervised learning with LightningModule
- Manual optimization for complex scenarios
- Custom training loops with K-fold and curriculum learning
- Distributed training with DDP

### Data Handling
- LightningDataModule best practices
- Efficient data loading with DataLoaders
- Proper train/val/test split management

### Optimization Techniques
- Mixed precision training (AMP)
- Gradient accumulation and clipping
- Learning rate scheduling
- Model compilation with PyTorch 2.0

### Monitoring & Checkpointing
- Proper logging with Lightning loggers
- Custom callbacks for intervention
- Model checkpointing strategies
- Early stopping and performance monitoring

### Testing & Validation
- Proper validation and test workflows
- Prediction loop implementation
- Model export and inference optimization

## Synthetic Data

All examples use built-in synthetic data generators, eliminating external dataset dependencies:

```python
from lmpro.data import create_synthetic_image_dataset, create_synthetic_text_dataset

# Vision data with augmentations
vision_dm = VisionDataModule(
    data_config=VisionDatasetConfig(num_samples=10000),
    batch_size=64
)

# NLP data with configurable vocabulary
nlp_dm = NLPDataModule(
    data_config=NLPDatasetConfig(vocab_size=10000),
    batch_size=32
)
```

## Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_datamodules.py -v
pytest tests/test_modules_shapes.py -v

# Quick smoke tests
pytest tests/test_step_cpu_smoke.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- TorchMetrics

See `requirements.txt` for complete dependencies.

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -e .`
3. Open `notebooks/01_lightning_fundamentals/01_pl_architecture.ipynb` to begin
4. Follow the numbered notebooks in order for a structured learning experience
5. Reference the source code in `src/lmpro/` for implementation patterns

## Use Cases

**Perfect for:**
- Learning PyTorch Lightning syntax and patterns
- Quick reference guide for common Lightning patterns
- Understanding best practices in ML training workflows
- Building reproducible experiments with configuration-driven approaches

**Not intended for:**
- Production deployment (see official Lightning docs for that)
- State-of-the-art model implementations
- Advanced distributed training at scale

## Resources

- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [Official Examples](https://github.com/Lightning-AI/lightning/tree/master/examples)
- [Lightning Blog](https://www.pytorchlightning.ai/blog)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**LightningMasterPro** - Master PyTorch Lightning through hands-on learning and practical examples.
