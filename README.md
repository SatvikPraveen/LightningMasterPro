# LightningMasterPro

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning_2.x-792EE5?style=for-the-badge&logo=pytorch-lightning&logoColor=white)
![Python](https://img.shields.io/badge/python_3.9+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

A hands-on **PyTorch Lightning 2.x** learning framework: 20 notebooks that walk from `LightningModule` basics to
manual optimization, DDP, profiling and ONNX export, backed by a small, fully tested library (`lmpro`) that shows the
idiomatic way to build modules, datamodules, callbacks, training drivers and a `LightningCLI`.

Everything runs on CPU with synthetic data. No downloads, no GPU required.

## Quick start

```bash
git clone https://github.com/SatvikPraveen/LightningMasterPro.git
cd LightningMasterPro
pip install -e ".[dev,export]"        # add ,notebooks for Jupyter

# Train any domain from a YAML config (LightningCLI subcommands: fit / validate / test / predict)
python scripts/train.py fit --config configs/vision/classifier.yaml
python scripts/train.py fit --config configs/nlp/sentiment.yaml --trainer.max_epochs 3
python scripts/train.py test --config configs/tabular/mlp.yaml --ckpt_path checkpoints/tabular/mlp/last.ckpt

# The same CLI is installed as a console script
lmpro fit --config configs/timeseries/forecaster.yaml --trainer.fast_dev_run true

# Run the test suite (≈2 minutes on a laptop CPU)
pytest
```

## What is in the box

| Area | Contents |
|---|---|
| `notebooks/` | 20 notebooks in 8 modules, Lightning 2.x APIs only (see [notebooks/README.md](notebooks/README.md)) |
| `src/lmpro/modules/` | 6 `LightningModule`s: CNN/ResNet classifier, U-Net segmenter, char-level LM, sentiment classifier (LSTM/GRU/CNN/attention), tabular MLP, LSTM/GRU/Transformer forecaster |
| `src/lmpro/datamodules/` | 4 `LightningDataModule`s with `setup(stage)`, worker seeding, `persistent_workers`, custom collate |
| `src/lmpro/data/` | Synthetic generators for vision, text, tabular and time-series with distinct per-split seeds |
| `src/lmpro/callbacks/` | `EMACallback` (warm-up, resume-safe), `SWACallback` (SWALR + BatchNorm update), `EnhancedModelCheckpoint`, gradient/LR monitors |
| `src/lmpro/loops/` | `KFoldLoop` driver, `CurriculumLoop` callback + `CurriculumDataset`, `ProgressiveUnfreezingCallback` with discriminative LRs |
| `src/lmpro/cli.py` | `LightningMasterCLI`: `LightningCLI` subclass with `link_arguments`, experiment metadata and a `from_config` helper |
| `configs/` | One jsonargparse config per domain plus tuning configs for the LR finder, batch-size scaler and ablations |
| `scripts/` | train / evaluate / predict / export_onnx / tune_lr / scale_batch / benchmark / run_ablation / generate_data |

## Lightning concepts covered

**Core mechanics**: `LightningModule` hooks, `save_hyperparameters`, `configure_optimizers` returning
`{"optimizer", "lr_scheduler": {"scheduler", "interval", "monitor"}}`, `OneCycleLR` sized from
`trainer.estimated_stepping_batches`, torchmetrics objects passed straight to `self.log` (DDP-safe, auto-reset),
`predict_step`, `example_input_array`.

**Data**: `prepare_data` vs `setup(stage)`, `worker_init_fn`, `pin_memory` / `persistent_workers`, `pad_sequence`
collate, curriculum sampling via `reload_dataloaders_every_n_epochs=1`.

**Configuration**: `LightningCLI` subclassing, `add_arguments_to_parser`, `link_arguments(apply_on="instantiate")`,
`class_path` / `init_args`, config layering with multiple `--config`, `--print_config`, `SaveConfigCallback`.

**Training tricks**: mixed precision (`16-mixed` / `bf16-mixed`), gradient accumulation and clipping, `torch.compile`,
manual optimization (`automatic_optimization=False`, `manual_backward`, `toggle_optimizer`) for GANs, EMA, SWA,
progressive unfreezing, `Tuner.lr_find`, `Tuner.scale_batch_size`.

**Scale and observability**: `SimpleProfiler` / `PyTorchProfiler`, `grad_norm` logging in `on_before_optimizer_step`,
DDP with `torchrun`, `ddp_spawn` on CPU, gloo/nccl backends, checkpoint resume with `ckpt_path`.

**Evaluation and export**: `trainer.test` / `trainer.predict`, `BasePredictionWriter`, `to_onnx` with a dynamic batch
axis and an onnxruntime parity check, TorchScript.

## Learning path (20 notebooks)

| Module | Notebooks | Topics |
|---|---|---|
| 01 Fundamentals | 01–03 | Architecture, `fast_dev_run` / `overfit_batches` / `detect_anomaly`, `LightningCLI` |
| 02 Data and metrics | 04–05 | DataModules, TorchMetrics, `log_dict` |
| 03 Callbacks and checkpoints | 06–07 | `ModelCheckpoint`, `EarlyStopping`, resume, custom SWA/EMA callbacks |
| 04 Performance | 08–10 | AMP, gradient accumulation and clipping, `torch.compile`, profilers |
| 05 Strategies and DDP | 11–12 | Accelerators, precision, strategies, single-node DDP walkthrough |
| 06 Advanced mechanics | 13–15 | Manual optimization (GAN), k-fold with a fresh Trainer per fold, curriculum learning |
| 07 Evaluation and export | 16–17 | Test / predict loops, ONNX and TorchScript |
| 08 Projects and capstone | 18–20 | Mini vision and NLP projects, ablation-study capstone |

Lightning 2.0 removed the public `Loop` API. Section 06 shows the replacements: plain Python drivers that create a
fresh `Trainer` per fold, and callbacks that mutate the dataset between epochs.

## Using the library directly

```python
import lightning.pytorch as pl
from lmpro.callbacks import EMACallback, SWACallback
from lmpro.datamodules import VisionDataModule
from lmpro.data import VisionDatasetConfig
from lmpro.loops import KFoldLoop, CurriculumLoop, CurriculumDataset
from lmpro.modules.vision.classifier import VisionClassifier

dm = VisionDataModule(data_config=VisionDatasetConfig(num_samples=2000, image_size=(32, 32)), batch_size=64)
model = VisionClassifier(num_classes=10, architecture="resnet", learning_rate=3e-4)

trainer = pl.Trainer(max_epochs=5, callbacks=[EMACallback(decay=0.999), SWACallback(swa_epoch_start=0.8)])
trainer.fit(model, datamodule=dm)

# 5-fold cross-validation: a fresh model and Trainer per fold, results aggregated as mean/std
results = KFoldLoop(num_folds=5).run(model, dm, trainer_kwargs={"max_epochs": 3})

# Curriculum learning: the callback raises the difficulty threshold each epoch and the
# dataloader is rebuilt because reload_dataloaders_every_n_epochs=1
curriculum = CurriculumLoop(strategy="length")
pl.Trainer(max_epochs=5, reload_dataloaders_every_n_epochs=1, callbacks=[curriculum]).fit(model, datamodule=dm)
```

Programmatic access to the CLI, e.g. for tuning scripts:

```python
from lmpro.cli import LightningMasterCLI

cli = LightningMasterCLI.from_config("configs/vision/classifier.yaml", "--trainer.max_epochs=1")
cli.trainer.fit(cli.model, datamodule=cli.datamodule)
```

## Scripts

```bash
CFG=configs/vision/classifier.yaml
CKPT=checkpoints/vision/classifier/last.ckpt

python scripts/train.py fit --config $CFG                                    # train
python scripts/evaluate.py --config $CFG --checkpoint $CKPT --split test     # metrics -> JSON
python scripts/predict.py  --config $CFG --checkpoint $CKPT                  # predictions.pt
python scripts/export_onnx.py --config $CFG --checkpoint $CKPT --output exports/classifier.onnx
python scripts/tune_lr.py --config $CFG                                      # Tuner.lr_find + plot + updated config
python scripts/scale_batch.py --config $CFG                                  # Tuner.scale_batch_size
python scripts/benchmark.py --config $CFG --batch_sizes 32 64 128 --compile
python scripts/run_ablation.py --config $CFG --max_combinations 4            # grid sweep, CSV + plots
python scripts/generate_data.py --num_samples 1000                           # cache synthetic datasets
```

Every script accepts extra `--key value` overrides that are forwarded to the CLI, e.g.
`--data.init_args.num_workers 0` or `--trainer.accelerator cpu`.

## Development

```bash
pip install -e ".[dev,export]"
pre-commit install
pytest                       # 380+ tests, all CPU
black src tests scripts && isort src tests scripts && flake8 src tests scripts
```

CI runs the lint gates, the test suite on Linux and macOS for Python 3.10–3.12, and a `fast_dev_run` of the training CLI.

## Requirements

Python 3.9+, PyTorch 2.1+, Lightning 2.1+, TorchMetrics 1.2+, jsonargparse. See `requirements.txt` and the extras in
`setup.py` (`dev`, `docs`, `export`, `notebooks`).

## License

MIT. See [LICENSE](LICENSE).
