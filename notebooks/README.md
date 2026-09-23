# PyTorch Lightning Notebooks - Learning Path Guide

Twenty notebooks that take you from Lightning fundamentals to a capstone ablation study. All notebooks target **Lightning 2.x** (`import lightning.pytorch as pl`) and are written to run end-to-end on a CPU-only laptop; GPU-specific features are guarded or demonstrated with scripts.

## Notebook Index

### 01 - Lightning Fundamentals (`01_lightning_fundamentals/`)

| Notebook | Topics |
| --- | --- |
| [01_pl_architecture.ipynb](./01_lightning_fundamentals/01_pl_architecture.ipynb) | `LightningModule`, `Trainer`, the training/validation step contract, Lightning vs plain PyTorch |
| [02_trainer_sanity_and_debug.ipynb](./01_lightning_fundamentals/02_trainer_sanity_and_debug.ipynb) | `fast_dev_run`, `overfit_batches`, `limit_*_batches`, gradient-norm logging via a callback, `detect_anomaly`, profilers |
| [03_lightningcli_config_runs.ipynb](./01_lightning_fundamentals/03_lightningcli_config_runs.ipynb) | `LightningCLI`, YAML configs, config-driven experiments |

### 02 - DataModules and Metrics (`02_datamodules_and_metrics/`)

| Notebook | Topics |
| --- | --- |
| [04_building_datamodules.ipynb](./02_datamodules_and_metrics/04_building_datamodules.ipynb) | `LightningDataModule`, `prepare_data` vs `setup`, splits, transforms, DataLoader tuning |
| [05_torchmetrics_logging.ipynb](./02_datamodules_and_metrics/05_torchmetrics_logging.ipynb) | TorchMetrics, `MetricCollection`, `self.log` semantics, loggers |

### 03 - Callbacks and Checkpointing (`03_callbacks_and_checkpointing/`)

| Notebook | Topics |
| --- | --- |
| [06_checkpoint_earlystop.ipynb](./03_callbacks_and_checkpointing/06_checkpoint_earlystop.ipynb) | `ModelCheckpoint`, `EarlyStopping`, resuming, checkpoint contents |
| [07_custom_callbacks_swa_ema.ipynb](./03_callbacks_and_checkpointing/07_custom_callbacks_swa_ema.ipynb) | Writing callbacks, hook order, Stochastic Weight Averaging, EMA weights |

### 04 - Performance and Scaling (`04_performance_and_scaling/`)

| Notebook | Topics |
| --- | --- |
| [08_mixed_precision_amp.ipynb](./04_performance_and_scaling/08_mixed_precision_amp.ipynb) | `precision="16-mixed"` / `"bf16-mixed"`, memory and speed benchmarking callbacks (CPU falls back to bf16) |
| [09_grad_accum_clip_compile.ipynb](./04_performance_and_scaling/09_grad_accum_clip_compile.ipynb) | Gradient accumulation, gradient clipping, `torch.compile` |
| [10_profiler_and_perf_tuning.ipynb](./04_performance_and_scaling/10_profiler_and_perf_tuning.ipynb) | Simple / PyTorch profilers, finding bottlenecks, DataLoader tuning |

### 05 - Strategies and DDP (`05_strategies_and_ddp/`)

| Notebook | Topics |
| --- | --- |
| [11_devices_precision_strategies.ipynb](./05_strategies_and_ddp/11_devices_precision_strategies.ipynb) | Accelerators, devices, precision and strategy flags, production configurations |
| [12_ddp_single_node_walkthrough.ipynb](./05_strategies_and_ddp/12_ddp_single_node_walkthrough.ipynb) | DDP concepts, `DDPStrategy` (NCCL vs Gloo), a generated `train_ddp.py` launched with `python` / `torchrun`, `ddp_notebook` vs `ddp_spawn` |

### 06 - Advanced Mechanics (`06_advanced_mechanics/`)

Lightning 2.0 **removed the public Loop API** (`pytorch_lightning.loops.base.Loop`, `FitLoop` / `TrainingEpochLoop` / `EvaluationLoop` subclassing, `*_epoch_end(outputs)` hooks). These notebooks use the 2.x-idiomatic replacements: composing Trainers in plain Python, `on_*_epoch_end` hooks with manually accumulated outputs, and Callbacks.

| Notebook | Topics |
| --- | --- |
| [13_manual_optimization_gan.ipynb](./06_advanced_mechanics/13_manual_optimization_gan.ipynb) | `automatic_optimization=False`, multiple optimizers, `manual_backward`, GAN training, `on_train_epoch_end` |
| [14_custom_loops_kfold.ipynb](./06_advanced_mechanics/14_custom_loops_kfold.ipynb) | K-Fold cross validation with a fresh `Trainer` per fold, `on_validation_epoch_end`, model selection |
| [15_curriculum_batchloop.ipynb](./06_advanced_mechanics/15_curriculum_batchloop.ipynb) | Curriculum learning driven by a `Callback` + `reload_dataloaders_every_n_epochs=1`, pacing functions |

### 07 - Evaluation, Export and Prediction (`07_evaluation_export_predict/`)

| Notebook | Topics |
| --- | --- |
| [16_test_predict_loops.ipynb](./07_evaluation_export_predict/16_test_predict_loops.ipynb) | `trainer.test` / `trainer.predict`, `test_step` / `predict_step`, result-collecting callbacks, `BasePredictionWriter`, MC-dropout uncertainty |
| [17_onnx_torchscript_export.ipynb](./07_evaluation_export_predict/17_onnx_torchscript_export.ipynb) | ONNX and TorchScript export, validation of exported models, inference benchmarking |

### 08 - Projects and Capstone (`08_projects_and_capstone/`)

| Notebook | Topics |
| --- | --- |
| [18_mini_vision_project.ipynb](./08_projects_and_capstone/18_mini_vision_project.ipynb) | Multi-task vision model (classifier + segmenter), SWA vs non-SWA comparison |
| [19_mini_nlp_project.ipynb](./08_projects_and_capstone/19_mini_nlp_project.ipynb) | Character-level language model vs sentiment classifier |
| [20_capstone_ablation_study.ipynb](./08_projects_and_capstone/20_capstone_ablation_study.ipynb) | Configurable model + DataModule, systematic ablation framework (`QUICK=True` runs 4 experiments x 2 epochs), production pipeline |

## Recommended Learning Path

1. **Fundamentals (01-05)** - module/trainer contract, debugging flags, configs, data and metrics
2. **Training control (06-10)** - callbacks, checkpointing, precision, accumulation, profiling
3. **Scaling (11-12)** - devices, strategies and a real DDP script
4. **Advanced mechanics (13-15)** - manual optimization and 2.x replacements for custom loops
5. **Evaluation and deployment (16-17)** - testing, prediction, export
6. **Projects (18-20)** - end-to-end applications and the capstone ablation study

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision lightning torchmetrics
pip install scikit-learn matplotlib seaborn pandas pyyaml   # analysis and plotting
pip install onnx onnxruntime                                # notebook 17
pip install tensorboard                                     # optional: TensorBoard logging / image logging
```

Notebooks import Lightning as `import lightning.pytorch as pl`. The legacy `pytorch_lightning` package name still works with Lightning 2.x but is not used here.

## Hardware Notes

- Every notebook runs on CPU; training lengths are deliberately short (`max_epochs`, `limit_train_batches`, `QUICK` flags) with comments on how to scale them up.
- `precision="16-mixed"` needs a GPU; on CPU Lightning falls back to `"bf16-mixed"` with a warning (notebook 08).
- Multi-process DDP (notebook 12) is demonstrated with a generated `train_ddp.py` script. The in-notebook multi-process run is behind a `RUN_DDP` flag (default `False`) because `ddp` / `ddp_spawn` are not supported inside Jupyter; `ddp_notebook` is the interactive alternative.
- Notebooks 13-16 download MNIST into `./data` on first run.

## Troubleshooting

- **`ImportError: cannot import name 'Loop'`** - the public Loop API was removed in Lightning 2.0; see section 06 for the replacements.
- **`TypeError: unexpected keyword 'track_grad_norm'`** - removed in 2.0; log gradient norms from a callback with `lightning.pytorch.utilities.grad_norm` (notebook 02).
- **`precision=16` errors** - use the 2.x strings `"16-mixed"`, `"bf16-mixed"`, `"32-true"`.
- **Image logging does nothing** - `add_image` needs a TensorBoard logger; without `tensorboard` installed Lightning uses `CSVLogger`.
- **DataLoader worker errors inside Jupyter on macOS** - use `num_workers=0` (the notebooks' default) or move training into a script.

## Additional Resources

- [Lightning documentation](https://lightning.ai/docs/pytorch/stable/)
- [Lightning 2.0 upgrade guide](https://lightning.ai/docs/pytorch/stable/upgrade/migration_guide.html)
- [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/)
