# Project structure

Generated from the repository tree. Runtime output directories (`logs/`, `checkpoints/`, `lightning_logs/`,
`predictions/`, `*_results/`) are git-ignored and omitted.

```text
LightningMasterPro/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy-docs.yml
├── configs/   # jsonargparse configs for LightningCLI (one per domain) + tuning configs
│   ├── nlp/
│   │   ├── char_lm.yaml
│   │   └── sentiment.yaml
│   ├── tabular/
│   │   └── mlp.yaml
│   ├── timeseries/
│   │   └── forecaster.yaml
│   ├── tuning/
│   │   ├── ablation_study.yaml
│   │   ├── batch_scaler.yaml
│   │   └── lr_finder.yaml
│   ├── vision/
│   │   ├── classifier.yaml
│   │   └── segmenter.yaml
│   └── defaults.yaml
├── data/
│   └── synthetic/
│       └── README.md
├── docker/   # CPU image and compose services (train, jupyter, dev shell)
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/   # Sphinx docs (conf.py + source/), LEARNING_PATH.md, this file
│   ├── source/
│   │   ├── api/
│   │   │   └── modules.rst
│   │   └── index.rst
│   ├── LEARNING_PATH.md
│   ├── PROJECT_STRUCTURE.md
│   └── conf.py
├── models/
│   └── README.md
├── notebooks/   # 20 notebooks in 8 modules
│   ├── 01_lightning_fundamentals/
│   │   ├── 01_pl_architecture.ipynb
│   │   ├── 02_trainer_sanity_and_debug.ipynb
│   │   └── 03_lightningcli_config_runs.ipynb
│   ├── 02_datamodules_and_metrics/
│   │   ├── 04_building_datamodules.ipynb
│   │   └── 05_torchmetrics_logging.ipynb
│   ├── 03_callbacks_and_checkpointing/
│   │   ├── 06_checkpoint_earlystop.ipynb
│   │   └── 07_custom_callbacks_swa_ema.ipynb
│   ├── 04_performance_and_scaling/
│   │   ├── 08_mixed_precision_amp.ipynb
│   │   ├── 09_grad_accum_clip_compile.ipynb
│   │   └── 10_profiler_and_perf_tuning.ipynb
│   ├── 05_strategies_and_ddp/
│   │   ├── 11_devices_precision_strategies.ipynb
│   │   └── 12_ddp_single_node_walkthrough.ipynb
│   ├── 06_advanced_mechanics/
│   │   ├── 13_manual_optimization_gan.ipynb
│   │   ├── 14_custom_loops_kfold.ipynb
│   │   └── 15_curriculum_batchloop.ipynb
│   ├── 07_evaluation_export_predict/
│   │   ├── 16_test_predict_loops.ipynb
│   │   └── 17_onnx_torchscript_export.ipynb
│   ├── 08_projects_and_capstone/
│   │   ├── 18_mini_vision_project.ipynb
│   │   ├── 19_mini_nlp_project.ipynb
│   │   └── 20_capstone_ablation_study.ipynb
│   └── README.md
├── scripts/   # train / evaluate / predict / export_onnx / tune_lr / scale_batch / benchmark / run_ablation / generate_data
│   ├── benchmark.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── generate_data.py
│   ├── predict.py
│   ├── run_ablation.py
│   ├── scale_batch.py
│   ├── train.py
│   └── tune_lr.py
├── src/
│   └── lmpro/   # the library
│       ├── callbacks/   # EMA, SWA, EnhancedModelCheckpoint, gradient + LR monitors
│       │   ├── __init__.py
│       │   ├── checkpoints.py
│       │   ├── ema.py
│       │   ├── gradient_monitor.py
│       │   ├── lr_monitor.py
│       │   └── swa.py
│       ├── data/   # synthetic dataset generators
│       │   ├── __init__.py
│       │   ├── synth_nlp.py
│       │   ├── synth_tabular.py
│       │   ├── synth_timeseries.py
│       │   └── synth_vision.py
│       ├── datamodules/   # LightningDataModules
│       │   ├── __init__.py
│       │   ├── nlp_dm.py
│       │   ├── tabular_dm.py
│       │   ├── ts_dm.py
│       │   └── vision_dm.py
│       ├── loops/   # KFoldLoop driver, CurriculumLoop callback, ProgressiveUnfreezingCallback
│       │   ├── __init__.py
│       │   ├── curriculum_loop.py
│       │   ├── kfold_loop.py
│       │   └── progressive_unfreezing.py
│       ├── modules/   # LightningModules by domain
│       │   ├── nlp/
│       │   │   ├── __init__.py
│       │   │   ├── char_lm.py
│       │   │   └── sentiment.py
│       │   ├── tabular/
│       │   │   ├── __init__.py
│       │   │   └── mlp_reg_cls.py
│       │   ├── timeseries/
│       │   │   ├── __init__.py
│       │   │   └── forecaster.py
│       │   ├── vision/
│       │   │   ├── __init__.py
│       │   │   ├── classifier.py
│       │   │   └── segmenter.py
│       │   └── __init__.py
│       ├── utils/   # metrics, seeding, visualisation, interpretability
│       │   ├── __init__.py
│       │   ├── interpretability.py
│       │   ├── metrics.py
│       │   ├── seed.py
│       │   └── viz.py
│       ├── __init__.py
│       ├── cli.py
│       └── py.typed
├── tests/   # pytest suite (all CPU)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_callbacks_checkpoints.py
│   ├── test_callbacks_ema.py
│   ├── test_callbacks_gradient_monitor.py
│   ├── test_callbacks_lr_monitor.py
│   ├── test_callbacks_swa.py
│   ├── test_cli.py
│   ├── test_configs.py
│   ├── test_data_synth_nlp.py
│   ├── test_data_synth_tabular.py
│   ├── test_data_synth_timeseries.py
│   ├── test_data_synth_vision.py
│   ├── test_datamodules.py
│   ├── test_loops_curriculum.py
│   ├── test_loops_kfold.py
│   ├── test_loops_progressive_unfreezing.py
│   ├── test_modules_shapes.py
│   ├── test_notebooks_syntax.py
│   ├── test_step_cpu_smoke.py
│   ├── test_utils_metrics.py
│   ├── test_utils_seed.py
│   └── test_utils_viz.py
├── .dockerignore
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
└── setup.py
```

## How the pieces fit together

1. `configs/*.yaml` name a `LightningModule` and a `LightningDataModule` by `class_path` and their `init_args`.
2. `scripts/train.py` (or the `lmpro` console script) hands the config to `lmpro.cli.LightningMasterCLI`, a
   `LightningCLI` subclass, which exposes `fit` / `validate` / `test` / `predict`.
3. The other scripts call `LightningMasterCLI.from_config(...)` to rebuild the exact trainer, model and datamodule
   from the same YAML, then run the LR finder, batch-size scaler, evaluation, prediction, ONNX export or a benchmark.
4. `tests/` exercises every module, datamodule, callback, loop and config with real (tiny) `Trainer` runs.
5. `notebooks/` are self-contained teaching material; they use the same Lightning 2.x idioms as the library.
