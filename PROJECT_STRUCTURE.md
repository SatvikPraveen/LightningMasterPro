.
├── .DS_Store
├── .github
│   └── workflows
│       ├── ci.yml
│       └── deploy-docs.yml
├── .gitignore
├── configs
│   ├── defaults.yaml
│   ├── nlp
│   │   ├── char_lm.yaml
│   │   └── sentiment.yaml
│   ├── tabular
│   │   └── mlp.yaml
│   ├── timeseries
│   │   └── forecaster.yaml
│   ├── tuning
│   │   ├── ablation_study.yaml
│   │   ├── batch_scaler.yaml
│   │   └── lr_finder.yaml
│   └── vision
│       ├── classifier.yaml
│       └── segmenter.yaml
├── data
│   └── synthetic
│       └── README.md
├── docker
│   ├── docker-compose.yml
│   └── Dockerfile
├── docs
│   ├── conf.py
│   ├── Makefile
│   └── source
│       └── index.rst
├── generate_lightning_master_pro.sh
├── LEARNING_PATH.md
├── logs
│   └── README.md
├── models
│   └── README.md
├── notebooks
│   ├── 01_lightning_fundamentals
│   │   ├── 01_pl_architecture.ipynb
│   │   ├── 02_trainer_sanity_and_debug.ipynb
│   │   └── 03_lightningcli_config_runs.ipynb
│   ├── 02_datamodules_and_metrics
│   │   ├── 04_building_datamodules.ipynb
│   │   └── 05_torchmetrics_logging.ipynb
│   ├── 03_callbacks_and_checkpointing
│   │   ├── 06_checkpoint_earlystop.ipynb
│   │   └── 07_custom_callbacks_swa_ema.ipynb
│   ├── 04_performance_and_scaling
│   │   ├── 08_mixed_precision_amp.ipynb
│   │   ├── 09_grad_accum_clip_compile.ipynb
│   │   └── 10_profiler_and_perf_tuning.ipynb
│   ├── 05_strategies_and_ddp
│   │   ├── 11_devices_precision_strategies.ipynb
│   │   └── 12_ddp_single_node_walkthrough.ipynb
│   ├── 06_advanced_mechanics
│   │   ├── 13_manual_optimization_gan.ipynb
│   │   ├── 14_custom_loops_kfold.ipynb
│   │   └── 15_curriculum_batchloop.ipynb
│   ├── 07_evaluation_export_predict
│   │   ├── 16_test_predict_loops.ipynb
│   │   └── 17_onnx_torchscript_export.ipynb
│   ├── 08_projects_and_capstone
│   │   ├── 18_mini_vision_project.ipynb
│   │   ├── 19_mini_nlp_project.ipynb
│   │   └── 20_capstone_ablation_study.ipynb
│   └── README.md
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt
├── scripts
│   ├── export_onnx.py
│   ├── predict.py
│   ├── run_ablation.py
│   ├── scale_batch.py
│   ├── train.py
│   └── tune_lr.py
├── setup.py
├── src
│   └── lmpro
│       ├── __init__.py
│       ├── callbacks
│       │   ├── __init__.py
│       │   ├── checkpoints.py
│       │   ├── ema.py
│       │   └── swa.py
│       ├── cli.py
│       ├── data
│       │   ├── __init__.py
│       │   ├── synth_nlp.py
│       │   ├── synth_tabular.py
│       │   ├── synth_timeseries.py
│       │   └── synth_vision.py
│       ├── datamodules
│       │   ├── __init__.py
│       │   ├── nlp_dm.py
│       │   ├── tabular_dm.py
│       │   ├── ts_dm.py
│       │   └── vision_dm.py
│       ├── loops
│       │   ├── __init__.py
│       │   ├── curriculum_loop.py
│       │   └── kfold_loop.py
│       ├── modules
│       │   ├── __init__.py
│       │   ├── nlp
│       │   │   ├── __init__.py
│       │   │   ├── char_lm.py
│       │   │   └── sentiment.py
│       │   ├── tabular
│       │   │   ├── __init__.py
│       │   │   └── mlp_reg_cls.py
│       │   ├── timeseries
│       │   │   ├── __init__.py
│       │   │   └── forecaster.py
│       │   └── vision
│       │       ├── __init__.py
│       │       ├── classifier.py
│       │       └── segmenter.py
│       └── utils
│           ├── __init__.py
│           ├── metrics.py
│           ├── seed.py
│           └── viz.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── test_configs.py
    ├── test_datamodules.py
    ├── test_modules_shapes.py
    └── test_step_cpu_smoke.py

39 directories, 95 files
