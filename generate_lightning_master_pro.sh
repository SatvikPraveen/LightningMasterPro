#!/bin/bash

# Script to generate LightningMasterPro project structure
# Usage: ./generate_lightning_master_pro.sh

set -e  # Exit on any error

echo "🚀 Generating LightningMasterPro project structure..."

# Create main project directory
PROJECT_NAME="LightningMasterPro"
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Create .github workflows
mkdir -p .github/workflows
touch .github/workflows/ci.yml
touch .github/workflows/deploy-docs.yml

# Create root files
touch .gitignore
touch README.md
touch requirements.txt
touch setup.py
touch PROJECT_STRUCTURE.md
touch LEARNING_PATH.md

# Create data directories
mkdir -p data/synthetic

# Create models directory
mkdir -p models

# Create logs directory
mkdir -p logs

# Create docs structure
mkdir -p docs/source
touch docs/conf.py

# Create docker files
mkdir -p docker
touch docker/docker-compose.yml
touch docker/Dockerfile

# Create src/lmpro structure with all __init__.py files
mkdir -p src/lmpro
touch src/lmpro/__init__.py
touch src/lmpro/cli.py

# Utils package
mkdir -p src/lmpro/utils
touch src/lmpro/utils/__init__.py
touch src/lmpro/utils/seed.py
touch src/lmpro/utils/metrics.py
touch src/lmpro/utils/viz.py

# Data package
mkdir -p src/lmpro/data
touch src/lmpro/data/__init__.py
touch src/lmpro/data/synth_vision.py
touch src/lmpro/data/synth_nlp.py
touch src/lmpro/data/synth_tabular.py
touch src/lmpro/data/synth_timeseries.py

# DataModules package
mkdir -p src/lmpro/datamodules
touch src/lmpro/datamodules/__init__.py
touch src/lmpro/datamodules/vision_dm.py
touch src/lmpro/datamodules/nlp_dm.py
touch src/lmpro/datamodules/tabular_dm.py
touch src/lmpro/datamodules/ts_dm.py

# Modules package with subdomains
mkdir -p src/lmpro/modules/vision
mkdir -p src/lmpro/modules/nlp
mkdir -p src/lmpro/modules/tabular
mkdir -p src/lmpro/modules/timeseries

touch src/lmpro/modules/__init__.py

# Vision modules
touch src/lmpro/modules/vision/__init__.py
touch src/lmpro/modules/vision/classifier.py
touch src/lmpro/modules/vision/segmenter.py

# NLP modules
touch src/lmpro/modules/nlp/__init__.py
touch src/lmpro/modules/nlp/char_lm.py
touch src/lmpro/modules/nlp/sentiment.py

# Tabular modules
touch src/lmpro/modules/tabular/__init__.py
touch src/lmpro/modules/tabular/mlp_reg_cls.py

# Timeseries modules
touch src/lmpro/modules/timeseries/__init__.py
touch src/lmpro/modules/timeseries/forecaster.py

# Callbacks package
mkdir -p src/lmpro/callbacks
touch src/lmpro/callbacks/__init__.py
touch src/lmpro/callbacks/checkpoints.py
touch src/lmpro/callbacks/ema.py
touch src/lmpro/callbacks/swa.py

# Loops package
mkdir -p src/lmpro/loops
touch src/lmpro/loops/__init__.py
touch src/lmpro/loops/kfold_loop.py
touch src/lmpro/loops/curriculum_loop.py

# Create configs structure organized by domain
mkdir -p configs/vision
mkdir -p configs/nlp
mkdir -p configs/tabular
mkdir -p configs/timeseries
mkdir -p configs/tuning

touch configs/defaults.yaml

# Vision configs
touch configs/vision/classifier.yaml
touch configs/vision/segmenter.yaml

# NLP configs
touch configs/nlp/char_lm.yaml
touch configs/nlp/sentiment.yaml

# Tabular configs
touch configs/tabular/mlp.yaml

# Timeseries configs
touch configs/timeseries/forecaster.yaml

# Tuning configs
touch configs/tuning/lr_finder.yaml
touch configs/tuning/batch_scaler.yaml
touch configs/tuning/ablation_study.yaml

# Create scripts
mkdir -p scripts
touch scripts/train.py
touch scripts/predict.py
touch scripts/export_onnx.py
touch scripts/tune_lr.py
touch scripts/scale_batch.py
touch scripts/run_ablation.py

# Create tests
mkdir -p tests
touch tests/__init__.py
touch tests/conftest.py
touch tests/test_datamodules.py
touch tests/test_modules_shapes.py
touch tests/test_step_cpu_smoke.py
touch tests/test_configs.py

# Create notebooks structure with all directories and files
mkdir -p notebooks

# Add README to notebooks
touch notebooks/README.md

# 01 - Lightning Fundamentals
mkdir -p notebooks/01_lightning_fundamentals
touch notebooks/01_lightning_fundamentals/01_pl_architecture.ipynb
touch notebooks/01_lightning_fundamentals/02_trainer_sanity_and_debug.ipynb
touch notebooks/01_lightning_fundamentals/03_lightningcli_config_runs.ipynb

# 02 - DataModules and Metrics
mkdir -p notebooks/02_datamodules_and_metrics
touch notebooks/02_datamodules_and_metrics/04_building_datamodules.ipynb
touch notebooks/02_datamodules_and_metrics/05_torchmetrics_logging.ipynb

# 03 - Callbacks and Checkpointing
mkdir -p notebooks/03_callbacks_and_checkpointing
touch notebooks/03_callbacks_and_checkpointing/06_checkpoint_earlystop.ipynb
touch notebooks/03_callbacks_and_checkpointing/07_custom_callbacks_swa_ema.ipynb

# 04 - Performance and Scaling
mkdir -p notebooks/04_performance_and_scaling
touch notebooks/04_performance_and_scaling/08_mixed_precision_amp.ipynb
touch notebooks/04_performance_and_scaling/09_grad_accum_clip_compile.ipynb
touch notebooks/04_performance_and_scaling/10_profiler_and_perf_tuning.ipynb

# 05 - Strategies and DDP
mkdir -p notebooks/05_strategies_and_ddp
touch notebooks/05_strategies_and_ddp/11_devices_precision_strategies.ipynb
touch notebooks/05_strategies_and_ddp/12_ddp_single_node_walkthrough.ipynb

# 06 - Advanced Mechanics
mkdir -p notebooks/06_advanced_mechanics
touch notebooks/06_advanced_mechanics/13_manual_optimization_gan.ipynb
touch notebooks/06_advanced_mechanics/14_custom_loops_kfold.ipynb
touch notebooks/06_advanced_mechanics/15_curriculum_batchloop.ipynb

# 07 - Evaluation, Export, Predict
mkdir -p notebooks/07_evaluation_export_predict
touch notebooks/07_evaluation_export_predict/16_test_predict_loops.ipynb
touch notebooks/07_evaluation_export_predict/17_onnx_torchscript_export.ipynb

# 08 - Projects and Capstone
mkdir -p notebooks/08_projects_and_capstone
touch notebooks/08_projects_and_capstone/18_mini_vision_project.ipynb
touch notebooks/08_projects_and_capstone/19_mini_nlp_project.ipynb
touch notebooks/08_projects_and_capstone/20_capstone_ablation_study.ipynb

echo "✅ Project structure created successfully!"
echo ""
echo "📁 Project created: $PROJECT_NAME/"
echo "📝 Total files created: $(find $PROJECT_NAME -type f | wc -l)"
echo "📂 Total directories: $(find $PROJECT_NAME -type d | wc -l)"
echo ""
echo "🎯 Next steps:"
echo "   1. cd $PROJECT_NAME"
echo "   2. Start implementing your Lightning modules!"
echo "   3. Follow the notebook sequence (01 → 08) for learning"
echo ""
echo "🚀 Happy Lightning mastering!"