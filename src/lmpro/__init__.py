# File: src/lmpro/__init__.py

"""
LightningMasterPro - Professional PyTorch Lightning Framework
A comprehensive learning and production toolkit for PyTorch Lightning
"""

__version__ = "1.0.0"
__author__ = "LightningMasterPro Team"

# Core imports for easy access
from .cli import LightningMasterCLI

# Module imports
from .modules.vision.classifier import VisionClassifier
from .modules.vision.segmenter import VisionSegmenter
from .modules.nlp.char_lm import CharacterLanguageModel
from .modules.nlp.sentiment import SentimentClassifier
from .modules.tabular.mlp_reg_cls import MLPRegressorClassifier
from .modules.timeseries.forecaster import TimeSeriesForecaster

# DataModule imports
from .datamodules.vision_dm import VisionDataModule
from .datamodules.nlp_dm import NLPDataModule
from .datamodules.tabular_dm import TabularDataModule
from .datamodules.ts_dm import TimeSeriesDataModule

# Callback imports
from .callbacks.checkpoints import EnhancedModelCheckpoint
from .callbacks.ema import EMACallback
from .callbacks.swa import SWACallback

# Loop imports
from .loops.kfold_loop import KFoldLoop
from .loops.curriculum_loop import CurriculumLoop

# Utility imports
from .utils.seed import seed_everything_deterministic
from .utils.metrics import get_metrics_dict, log_confusion_matrix
from .utils.viz import plot_training_curves, plot_predictions

__all__ = [
    # Core
    "LightningMasterCLI",
    
    # Modules
    "VisionClassifier",
    "VisionSegmenter", 
    "CharacterLanguageModel",
    "SentimentClassifier",
    "MLPRegressorClassifier",
    "TimeSeriesForecaster",
    
    # DataModules
    "VisionDataModule",
    "NLPDataModule",
    "TabularDataModule", 
    "TimeSeriesDataModule",
    
    # Callbacks
    "EnhancedModelCheckpoint",
    "EMACallback",
    "SWACallback",
    
    # Loops
    "KFoldLoop",
    "CurriculumLoop",
    
    # Utils
    "seed_everything_deterministic",
    "get_metrics_dict",
    "log_confusion_matrix",
    "plot_training_curves",
    "plot_predictions",
]