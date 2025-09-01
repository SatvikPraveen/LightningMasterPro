# File: src/lmpro/modules/__init__.py

"""
Lightning Modules for different domains and tasks
"""

# Vision modules
from .vision.classifier import VisionClassifier
from .vision.segmenter import VisionSegmenter

# NLP modules
from .nlp.char_lm import CharacterLanguageModel
from .nlp.sentiment import SentimentClassifier

# Tabular modules
from .tabular.mlp_reg_cls import MLPRegressorClassifier

# Time series modules
from .timeseries.forecaster import TimeSeriesForecaster

__all__ = [
    # Vision
    "VisionClassifier",
    "VisionSegmenter",
    
    # NLP
    "CharacterLanguageModel", 
    "SentimentClassifier",
    
    # Tabular
    "MLPRegressorClassifier",
    
    # Time Series
    "TimeSeriesForecaster",
]