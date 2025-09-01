# File: src/lmpro/data/__init__.py

"""
Synthetic data generation modules for different domains
"""

from .synth_vision import (
    create_synthetic_image_dataset,
    create_synthetic_segmentation_dataset,
    VisionDatasetConfig,
)

from .synth_nlp import (
    create_synthetic_text_dataset,
    create_synthetic_sentiment_dataset,
    create_character_level_dataset,
    NLPDatasetConfig,
)

from .synth_tabular import (
    create_synthetic_tabular_dataset,
    create_synthetic_regression_dataset,
    create_synthetic_classification_dataset,
    TabularDatasetConfig,
)

from .synth_timeseries import (
    create_synthetic_timeseries_dataset,
    create_synthetic_forecasting_dataset,
    TimeSeriesDatasetConfig,
)

__all__ = [
    # Vision data generation
    "create_synthetic_image_dataset",
    "create_synthetic_segmentation_dataset", 
    "VisionDatasetConfig",
    
    # NLP data generation
    "create_synthetic_text_dataset",
    "create_synthetic_sentiment_dataset",
    "create_character_level_dataset",
    "NLPDatasetConfig",
    
    # Tabular data generation
    "create_synthetic_tabular_dataset",
    "create_synthetic_regression_dataset",
    "create_synthetic_classification_dataset",
    "TabularDatasetConfig",
    
    # Time series data generation
    "create_synthetic_timeseries_dataset", 
    "create_synthetic_forecasting_dataset",
    "TimeSeriesDatasetConfig",
]