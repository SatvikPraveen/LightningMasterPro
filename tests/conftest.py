# tests/conftest.py
"""Pytest fixtures and configuration for testing."""

import sys
from pathlib import Path
import pytest
import torch
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.datamodules.vision_dm import VisionDataModule
from lmpro.datamodules.nlp_dm import NLPDataModule
from lmpro.datamodules.tabular_dm import TabularDataModule
from lmpro.datamodules.ts_dm import TimeSeriesDataModule

from lmpro.modules.vision.classifier import VisionClassifier
from lmpro.modules.vision.segmenter import VisionSegmenter
from lmpro.modules.nlp.char_lm import CharacterLM
from lmpro.modules.nlp.sentiment import SentimentClassifier
from lmpro.modules.tabular.mlp_reg_cls import MLPRegCls
from lmpro.modules.timeseries.forecaster import TimeSeriesForecaster


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def vision_datamodule(temp_data_dir):
    """Create vision datamodule for testing."""
    return VisionDataModule(
        data_dir=str(temp_data_dir),
        dataset_type="CIFAR10",
        batch_size=4,
        num_workers=0,
        image_size=[32, 32]
    )


@pytest.fixture
def nlp_datamodule(temp_data_dir):
    """Create NLP datamodule for testing."""
    return NLPDataModule(
        data_dir=str(temp_data_dir),
        dataset_type="char_lm",
        batch_size=4,
        num_workers=0,
        sequence_length=64,
        vocab_size=128
    )


@pytest.fixture
def tabular_datamodule(temp_data_dir):
    """Create tabular datamodule for testing."""
    return TabularDataModule(
        data_dir=str(temp_data_dir),
        dataset_type="regression",
        batch_size=4,
        num_workers=0,
        normalize_features=True
    )


@pytest.fixture
def timeseries_datamodule(temp_data_dir):
    """Create time series datamodule for testing."""
    return TimeSeriesDataModule(
        data_dir=str(temp_data_dir),
        dataset_type="univariate",
        batch_size=4,
        num_workers=0,
        sequence_length=50,
        prediction_length=5
    )


@pytest.fixture
def vision_classifier():
    """Create vision classifier for testing."""
    return VisionClassifier(
        backbone="resnet18",
        num_classes=10,
        learning_rate=1e-3
    )


@pytest.fixture
def vision_segmenter():
    """Create vision segmenter for testing."""
    return VisionSegmenter(
        backbone="resnet18",
        num_classes=21,
        learning_rate=1e-3
    )


@pytest.fixture
def char_lm():
    """Create character language model for testing."""
    return CharacterLM(
        vocab_size=128,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        learning_rate=1e-3
    )


@pytest.fixture
def sentiment_classifier():
    """Create sentiment classifier for testing."""
    return SentimentClassifier(
        vocab_size=1000,
        embed_dim=64,
        hidden_dim=128,
        num_classes=2,
        learning_rate=1e-3
    )


@pytest.fixture
def mlp_regressor():
    """Create MLP regressor for testing."""
    return MLPRegCls(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=1,
        task_type="regression",
        learning_rate=1e-3
    )


@pytest.fixture
def ts_forecaster():
    """Create time series forecaster for testing."""
    return TimeSeriesForecaster(
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        sequence_length=50,
        prediction_length=5,
        learning_rate=1e-3
    )


@pytest.fixture
def dummy_vision_batch():
    """Create dummy vision batch."""
    return torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,))


@pytest.fixture
def dummy_segmentation_batch():
    """Create dummy segmentation batch."""
    return torch.randn(2, 3, 256, 256), torch.randint(0, 21, (2, 256, 256))


@pytest.fixture
def dummy_nlp_batch():
    """Create dummy NLP batch."""
    return torch.randint(0, 128, (2, 64))


@pytest.fixture
def dummy_sentiment_batch():
    """Create dummy sentiment batch."""
    return torch.randint(0, 1000, (2, 512)), torch.randint(0, 2, (2,))


@pytest.fixture
def dummy_tabular_batch():
    """Create dummy tabular batch."""
    return torch.randn(2, 20), torch.randn(2, 1)


@pytest.fixture
def dummy_timeseries_batch():
    """Create dummy time series batch."""
    x = torch.randn(2, 50, 1)
    y = torch.randn(2, 5, 1)
    return x, y


@pytest.fixture(autouse=True)
def set_deterministic():
    """Set deterministic behavior for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False