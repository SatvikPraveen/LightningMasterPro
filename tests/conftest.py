# tests/conftest.py
"""Pytest fixtures and configuration for testing."""

import pytest
import torch

from lmpro.data.synth_nlp import NLPDatasetConfig
from lmpro.data.synth_tabular import TabularDatasetConfig
from lmpro.data.synth_timeseries import TimeSeriesDatasetConfig
from lmpro.data.synth_vision import VisionDatasetConfig
from lmpro.datamodules.nlp_dm import NLPDataModule
from lmpro.datamodules.tabular_dm import TabularDataModule
from lmpro.datamodules.ts_dm import TimeSeriesDataModule
from lmpro.datamodules.vision_dm import VisionDataModule
from lmpro.modules.nlp.char_lm import CharacterLanguageModel as CharacterLM
from lmpro.modules.nlp.sentiment import SentimentClassifier
from lmpro.modules.tabular.mlp_reg_cls import MLPRegressorClassifier as MLPRegCls
from lmpro.modules.timeseries.forecaster import TimeSeriesForecaster
from lmpro.modules.vision.classifier import VisionClassifier
from lmpro.modules.vision.segmenter import VisionSegmenter

# ─── DataModules (small synthetic configs, real constructor arguments) ────────


@pytest.fixture
def vision_datamodule():
    """Vision classification datamodule with 32x32 images and batch_size=4."""
    return VisionDataModule(
        task="classification",
        data_config=VisionDatasetConfig(num_samples=40, image_size=(32, 32), num_classes=10),
        batch_size=4,
        num_workers=0,
        image_size=[32, 32],
    )


@pytest.fixture
def nlp_datamodule():
    """Character-level language-modeling datamodule with sequence length 64."""
    return NLPDataModule(
        task="language_modeling",
        data_config=NLPDatasetConfig(num_samples=200, max_sequence_length=64),
        batch_size=4,
        num_workers=0,
    )


@pytest.fixture
def tabular_datamodule():
    """Tabular regression datamodule."""
    return TabularDataModule(
        task="regression",
        dataset_type="simple",
        data_config=TabularDatasetConfig(num_samples=200),
        batch_size=4,
        num_workers=0,
        normalize_features=True,
    )


@pytest.fixture
def timeseries_datamodule():
    """Univariate forecasting datamodule: 50-step inputs, 5-step horizon."""
    return TimeSeriesDataModule(
        task="forecasting",
        dataset_type="univariate",
        data_config=TimeSeriesDatasetConfig(num_samples=120),
        batch_size=4,
        num_workers=0,
        sequence_length=50,
        prediction_length=5,
    )


# ─── Modules ─────────────────────────────────────────────────────────────────


@pytest.fixture
def vision_classifier():
    """Small ResNet-style vision classifier."""
    return VisionClassifier(
        num_classes=10,
        architecture="resnet",
        hidden_dims=[16, 32, 64],
        learning_rate=1e-3,
    )


@pytest.fixture
def vision_segmenter():
    """Small UNet segmenter with 21 classes."""
    return VisionSegmenter(
        num_classes=21,
        hidden_dims=[8, 16, 32],
        learning_rate=1e-3,
    )


@pytest.fixture
def char_lm():
    """Character language model with a 128-token vocabulary."""
    return CharacterLM(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        learning_rate=1e-3,
    )


@pytest.fixture
def sentiment_classifier():
    """Binary LSTM sentiment classifier."""
    return SentimentClassifier(
        vocab_size=1000,
        embedding_dim=64,
        hidden_dim=128,
        num_classes=2,
        learning_rate=1e-3,
    )


@pytest.fixture
def mlp_regressor():
    """MLP regressor with scalar output."""
    return MLPRegCls(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=1,
        task="regression",
        learning_rate=1e-3,
    )


@pytest.fixture
def ts_forecaster():
    """Univariate LSTM forecaster: 50-step input, 5-step horizon."""
    return TimeSeriesForecaster(
        input_dim=1,
        output_dim=1,
        sequence_length=50,
        prediction_horizon=5,
        hidden_dim=64,
        num_layers=2,
        learning_rate=1e-3,
    )


# ─── Dummy batches ───────────────────────────────────────────────────────────


@pytest.fixture
def dummy_vision_batch():
    return torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,))


@pytest.fixture
def dummy_segmentation_batch():
    return torch.randn(2, 3, 64, 64), torch.randint(0, 21, (2, 64, 64))


@pytest.fixture
def dummy_nlp_batch():
    """Token ids in [1, 128): id 0 is the pad / ignore index."""
    return torch.randint(1, 128, (2, 64))


@pytest.fixture
def dummy_sentiment_batch():
    return torch.randint(1, 1000, (2, 512)), torch.randint(0, 2, (2,))


@pytest.fixture
def dummy_tabular_batch():
    return torch.randn(2, 20), torch.randn(2)


@pytest.fixture
def dummy_timeseries_batch():
    """(batch, seq_len, input_dim) inputs and (batch, horizon, output_dim) targets."""
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


@pytest.fixture(autouse=True)
def _run_from_tmp_dir(tmp_path, monkeypatch):
    """Run every test from a scratch directory so relative log/checkpoint paths never land in the repo."""
    monkeypatch.chdir(tmp_path)
