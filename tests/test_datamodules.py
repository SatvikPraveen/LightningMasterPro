# tests/test_datamodules.py
"""Tests for all datamodules."""

import pytest
import torch
from torch.utils.data import DataLoader


def test_vision_datamodule_setup(vision_datamodule):
    """Test vision datamodule setup."""
    vision_datamodule.setup("fit")
    
    assert hasattr(vision_datamodule, 'train_dataset')
    assert hasattr(vision_datamodule, 'val_dataset')
    assert len(vision_datamodule.train_dataset) > 0
    assert len(vision_datamodule.val_dataset) > 0


def test_vision_datamodule_dataloaders(vision_datamodule):
    """Test vision datamodule dataloaders."""
    vision_datamodule.setup("fit")
    
    train_loader = vision_datamodule.train_dataloader()
    val_loader = vision_datamodule.val_dataloader()
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # Test batch shapes
    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape == (4, 3, 32, 32)  # batch_size=4
    assert y.shape == (4,)


def test_nlp_datamodule_setup(nlp_datamodule):
    """Test NLP datamodule setup."""
    nlp_datamodule.setup("fit")
    
    assert hasattr(nlp_datamodule, 'train_dataset')
    assert hasattr(nlp_datamodule, 'val_dataset')
    assert len(nlp_datamodule.train_dataset) > 0
    assert len(nlp_datamodule.val_dataset) > 0


def test_nlp_datamodule_dataloaders(nlp_datamodule):
    """Test NLP datamodule dataloaders."""
    nlp_datamodule.setup("fit")
    
    train_loader = nlp_datamodule.train_dataloader()
    val_loader = nlp_datamodule.val_dataloader()
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # Test batch shapes
    batch = next(iter(train_loader))
    assert batch.shape == (4, 64)  # batch_size=4, sequence_length=64


def test_tabular_datamodule_setup(tabular_datamodule):
    """Test tabular datamodule setup."""
    tabular_datamodule.setup("fit")
    
    assert hasattr(tabular_datamodule, 'train_dataset')
    assert hasattr(tabular_datamodule, 'val_dataset')
    assert len(tabular_datamodule.train_dataset) > 0
    assert len(tabular_datamodule.val_dataset) > 0


def test_tabular_datamodule_dataloaders(tabular_datamodule):
    """Test tabular datamodule dataloaders."""
    tabular_datamodule.setup("fit")
    
    train_loader = tabular_datamodule.train_dataloader()
    val_loader = tabular_datamodule.val_dataloader()
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # Test batch shapes
    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape[0] == 4  # batch_size=4
    assert y.shape[0] == 4


def test_timeseries_datamodule_setup(timeseries_datamodule):
    """Test time series datamodule setup."""
    timeseries_datamodule.setup("fit")
    
    assert hasattr(timeseries_datamodule, 'train_dataset')
    assert hasattr(timeseries_datamodule, 'val_dataset')
    assert len(timeseries_datamodule.train_dataset) > 0
    assert len(timeseries_datamodule.val_dataset) > 0


def test_timeseries_datamodule_dataloaders(timeseries_datamodule):
    """Test time series datamodule dataloaders."""
    timeseries_datamodule.setup("fit")
    
    train_loader = timeseries_datamodule.train_dataloader()
    val_loader = timeseries_datamodule.val_dataloader()
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # Test batch shapes
    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape == (4, 50, 1)  # batch_size=4, sequence_length=50, input_dim=1
    assert y.shape == (4, 5, 1)   # batch_size=4, prediction_length=5, input_dim=1


def test_all_datamodules_batch_size_consistency():
    """Test that all datamodules respect batch_size parameter."""
    from lmpro.datamodules.vision_dm import VisionDataModule
    from lmpro.datamodules.nlp_dm import NLPDataModule
    from lmpro.datamodules.tabular_dm import TabularDataModule
    from lmpro.datamodules.ts_dm import TimeSeriesDataModule
    
    batch_size = 8
    
    # Vision
    vision_dm = VisionDataModule(batch_size=batch_size, num_workers=0)
    vision_dm.setup("fit")
    batch = next(iter(vision_dm.train_dataloader()))
    assert batch[0].shape[0] == batch_size
    
    # NLP
    nlp_dm = NLPDataModule(batch_size=batch_size, num_workers=0)
    nlp_dm.setup("fit")
    batch = next(iter(nlp_dm.train_dataloader()))
    assert batch.shape[0] == batch_size
    
    # Tabular
    tabular_dm = TabularDataModule(batch_size=batch_size, num_workers=0)
    tabular_dm.setup("fit")
    batch = next(iter(tabular_dm.train_dataloader()))
    assert batch[0].shape[0] == batch_size
    
    # Time series
    ts_dm = TimeSeriesDataModule(batch_size=batch_size, num_workers=0)
    ts_dm.setup("fit")
    batch = next(iter(ts_dm.train_dataloader()))
    assert batch[0].shape[0] == batch_size