# tests/test_datamodules.py
"""Tests for all datamodules."""

import pickle

from torch.utils.data import DataLoader


def test_vision_datamodule_setup(vision_datamodule):
    vision_datamodule.setup("fit")

    assert hasattr(vision_datamodule, "train_dataset")
    assert hasattr(vision_datamodule, "val_dataset")
    assert len(vision_datamodule.train_dataset) > 0
    assert len(vision_datamodule.val_dataset) > 0


def test_vision_datamodule_dataloaders(vision_datamodule):
    vision_datamodule.setup("fit")

    train_loader = vision_datamodule.train_dataloader()
    val_loader = vision_datamodule.val_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    x, y = next(iter(train_loader))
    assert x.shape == (4, 3, 32, 32)  # batch_size=4, image_size=[32,32] from fixture
    assert y.shape == (4,)


def test_vision_datamodule_transforms_are_picklable(vision_datamodule):
    """Transforms must survive pickling (spawn-based DataLoader workers on macOS)."""
    vision_datamodule.setup("fit")
    for tf in (vision_datamodule.train_transforms, vision_datamodule.val_transforms, vision_datamodule.test_transforms):
        pickle.dumps(tf)
    pickle.dumps(vision_datamodule.train_dataset.transform)


def test_nlp_datamodule_setup(nlp_datamodule):
    nlp_datamodule.setup("fit")

    assert hasattr(nlp_datamodule, "train_dataset")
    assert hasattr(nlp_datamodule, "val_dataset")
    assert len(nlp_datamodule.train_dataset) > 0
    assert len(nlp_datamodule.val_dataset) > 0
    assert nlp_datamodule.vocab_size == len(nlp_datamodule.train_dataset.chars)
    assert nlp_datamodule.word_to_idx["<pad>"] == nlp_datamodule.pad_token_id == 0


def test_nlp_datamodule_dataloaders(nlp_datamodule):
    nlp_datamodule.setup("fit")

    train_loader = nlp_datamodule.train_dataloader()
    val_loader = nlp_datamodule.val_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    batch = next(iter(train_loader))
    assert isinstance(batch, (list, tuple))
    x, y = batch
    assert x.shape == (4, 64)
    assert y.shape == (4, 64)
    assert (x > 0).all()  # id 0 is reserved for padding


def test_tabular_datamodule_setup(tabular_datamodule):
    tabular_datamodule.setup("fit")

    assert hasattr(tabular_datamodule, "train_dataset")
    assert hasattr(tabular_datamodule, "val_dataset")
    assert len(tabular_datamodule.train_dataset) > 0
    assert len(tabular_datamodule.val_dataset) > 0


def test_tabular_datamodule_dataloaders(tabular_datamodule):
    tabular_datamodule.setup("fit")

    train_loader = tabular_datamodule.train_dataloader()
    val_loader = tabular_datamodule.val_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert train_loader.drop_last is True
    assert val_loader.drop_last is False

    x, y = next(iter(train_loader))
    assert x.shape[0] == 4
    assert y.shape[0] == 4


def test_timeseries_datamodule_setup(timeseries_datamodule):
    timeseries_datamodule.setup("fit")

    assert hasattr(timeseries_datamodule, "train_dataset")
    assert hasattr(timeseries_datamodule, "val_dataset")
    assert len(timeseries_datamodule.train_dataset) > 0
    assert len(timeseries_datamodule.val_dataset) > 0

    info = timeseries_datamodule.get_dataset_info()
    assert info["sequence_length"] == 50
    assert info["prediction_horizon"] == 5
    assert info["num_features"] == 1


def test_timeseries_datamodule_dataloaders(timeseries_datamodule):
    timeseries_datamodule.setup("fit")

    train_loader = timeseries_datamodule.train_dataloader()
    val_loader = timeseries_datamodule.val_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    x, y = next(iter(train_loader))
    assert x.shape == (4, 50, 1)  # batch_size=4, sequence_length=50, input_dim=1
    assert y.shape == (4, 5, 1)  # batch_size=4, prediction_length=5, num_features=1


def test_datamodules_generate_data_in_setup_not_prepare_data():
    """prepare_data() must not set state; setup() alone must yield usable datasets."""
    from lmpro.data.synth_nlp import NLPDatasetConfig
    from lmpro.data.synth_tabular import TabularDatasetConfig
    from lmpro.data.synth_timeseries import TimeSeriesDatasetConfig
    from lmpro.data.synth_vision import VisionDatasetConfig
    from lmpro.datamodules.nlp_dm import NLPDataModule
    from lmpro.datamodules.tabular_dm import TabularDataModule
    from lmpro.datamodules.ts_dm import TimeSeriesDataModule
    from lmpro.datamodules.vision_dm import VisionDataModule

    dms = [
        VisionDataModule(data_config=VisionDatasetConfig(num_samples=20, image_size=(8, 8)), num_workers=0),
        NLPDataModule(task="sentiment", data_config=NLPDatasetConfig(num_samples=40), num_workers=0),
        TabularDataModule(data_config=TabularDatasetConfig(num_samples=40), num_workers=0),
        TimeSeriesDataModule(
            data_config=TimeSeriesDatasetConfig(num_samples=30, sequence_length=10, prediction_horizon=2), num_workers=0
        ),
    ]
    for dm in dms:
        assert dm.datasets is None
        dm.prepare_data()
        assert dm.datasets is None, type(dm).__name__
        dm.setup("fit")
        assert len(dm.train_dataset) > 0 and len(dm.val_dataset) > 0
        dm.setup("test")
        assert len(dm.test_dataset) > 0


def test_all_datamodules_batch_size_consistency():
    """Test that all datamodules respect batch_size parameter."""
    from lmpro.data.synth_nlp import NLPDatasetConfig
    from lmpro.data.synth_tabular import TabularDatasetConfig
    from lmpro.data.synth_timeseries import TimeSeriesDatasetConfig
    from lmpro.data.synth_vision import VisionDatasetConfig
    from lmpro.datamodules.nlp_dm import NLPDataModule
    from lmpro.datamodules.tabular_dm import TabularDataModule
    from lmpro.datamodules.ts_dm import TimeSeriesDataModule
    from lmpro.datamodules.vision_dm import VisionDataModule

    batch_size = 8

    vision_dm = VisionDataModule(
        data_config=VisionDatasetConfig(num_samples=60, image_size=(16, 16)), batch_size=batch_size, num_workers=0
    )
    vision_dm.setup("fit")
    assert next(iter(vision_dm.train_dataloader()))[0].shape[0] == batch_size

    nlp_dm = NLPDataModule(data_config=NLPDatasetConfig(num_samples=60), batch_size=batch_size, num_workers=0)
    nlp_dm.setup("fit")
    assert next(iter(nlp_dm.train_dataloader()))[0].shape[0] == batch_size

    tabular_dm = TabularDataModule(
        data_config=TabularDatasetConfig(num_samples=60), batch_size=batch_size, num_workers=0
    )
    tabular_dm.setup("fit")
    assert next(iter(tabular_dm.train_dataloader()))[0].shape[0] == batch_size

    ts_dm = TimeSeriesDataModule(
        data_config=TimeSeriesDatasetConfig(num_samples=60, sequence_length=10, prediction_horizon=2),
        batch_size=batch_size,
        num_workers=0,
    )
    ts_dm.setup("fit")
    assert next(iter(ts_dm.train_dataloader()))[0].shape[0] == batch_size
