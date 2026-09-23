# tests/test_step_cpu_smoke.py
"""Smoke tests for training/validation steps on CPU."""

import lightning.pytorch as L
import torch

from lmpro.data.synth_tabular import TabularDatasetConfig
from lmpro.data.synth_timeseries import TimeSeriesDatasetConfig
from lmpro.data.synth_vision import VisionDatasetConfig


def _trainer(**kwargs) -> L.Trainer:
    defaults = dict(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    defaults.update(kwargs)
    return L.Trainer(**defaults)


def test_vision_classifier_training_step_smoke(vision_classifier, dummy_vision_batch):
    x, y = dummy_vision_batch

    loss = vision_classifier.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    val_result = vision_classifier.validation_step((x, y), 0)
    assert val_result is None or isinstance(val_result, torch.Tensor)


def test_vision_segmenter_training_step_smoke(vision_segmenter, dummy_segmentation_batch):
    x, y = dummy_segmentation_batch

    loss = vision_segmenter.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)

    val_result = vision_segmenter.validation_step((x, y), 0)
    assert val_result is None or isinstance(val_result, torch.Tensor)


def test_char_lm_training_step_smoke(char_lm, dummy_nlp_batch):
    x = dummy_nlp_batch
    y = torch.roll(x, shifts=-1, dims=1)  # Next token prediction

    loss = char_lm.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)

    val_result = char_lm.validation_step((x, y), 0)
    assert val_result is None or isinstance(val_result, torch.Tensor)


def test_sentiment_classifier_training_step_smoke(sentiment_classifier, dummy_sentiment_batch):
    x, y = dummy_sentiment_batch

    loss = sentiment_classifier.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)

    val_result = sentiment_classifier.validation_step((x, y), 0)
    assert val_result is None or isinstance(val_result, torch.Tensor)


def test_mlp_regressor_training_step_smoke(mlp_regressor, dummy_tabular_batch):
    x, y = dummy_tabular_batch

    loss = mlp_regressor.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)

    val_result = mlp_regressor.validation_step((x, y), 0)
    assert val_result is None or isinstance(val_result, torch.Tensor)


def test_ts_forecaster_training_step_smoke(ts_forecaster, dummy_timeseries_batch):
    x, y = dummy_timeseries_batch

    loss = ts_forecaster.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)

    val_result = ts_forecaster.validation_step((x, y), 0)
    assert val_result is None or isinstance(val_result, torch.Tensor)


def test_full_training_cycle_smoke():
    """Full fit + validate cycle for the vision classifier on tiny synthetic data."""
    from lmpro.datamodules.vision_dm import VisionDataModule
    from lmpro.modules.vision.classifier import VisionClassifier

    model = VisionClassifier(num_classes=10, hidden_dims=[8, 16], learning_rate=1e-3)

    datamodule = VisionDataModule(
        task="classification",
        data_config=VisionDatasetConfig(num_samples=40, image_size=(32, 32), num_classes=10),
        batch_size=2,
        num_workers=0,
        image_size=[32, 32],
    )

    trainer = _trainer(max_epochs=1, max_steps=2)
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)

    for key in ("val/loss", "val/acc", "val/f1", "val/precision", "val/recall", "val/auroc"):
        assert key in trainer.callback_metrics, key


class _MetricSampleRecorder(L.Callback):
    """Record how many samples a val metric has seen at the end of each val epoch."""

    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.samples_per_epoch = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metric = pl_module.val_metrics[self.metric_name]
        if hasattr(metric, "tp"):  # stat-score based metrics (accuracy/precision/recall/f1)
            seen = int((metric.tp + metric.fn).sum())
        else:  # AUROC keeps the raw predictions
            seen = int(sum(p.shape[0] for p in metric.preds))
        self.samples_per_epoch.append(seen)


def test_val_metrics_reset_every_epoch_tabular():
    """Over 2 epochs a val metric's state must only hold the current epoch's samples."""
    from lmpro.datamodules.tabular_dm import TabularDataModule
    from lmpro.modules.tabular.mlp_reg_cls import MLPRegressorClassifier

    dm = TabularDataModule(
        task="classification",
        data_config=TabularDatasetConfig(num_samples=200, num_features=10, num_informative=6, num_classes=3),
        batch_size=16,
        num_workers=0,
    )
    model = MLPRegressorClassifier(input_dim=10, output_dim=3, hidden_dims=[16], scheduler="cosine")

    f1_recorder = _MetricSampleRecorder("f1")
    auroc_recorder = _MetricSampleRecorder("auroc")
    trainer = _trainer(max_epochs=2, callbacks=[f1_recorder, auroc_recorder])
    trainer.fit(model, dm)

    dm.setup("fit")
    n_val = len(dm.val_dataset)
    assert n_val > 0
    assert f1_recorder.samples_per_epoch == [n_val, n_val]
    assert auroc_recorder.samples_per_epoch == [n_val, n_val]

    # After fit, Lightning has reset the logged metric objects
    assert int((model.val_metrics["f1"].tp + model.val_metrics["f1"].fn).sum()) == 0
    assert len(model.val_metrics["auroc"].preds) == 0

    for key in (
        "val/loss",
        "val/acc",
        "val/accuracy",
        "val/f1",
        "val/precision",
        "val/recall",
        "val/auroc",
        "train/loss",
        "train/acc",
    ):
        assert key in trainer.callback_metrics, key


def test_val_metrics_reset_every_epoch_vision():
    """Same guarantee for the vision classifier, whose epoch-end hook also calls compute()."""
    from lmpro.datamodules.vision_dm import VisionDataModule
    from lmpro.modules.vision.classifier import VisionClassifier

    dm = VisionDataModule(
        task="classification",
        data_config=VisionDatasetConfig(num_samples=60, image_size=(16, 16), num_classes=4),
        batch_size=8,
        num_workers=0,
    )
    model = VisionClassifier(num_classes=4, hidden_dims=[8, 16], scheduler="cosine")

    recorder = _MetricSampleRecorder("precision")
    trainer = _trainer(max_epochs=2, callbacks=[recorder])
    trainer.fit(model, dm)

    dm.setup("fit")
    n_val = len(dm.val_dataset)
    assert recorder.samples_per_epoch == [n_val, n_val]
    assert "val/best_acc" in trainer.callback_metrics


def test_test_metrics_reset_after_test():
    from lmpro.datamodules.tabular_dm import TabularDataModule
    from lmpro.modules.tabular.mlp_reg_cls import MLPRegressorClassifier

    dm = TabularDataModule(
        task="classification",
        data_config=TabularDatasetConfig(num_samples=100, num_features=10, num_informative=6, num_classes=3),
        batch_size=16,
        num_workers=0,
    )
    model = MLPRegressorClassifier(input_dim=10, output_dim=3, hidden_dims=[16])
    trainer = _trainer(max_epochs=1)
    trainer.test(model, dm)
    assert "test/f1" in trainer.callback_metrics
    assert int((model.test_metrics["f1"].tp + model.test_metrics["f1"].fn).sum()) == 0


def test_forecaster_fits_with_timeseries_datamodule():
    """The forecaster consumes TimeSeriesDataModule batches ((B, H, 1) targets) end to end."""
    from lmpro.datamodules.ts_dm import TimeSeriesDataModule
    from lmpro.modules.timeseries.forecaster import TimeSeriesForecaster

    dm = TimeSeriesDataModule(
        task="forecasting",
        dataset_type="univariate",
        data_config=TimeSeriesDatasetConfig(num_samples=64, sequence_length=20, prediction_horizon=5),
        batch_size=8,
        num_workers=0,
    )
    model = TimeSeriesForecaster(
        input_dim=1,
        output_dim=1,
        sequence_length=20,
        prediction_horizon=5,
        hidden_dim=32,
        num_layers=1,
    )

    trainer = _trainer(fast_dev_run=True)
    trainer.fit(model, dm)
    for key in ("train/loss", "val/loss", "val/mse", "val/mae", "val/rmse", "val/r2"):
        assert key in trainer.callback_metrics, key

    trainer.test(model, dm)
    assert "test/r2" in trainer.callback_metrics

    preds = trainer.predict(model, dm)
    assert preds[0]["forecast"].shape[1:] == (5, 1)


def test_forecaster_transformer_fits_with_batch_not_equal_seq_len():
    """architecture='transformer' with batch_size != sequence_length must not crash."""
    from lmpro.datamodules.ts_dm import TimeSeriesDataModule
    from lmpro.modules.timeseries.forecaster import TimeSeriesForecaster

    dm = TimeSeriesDataModule(
        task="forecasting",
        dataset_type="multivariate",
        data_config=TimeSeriesDatasetConfig(num_samples=64, sequence_length=24, prediction_horizon=4, num_features=3),
        batch_size=7,
        num_workers=0,
    )
    model = TimeSeriesForecaster(
        input_dim=3,
        output_dim=1,
        sequence_length=24,
        prediction_horizon=4,
        hidden_dim=32,
        num_layers=1,
        architecture="transformer",
    )

    trainer = _trainer(fast_dev_run=True)
    trainer.fit(model, dm)
    assert "val/mae" in trainer.callback_metrics


def test_tabular_datamodule_trailing_batch_of_one_with_batchnorm():
    """train loader drops the last batch so BatchNorm never sees a single sample."""
    from lmpro.datamodules.tabular_dm import TabularDataModule
    from lmpro.modules.tabular.mlp_reg_cls import MLPRegressorClassifier

    # 130 * 0.7 = 91 train samples -> 91 % 10 == 1 trailing sample
    dm = TabularDataModule(
        task="regression",
        data_config=TabularDatasetConfig(num_samples=130, num_features=8, num_informative=5),
        batch_size=10,
        num_workers=0,
    )
    dm.setup("fit")
    assert len(dm.train_dataset) % 10 == 1
    assert dm.train_dataloader().drop_last is True

    model = MLPRegressorClassifier(input_dim=8, output_dim=1, task="regression", hidden_dims=[16], use_batch_norm=True)
    trainer = _trainer(max_epochs=1)
    trainer.fit(model, dm)  # would raise "Expected more than 1 value per channel" without drop_last
    assert "val/r2" in trainer.callback_metrics


def test_char_lm_fits_with_nlp_datamodule():
    """Char LM + NLPDataModule: vocab includes pad at 0 and perplexity is exp(mean loss)."""
    from lmpro.data.synth_nlp import NLPDatasetConfig
    from lmpro.datamodules.nlp_dm import NLPDataModule
    from lmpro.modules.nlp.char_lm import CharacterLanguageModel

    dm = NLPDataModule(
        task="language_modeling",
        data_config=NLPDatasetConfig(num_samples=300, max_sequence_length=32),
        batch_size=8,
        num_workers=0,
    )
    dm.setup("fit")
    assert dm.word_to_idx["<pad>"] == 0

    model = CharacterLanguageModel(vocab_size=dm.vocab_size, embedding_dim=16, hidden_dim=32, num_layers=1)
    trainer = _trainer(max_epochs=1, limit_train_batches=3, limit_val_batches=2)
    trainer.fit(model, dm)

    metrics = trainer.callback_metrics
    for key in ("train/loss", "train/perplexity", "val/loss", "val/perplexity", "val/accuracy"):
        assert key in metrics, key
    assert torch.isclose(metrics["val/perplexity"], torch.exp(metrics["val/loss"]), rtol=1e-3)
    # Running means are reset for the next epoch
    assert model.val_loss_mean.weight.item() == 0


def test_optimizer_step_smoke():
    """Test that optimizer steps work correctly."""
    from lmpro.modules.vision.classifier import VisionClassifier

    model = VisionClassifier(num_classes=10, hidden_dims=[8, 16], learning_rate=1e-3)
    initial_params = [p.clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    loss = torch.nn.functional.cross_entropy(model(x), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert any(
        not torch.allclose(initial, current, atol=1e-7) for initial, current in zip(initial_params, model.parameters())
    ), "Parameters should change after optimizer step"


def test_lr_scheduler_smoke():
    """Test that learning rate schedulers work."""
    from lmpro.modules.vision.classifier import VisionClassifier

    model = VisionClassifier(num_classes=10, hidden_dims=[8, 16], learning_rate=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    assert optimizer.param_groups[0]["lr"] == 1e-3
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    assert isinstance(current_lr, float)
    assert current_lr > 0
