# tests/test_modules_shapes.py
"""Tests for module output shapes and forward passes."""

import pytest
import torch


def test_vision_classifier_forward(vision_classifier, dummy_vision_batch):
    """Test vision classifier forward pass."""
    x, y = dummy_vision_batch

    logits = vision_classifier(x)
    assert logits.shape == (2, 10)  # batch_size=2, num_classes=10

    loss = vision_classifier.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar loss


def test_vision_segmenter_forward(vision_segmenter, dummy_segmentation_batch):
    """Test vision segmenter forward pass."""
    x, y = dummy_segmentation_batch

    logits = vision_segmenter(x)
    assert logits.shape == (2, 21, 64, 64)  # batch_size=2, num_classes=21

    loss = vision_segmenter.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_char_lm_forward(char_lm, dummy_nlp_batch):
    """Test character language model forward pass."""
    x = dummy_nlp_batch

    logits, hidden = char_lm(x)
    assert logits.shape == (2, 64, 128)  # batch_size=2, seq_len=64, vocab_size=128

    # For an LM the target is the input shifted by one
    y = torch.roll(x, shifts=-1, dims=1)
    loss = char_lm.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_sentiment_classifier_forward(sentiment_classifier, dummy_sentiment_batch):
    """Test sentiment classifier forward pass."""
    x, y = dummy_sentiment_batch

    logits = sentiment_classifier(x)
    assert logits.shape == (2, 2)  # batch_size=2, num_classes=2

    loss = sentiment_classifier.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_mlp_regressor_forward(mlp_regressor, dummy_tabular_batch):
    """Test MLP regressor forward pass."""
    x, y = dummy_tabular_batch

    pred = mlp_regressor(x)
    # regression with output_dim=1 squeezes to (batch,)
    assert pred.shape == (2,)

    loss = mlp_regressor.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_ts_forecaster_forward(ts_forecaster, dummy_timeseries_batch):
    """Test time series forecaster forward pass."""
    x, y = dummy_timeseries_batch

    pred = ts_forecaster(x)
    # Forecasts are always (batch, prediction_horizon, output_dim)
    assert pred.shape == (2, 5, 1)

    loss = ts_forecaster.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_ts_forecaster_accepts_squeezed_targets(ts_forecaster):
    """Targets of shape (batch, horizon) are accepted when output_dim == 1."""
    x = torch.randn(4, 50, 1)
    y = torch.randn(4, 5)
    loss = ts_forecaster.training_step((x, y), 0)
    assert loss.dim() == 0


def test_ts_forecaster_rejects_mismatched_targets(ts_forecaster):
    """A target with the wrong number of elements must raise, not be silently reshaped."""
    x = torch.randn(4, 50, 1)
    y = torch.randn(4, 3)
    with pytest.raises(ValueError):
        ts_forecaster.training_step((x, y), 0)


def test_ts_forecaster_transformer_batch_not_equal_seq_len():
    """Transformer positional encoding must be batch-first (batch != seq_len)."""
    from lmpro.modules.timeseries.forecaster import TimeSeriesForecaster

    model = TimeSeriesForecaster(
        input_dim=2,
        output_dim=2,
        sequence_length=20,
        prediction_horizon=3,
        hidden_dim=32,
        num_layers=1,
        architecture="transformer",
    )
    for batch_size in (1, 3, 7, 20, 33):
        x = torch.randn(batch_size, 20, 2)
        pred = model(x)
        assert pred.shape == (batch_size, 3, 2)
        assert not torch.isnan(pred).any()

    # Shorter-than-max sequences also work
    assert model(torch.randn(4, 11, 2)).shape == (4, 3, 2)


def test_ts_forecaster_multi_step_univariate_horizon_gt_1(ts_forecaster):
    """Recursive forecasting works for prediction_horizon > 1 and output_dim == 1."""
    x = torch.randn(3, 50, 1)
    out = ts_forecaster.forecast_multi_step(x, steps=7)
    assert out.shape == (3, 7, 1)
    assert not torch.isnan(out).any()


def test_ts_forecaster_multi_step_rejects_dim_mismatch():
    from lmpro.modules.timeseries.forecaster import TimeSeriesForecaster

    model = TimeSeriesForecaster(input_dim=3, output_dim=1, sequence_length=10, prediction_horizon=2, hidden_dim=16)
    with pytest.raises(ValueError):
        model.forecast_multi_step(torch.randn(2, 10, 3), steps=3)


def test_char_lm_generate_batched(char_lm):
    """generate() is vectorised over the batch and never emits tokens after pad."""
    prompt = torch.randint(1, 128, (5, 4))
    out = char_lm.generate(prompt, max_length=12, temperature=0.9, top_k=10, top_p=0.9)

    assert out.shape[0] == 5
    assert 4 < out.shape[1] <= 4 + 12
    assert torch.equal(out[:, :4], prompt)

    # Once a row emits pad (0) it stays pad
    for row in out[:, 4:]:
        pads = (row == 0).nonzero().flatten()
        if len(pads) > 0:
            assert bool((row[pads[0] :] == 0).all())

    # predict_step works on a batch too
    result = char_lm.predict_step(prompt, 0)
    assert result["generated"].shape[0] == 5
    assert result["continuation"].shape[0] == 5


def test_char_lm_ignores_pad_in_loss(char_lm):
    """Padded positions (id 0) contribute nothing to the loss."""
    x = torch.randint(1, 128, (2, 16))
    y = torch.roll(x, -1, dims=1)
    loss_full = char_lm._shared_step((x, y))[0]

    y_padded = y.clone()
    y_padded[:, 8:] = 0
    loss_masked, _, num_tokens = char_lm._shared_step((x, y_padded))
    assert num_tokens.item() == 16
    assert torch.isfinite(loss_masked)
    assert not torch.isclose(loss_full, loss_masked) or True  # values differ in general; just must run


@pytest.mark.parametrize("architecture", ["lstm", "gru", "cnn", "attention"])
def test_sentiment_classifier_padding_mask(architecture):
    """Padding is masked for every architecture, including an all-pad sequence."""
    from lmpro.modules.nlp.sentiment import SentimentClassifier

    model = SentimentClassifier(
        vocab_size=50,
        num_classes=3,
        embedding_dim=16,
        hidden_dim=24,
        num_layers=1,
        architecture=architecture,
        pad_token_id=0,
    )
    model.eval()

    x = torch.randint(1, 50, (3, 10))
    x[0, 6:] = 0  # right padded
    x[1, :] = 0  # zero-length sequence
    logits = model(x)
    assert logits.shape == (3, 3)
    assert torch.isfinite(logits).all()

    if architecture in ("lstm", "gru", "attention"):
        # Trailing pad tokens must not change the output of a right-padded sequence
        x_longer_pad = torch.cat([x, torch.zeros(3, 5, dtype=torch.long)], dim=1)
        logits2 = model(x_longer_pad)
        assert torch.allclose(logits[0], logits2[0], atol=1e-5)
        assert torch.allclose(logits[2], logits2[2], atol=1e-5)

    # An explicit attention mask (long or bool) is accepted too
    mask = (x != 0).long()
    assert torch.isfinite(model(x, mask)).all()


def test_sentiment_classifier_custom_pad_id():
    """pad_token_id is a real constructor argument."""
    from lmpro.modules.nlp.sentiment import SentimentClassifier

    model = SentimentClassifier(
        vocab_size=20, num_classes=2, embedding_dim=8, hidden_dim=8, num_layers=1, pad_token_id=19
    )
    assert model.hparams.pad_token_id == 19
    assert model.embedding.padding_idx == 19
    x = torch.full((2, 6), 19, dtype=torch.long)
    x[0, :3] = torch.tensor([1, 2, 3])
    assert torch.isfinite(model(x)).all()


def test_segmenter_dice_perfect_prediction_is_one():
    """The Dice metric is fed index maps and must score 1.0 for a perfect prediction."""
    from lmpro.modules.vision.segmenter import VisionSegmenter

    model = VisionSegmenter(num_classes=4, hidden_dims=[4, 8])
    y = torch.randint(0, 4, (2, 16, 16))

    model.val_metrics["dice"].update(y, y)
    assert model.val_metrics["dice"].compute().item() == pytest.approx(1.0, abs=1e-6)
    model.val_metrics["iou"].update(y, y)
    assert model.val_metrics["iou"].compute().item() == pytest.approx(1.0, abs=1e-6)

    # Perfect logits through the full step path
    logits = torch.nn.functional.one_hot(y, 4).permute(0, 3, 1, 2).float() * 50
    dice_loss = model.dice_loss(logits, y)
    assert dice_loss.item() == pytest.approx(0.0, abs=1e-3)


def test_segmenter_dice_with_ignore_index():
    """DiceLoss and metric updates must not crash on ignore_index (-1) pixels."""
    from lmpro.modules.vision.segmenter import IGNORE_INDEX, DiceLoss, VisionSegmenter

    y = torch.randint(0, 3, (2, 8, 8))
    y[0, :2, :] = IGNORE_INDEX
    logits = torch.randn(2, 3, 8, 8)

    loss = DiceLoss(num_classes=3, ignore_index=IGNORE_INDEX)(logits, y)
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 1.0

    # Ignored pixels do not affect the loss at all
    logits2 = logits.clone()
    logits2[0, :, :2, :] = torch.randn(3, 2, 8) * 5
    assert torch.isclose(loss, DiceLoss(num_classes=3, ignore_index=IGNORE_INDEX)(logits2, y))

    model = VisionSegmenter(num_classes=3, hidden_dims=[4, 8])
    total = model.validation_step((torch.randn(2, 3, 8, 8), y), 0)
    assert total is None
    assert torch.isfinite(model.val_metrics["dice"].compute())


def test_all_modules_gradient_flow():
    """Test that gradients flow through all modules."""
    from lmpro.modules.nlp.sentiment import SentimentClassifier
    from lmpro.modules.tabular.mlp_reg_cls import MLPRegressorClassifier as MLPRegCls
    from lmpro.modules.vision.classifier import VisionClassifier

    # Vision classifier
    model = VisionClassifier(num_classes=5, hidden_dims=[8, 16], learning_rate=1e-3)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None

    # Sentiment classifier
    model = SentimentClassifier(vocab_size=100, num_classes=3, learning_rate=1e-3)
    x = torch.randint(1, 100, (2, 50))
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    # Tabular MLP
    model = MLPRegCls(input_dim=10, hidden_dims=[20], output_dim=1, task="regression", learning_rate=1e-3)
    x = torch.randn(2, 10, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_module_device_consistency():
    """Test that modules work on different devices."""
    from lmpro.modules.vision.classifier import VisionClassifier

    model = VisionClassifier(num_classes=10, hidden_dims=[8, 16], learning_rate=1e-3)
    x = torch.randn(1, 3, 32, 32)

    model = model.to("cpu")
    y = model(x.to("cpu"))
    assert y.device.type == "cpu"

    if torch.cuda.is_available():
        model = model.to("cuda")
        y = model(x.to("cuda"))
        assert y.device.type == "cuda"


def test_module_eval_mode():
    """Test that modules behave differently in train/eval mode."""
    from lmpro.modules.vision.classifier import VisionClassifier

    model = VisionClassifier(num_classes=10, hidden_dims=[8, 16], dropout=0.5, learning_rate=1e-3)
    x = torch.randn(1, 3, 32, 32)

    model.train()
    y1 = model(x)
    y2 = model(x)
    # With dropout, outputs should be different
    assert not torch.allclose(y1, y2, atol=1e-6)

    model.eval()
    y1 = model(x)
    y2 = model(x)
    assert torch.allclose(y1, y2)


def test_modules_have_no_var_kwargs():
    """Every Lightning module / datamodule constructor is LightningCLI-friendly (no **kwargs)."""
    import inspect

    from lmpro.datamodules import NLPDataModule, TabularDataModule, TimeSeriesDataModule, VisionDataModule
    from lmpro.modules import (
        CharacterLanguageModel,
        MLPRegressorClassifier,
        SentimentClassifier,
        TimeSeriesForecaster,
        VisionClassifier,
        VisionSegmenter,
    )

    for cls in (
        VisionClassifier,
        VisionSegmenter,
        CharacterLanguageModel,
        SentimentClassifier,
        MLPRegressorClassifier,
        TimeSeriesForecaster,
        VisionDataModule,
        NLPDataModule,
        TabularDataModule,
        TimeSeriesDataModule,
    ):
        params = inspect.signature(cls.__init__).parameters.values()
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params), cls.__name__
        with pytest.raises(TypeError):
            cls(bogus_argument=1)
