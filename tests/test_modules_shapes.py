# tests/test_modules_shapes.py
"""Tests for module output shapes and forward passes."""

import pytest
import torch


def test_vision_classifier_forward(vision_classifier, dummy_vision_batch):
    """Test vision classifier forward pass."""
    x, y = dummy_vision_batch
    
    # Forward pass
    logits = vision_classifier(x)
    
    # Check output shape
    assert logits.shape == (2, 10)  # batch_size=2, num_classes=10
    
    # Test training step
    loss = vision_classifier.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar loss


def test_vision_segmenter_forward(vision_segmenter, dummy_segmentation_batch):
    """Test vision segmenter forward pass."""
    x, y = dummy_segmentation_batch
    
    # Forward pass
    logits = vision_segmenter(x)
    
    # Check output shape
    assert logits.shape == (2, 21, 256, 256)  # batch_size=2, num_classes=21
    
    # Test training step
    loss = vision_segmenter.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_char_lm_forward(char_lm, dummy_nlp_batch):
    """Test character language model forward pass."""
    x = dummy_nlp_batch
    
    # Forward pass
    logits = char_lm(x)
    
    # Check output shape
    assert logits.shape == (2, 64, 128)  # batch_size=2, seq_len=64, vocab_size=128
    
    # Test training step (for LM, target is input shifted by 1)
    y = torch.roll(x, shifts=-1, dims=1)
    loss = char_lm.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_sentiment_classifier_forward(sentiment_classifier, dummy_sentiment_batch):
    """Test sentiment classifier forward pass."""
    x, y = dummy_sentiment_batch
    
    # Forward pass
    logits = sentiment_classifier(x)
    
    # Check output shape
    assert logits.shape == (2, 2)  # batch_size=2, num_classes=2
    
    # Test training step
    loss = sentiment_classifier.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_mlp_regressor_forward(mlp_regressor, dummy_tabular_batch):
    """Test MLP regressor forward pass."""
    x, y = dummy_tabular_batch
    
    # Forward pass
    pred = mlp_regressor(x)
    
    # Check output shape
    assert pred.shape == (2, 1)  # batch_size=2, output_dim=1
    
    # Test training step
    loss = mlp_regressor.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_ts_forecaster_forward(ts_forecaster, dummy_timeseries_batch):
    """Test time series forecaster forward pass."""
    x, y = dummy_timeseries_batch
    
    # Forward pass
    pred = ts_forecaster(x)
    
    # Check output shape
    assert pred.shape == (2, 5, 1)  # batch_size=2, pred_len=5, input_dim=1
    
    # Test training step
    loss = ts_forecaster.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_all_modules_gradient_flow():
    """Test that gradients flow through all modules."""
    from lmpro.modules.vision.classifier import VisionClassifier
    from lmpro.modules.nlp.sentiment import SentimentClassifier
    from lmpro.modules.tabular.mlp_reg_cls import MLPRegCls
    
    # Vision classifier
    model = VisionClassifier(num_classes=5, learning_rate=1e-3)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    
    # Sentiment classifier
    model = SentimentClassifier(vocab_size=100, num_classes=3, learning_rate=1e-3)
    x = torch.randint(0, 100, (2, 50))
    y = model(x)
    loss = y.sum()
    loss.backward()
    # Check if model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            break
    
    # Tabular MLP
    model = MLPRegCls(input_dim=10, hidden_dims=[20], output_dim=1, learning_rate=1e-3)
    x = torch.randn(2, 10, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_module_device_consistency():
    """Test that modules work on different devices."""
    from lmpro.modules.vision.classifier import VisionClassifier
    
    model = VisionClassifier(num_classes=10, learning_rate=1e-3)
    x = torch.randn(1, 3, 32, 32)
    
    # CPU test
    model = model.to("cpu")
    x = x.to("cpu")
    y = model(x)
    assert y.device.type == "cpu"
    
    # GPU test (if available)
    if torch.cuda.is_available():
        model = model.to("cuda")
        x = x.to("cuda")
        y = model(x)
        assert y.device.type == "cuda"


def test_module_eval_mode():
    """Test that modules behave differently in train/eval mode."""
    from lmpro.modules.vision.classifier import VisionClassifier
    
    model = VisionClassifier(num_classes=10, dropout_rate=0.5, learning_rate=1e-3)
    x = torch.randn(1, 3, 32, 32)
    
    # Training mode
    model.train()
    y1 = model(x)
    y2 = model(x)
    # With dropout, outputs should be different
    assert not torch.allclose(y1, y2, atol=1e-6)
    
    # Eval mode
    model.eval()
    y1 = model(x)
    y2 = model(x)
    # Without dropout, outputs should be identical
    assert torch.allclose(y1, y2)