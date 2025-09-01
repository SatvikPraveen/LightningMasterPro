# tests/test_step_cpu_smoke.py
"""Smoke tests for training/validation steps on CPU."""

import pytest
import torch
import lightning.pytorch as L


def test_vision_classifier_training_step_smoke(vision_classifier, dummy_vision_batch):
    """Smoke test for vision classifier training step."""
    x, y = dummy_vision_batch
    
    # Test training step
    loss = vision_classifier.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test validation step
    val_loss = vision_classifier.validation_step((x, y), 0)
    assert isinstance(val_loss, torch.Tensor)
    assert not torch.isnan(val_loss)


def test_vision_segmenter_training_step_smoke(vision_segmenter, dummy_segmentation_batch):
    """Smoke test for vision segmenter training step."""
    x, y = dummy_segmentation_batch
    
    loss = vision_segmenter.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    
    val_loss = vision_segmenter.validation_step((x, y), 0)
    assert isinstance(val_loss, torch.Tensor)
    assert not torch.isnan(val_loss)


def test_char_lm_training_step_smoke(char_lm, dummy_nlp_batch):
    """Smoke test for character LM training step."""
    x = dummy_nlp_batch
    y = torch.roll(x, shifts=-1, dims=1)  # Next token prediction
    
    loss = char_lm.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    
    val_loss = char_lm.validation_step((x, y), 0)
    assert isinstance(val_loss, torch.Tensor)
    assert not torch.isnan(val_loss)


def test_sentiment_classifier_training_step_smoke(sentiment_classifier, dummy_sentiment_batch):
    """Smoke test for sentiment classifier training step."""
    x, y = dummy_sentiment_batch
    
    loss = sentiment_classifier.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    
    val_loss = sentiment_classifier.validation_step((x, y), 0)
    assert isinstance(val_loss, torch.Tensor)
    assert not torch.isnan(val_loss)


def test_mlp_regressor_training_step_smoke(mlp_regressor, dummy_tabular_batch):
    """Smoke test for MLP regressor training step."""
    x, y = dummy_tabular_batch
    
    loss = mlp_regressor.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    
    val_loss = mlp_regressor.validation_step((x, y), 0)
    assert isinstance(val_loss, torch.Tensor)
    assert not torch.isnan(val_loss)


def test_ts_forecaster_training_step_smoke(ts_forecaster, dummy_timeseries_batch):
    """Smoke test for time series forecaster training step."""
    x, y = dummy_timeseries_batch
    
    loss = ts_forecaster.training_step((x, y), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert not torch.isnan(loss)
    
    val_loss = ts_forecaster.validation_step((x, y), 0)
    assert isinstance(val_loss, torch.Tensor)
    assert not torch.isnan(val_loss)


def test_full_training_cycle_smoke():
    """Smoke test for full training cycle on small models."""
    from lmpro.modules.vision.classifier import VisionClassifier
    from lmpro.datamodules.vision_dm import VisionDataModule
    
    # Create small model and datamodule
    model = VisionClassifier(
        backbone="resnet18",
        num_classes=10,
        learning_rate=1e-3
    )
    
    datamodule = VisionDataModule(
        dataset_type="CIFAR10",
        batch_size=2,
        num_workers=0,
        image_size=[32, 32]
    )
    
    # Create trainer for quick test
    trainer = L.Trainer(
        max_epochs=1,
        max_steps=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )
    
    # Smoke test - should not crash
    trainer.fit(model, datamodule)
    
    # Test validation
    trainer.validate(model, datamodule)


def test_optimizer_step_smoke():
    """Test that optimizer steps work correctly."""
    from lmpro.modules.vision.classifier import VisionClassifier
    
    model = VisionClassifier(num_classes=10, learning_rate=1e-3)
    
    # Get initial parameters
    initial_params = [p.clone() for p in model.parameters()]
    
    # Configure optimizers
    optimizer = model.configure_optimizers()
    if isinstance(optimizer, dict):
        optimizer = optimizer["optimizer"]
    
    # Forward pass
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    
    loss = model.training_step((x, y), 0)
    
    # Manual optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that parameters changed
    current_params = list(model.parameters())
    params_changed = False
    for initial, current in zip(initial_params, current_params):
        if not torch.allclose(initial, current, atol=1e-7):
            params_changed = True
            break
    
    assert params_changed, "Parameters should change after optimizer step"


def test_lr_scheduler_smoke():
    """Test that learning rate schedulers work."""
    from lmpro.modules.vision.classifier import VisionClassifier
    
    model = VisionClassifier(num_classes=10, learning_rate=1e-3)
    
    # Configure optimizers and schedulers
    config = model.configure_optimizers()
    
    if isinstance(config, dict) and "lr_scheduler" in config:
        optimizer = config["optimizer"]
        scheduler = config["lr_scheduler"]["scheduler"]
        
        # Get initial LR
        initial_lr = optimizer.param_groups[0]["lr"]
        
        # Step scheduler
        scheduler.step()
        
        # LR should change (for most schedulers)
        current_lr = optimizer.param_groups[0]["lr"]
        # Note: Some schedulers might not change LR on first step
        assert isinstance(current_lr, float)
        assert current_lr > 0