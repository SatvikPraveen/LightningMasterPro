# File: src/lmpro/callbacks/ema.py

"""
Exponential Moving Average callback for model weights
"""

import torch
from copy import deepcopy
from typing import Any, Dict, Optional
from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer


class EMACallback(Callback):
    """
    Exponential Moving Average callback
    
    Maintains exponential moving average of model weights during training
    and optionally uses EMA weights for validation/testing
    """
    
    def __init__(
        self,
        decay: float = 0.999,
        start_epoch: int = 0,
        update_every: int = 1,
        use_ema_for_validation: bool = True,
        save_ema_checkpoint: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.decay = decay
        self.start_epoch = start_epoch
        self.update_every = update_every
        self.use_ema_for_validation = use_ema_for_validation
        self.save_ema_checkpoint = save_ema_checkpoint
        
        self.ema_model = None
        self.original_model_state = None
        self.update_counter = 0
        
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize EMA model"""
        # Create EMA model as a deep copy
        self.ema_model = deepcopy(pl_module.state_dict())
        
        # Move to same device
        for key in self.ema_model:
            if isinstance(self.ema_model[key], torch.Tensor):
                self.ema_model[key] = self.ema_model[key].to(pl_module.device)
    
    def on_train_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Any,
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Update EMA after each training batch"""
        if trainer.current_epoch < self.start_epoch:
            return
            
        self.update_counter += 1
        
        if self.update_counter % self.update_every == 0:
            self._update_ema(pl_module)
    
    def _update_ema(self, pl_module: LightningModule) -> None:
        """Update exponential moving average"""
        current_state = pl_module.state_dict()
        
        for key in self.ema_model:
            if isinstance(current_state[key], torch.Tensor):
                self.ema_model[key] = (
                    self.decay * self.ema_model[key] + 
                    (1 - self.decay) * current_state[key].detach()
                )
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Switch to EMA weights for validation"""
        if self.use_ema_for_validation and self.ema_model is not None:
            # Save original state
            self.original_model_state = deepcopy(pl_module.state_dict())
            
            # Load EMA weights
            pl_module.load_state_dict(self.ema_model, strict=False)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore original weights after validation"""
        if self.use_ema_for_validation and self.original_model_state is not None:
            # Restore original weights
            pl_module.load_state_dict(self.original_model_state, strict=False)
            self.original_model_state = None
    
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Switch to EMA weights for testing"""
        if self.ema_model is not None:
            self.original_model_state = deepcopy(pl_module.state_dict())
            pl_module.load_state_dict(self.ema_model, strict=False)
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore original weights after testing"""
        if self.original_model_state is not None:
            pl_module.load_state_dict(self.original_model_state, strict=False)
            self.original_model_state = None
    
    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Save EMA state in checkpoint"""
        if self.save_ema_checkpoint and self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model
            checkpoint['ema_decay'] = self.decay
            checkpoint['ema_update_counter'] = self.update_counter
    
    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint"""
        if 'ema_state_dict' in checkpoint:
            self.ema_model = checkpoint['ema_state_dict']
            self.decay = checkpoint.get('ema_decay', self.decay)
            self.update_counter = checkpoint.get('ema_update_counter', 0)
    
    def apply_ema_weights(self, pl_module: LightningModule) -> None:
        """Manually apply EMA weights to model"""
        if self.ema_model is not None:
            pl_module.load_state_dict(self.ema_model, strict=False)
    
    def get_ema_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get EMA model state dict"""
        return self.ema_model
    
    def state_dict(self) -> Dict[str, Any]:
        """Get callback state"""
        return {
            'ema_model': self.ema_model,
            'decay': self.decay,
            'update_counter': self.update_counter,
            'start_epoch': self.start_epoch,
            'update_every': self.update_every,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state"""
        self.ema_model = state_dict.get('ema_model')
        self.decay = state_dict.get('decay', self.decay)
        self.update_counter = state_dict.get('update_counter', 0)
        self.start_epoch = state_dict.get('start_epoch', self.start_epoch)
        self.update_every = state_dict.get('update_every', self.update_every)


def create_ema_callback(
    decay: float = 0.999,
    start_epoch: int = 0,
    use_for_validation: bool = True,
    **kwargs
) -> EMACallback:
    """Create EMA callback with common settings"""
    return EMACallback(
        decay=decay,
        start_epoch=start_epoch,
        use_ema_for_validation=use_for_validation,
        **kwargs
    )