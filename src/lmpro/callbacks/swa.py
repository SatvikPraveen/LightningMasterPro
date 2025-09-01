# File: src/lmpro/callbacks/swa.py

"""
Stochastic Weight Averaging callback
"""

import torch
from copy import deepcopy
from typing import Any, Dict, Optional, Union
from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class SWACallback(Callback):
    """
    Stochastic Weight Averaging callback
    
    Implements SWA to improve model generalization by averaging weights
    from multiple epochs during training
    """
    
    def __init__(
        self,
        swa_lrs: Union[float, list] = 1e-2,
        swa_epoch_start: Union[int, float] = 0.8,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        avg_fn: Optional[callable] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        super().__init__()
        
        self.swa_lrs = swa_lrs if isinstance(swa_lrs, list) else [swa_lrs]
        self.swa_epoch_start = swa_epoch_start
        self.annealing_epochs = annealing_epochs
        self.annealing_strategy = annealing_strategy
        self.avg_fn = avg_fn or self._default_avg_fn
        self.device = device
        
        self.swa_model = None
        self.swa_n = 0
        self._swa_epoch_start_absolute = None
        self._original_optimizers = None
        self._swa_schedulers = []
        
    def _default_avg_fn(self, averaged_model_parameter: torch.Tensor, model_parameter: torch.Tensor, num_averaged: int):
        """Default averaging function"""
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize SWA"""
        # Calculate absolute start epoch
        if isinstance(self.swa_epoch_start, float):
            self._swa_epoch_start_absolute = int(trainer.max_epochs * self.swa_epoch_start)
        else:
            self._swa_epoch_start_absolute = self.swa_epoch_start
        
        rank_zero_info(f"SWA will start at epoch {self._swa_epoch_start_absolute}")
        
        # Initialize SWA model
        self.swa_model = deepcopy(pl_module.state_dict())
        if self.device is None:
            self.device = pl_module.device
        
        # Move SWA model to device
        for key in self.swa_model:
            if isinstance(self.swa_model[key], torch.Tensor):
                self.swa_model[key] = self.swa_model[key].to(self.device)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Update SWA model after each epoch"""
        if trainer.current_epoch >= self._swa_epoch_start_absolute:
            self._update_swa_model(pl_module)
            
            # Update learning rate schedulers for SWA
            if trainer.current_epoch == self._swa_epoch_start_absolute:
                self._setup_swa_schedulers(trainer)
            
            # Step SWA schedulers
            for scheduler in self._swa_schedulers:
                scheduler.step()
    
    def _update_swa_model(self, pl_module: LightningModule) -> None:
        """Update SWA model with current model weights"""
        current_state = pl_module.state_dict()
        
        for key in self.swa_model:
            if isinstance(current_state[key], torch.Tensor):
                if key in current_state:
                    self.swa_model[key] = self.avg_fn(
                        self.swa_model[key],
                        current_state[key].to(self.device),
                        self.swa_n
                    )
        
        self.swa_n += 1
        
        if self.swa_n % 10 == 0:
            rank_zero_info(f"Updated SWA model (n={self.swa_n})")
    
    def _setup_swa_schedulers(self, trainer: Trainer) -> None:
        """Setup SWA learning rate schedulers"""
        self._original_optimizers = trainer.optimizers.copy()
        self._swa_schedulers = []
        
        for i, optimizer in enumerate(trainer.optimizers):
            swa_lr = self.swa_lrs[min(i, len(self.swa_lrs) - 1)]
            
            if self.annealing_strategy == "cos":
                scheduler = torch.optim.swa_utils.SWALR(
                    optimizer, 
                    swa_lr=swa_lr,
                    anneal_epochs=self.annealing_epochs,
                    anneal_strategy="cos"
                )
            elif self.annealing_strategy == "linear":
                scheduler = torch.optim.swa_utils.SWALR(
                    optimizer,
                    swa_lr=swa_lr, 
                    anneal_epochs=self.annealing_epochs,
                    anneal_strategy="linear"
                )
            else:
                # Constant learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = swa_lr
                scheduler = None
            
            if scheduler is not None:
                self._swa_schedulers.append(scheduler)
        
        rank_zero_info(f"Setup SWA schedulers with LR: {self.swa_lrs}")
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Apply SWA model and update batch norm statistics"""
        if self.swa_model is not None and self.swa_n > 0:
            # Apply SWA weights to model
            pl_module.load_state_dict(self.swa_model, strict=False)
            
            rank_zero_info(f"Applied SWA model (averaged over {self.swa_n} epochs)")
            
            # Update batch normalization statistics
            self._update_bn_statistics(trainer, pl_module)
    
    def _update_bn_statistics(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Update batch normalization statistics with SWA model"""
        try:
            # Get training dataloader
            train_dataloader = trainer.train_dataloader
            if train_dataloader is None:
                return
            
            # Set model to training mode for BN statistics update
            pl_module.train()
            
            # Update BN statistics
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_dataloader):
                    # Move batch to device
                    if isinstance(batch, (list, tuple)):
                        batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                        x = batch[0]
                    else:
                        x = batch.to(self.device)
                    
                    # Forward pass to update BN statistics
                    pl_module(x)
                    
                    # Limit number of batches for efficiency
                    if batch_idx >= 100:
                        break
            
            rank_zero_info("Updated batch normalization statistics for SWA model")
            
        except Exception as e:
            rank_zero_info(f"Could not update BN statistics: {e}")
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Use SWA model for validation if available"""
        if (self.swa_model is not None and 
            trainer.current_epoch >= self._swa_epoch_start_absolute and 
            self.swa_n > 0):
            
            # Save original state
            self.original_model_state = deepcopy(pl_module.state_dict())
            
            # Load SWA weights
            pl_module.load_state_dict(self.swa_model, strict=False)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore original model after validation"""
        if hasattr(self, 'original_model_state') and self.original_model_state is not None:
            pl_module.load_state_dict(self.original_model_state, strict=False)
            self.original_model_state = None
    
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Use SWA model for testing"""
        if self.swa_model is not None and self.swa_n > 0:
            self.original_model_state = deepcopy(pl_module.state_dict())
            pl_module.load_state_dict(self.swa_model, strict=False)
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore original model after testing"""
        if hasattr(self, 'original_model_state') and self.original_model_state is not None:
            pl_module.load_state_dict(self.original_model_state, strict=False)
            self.original_model_state = None
    
    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Save SWA state in checkpoint"""
        if self.swa_model is not None:
            checkpoint['swa_state_dict'] = self.swa_model
            checkpoint['swa_n'] = self.swa_n
            checkpoint['swa_epoch_start'] = self._swa_epoch_start_absolute
    
    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Load SWA state from checkpoint"""
        if 'swa_state_dict' in checkpoint:
            self.swa_model = checkpoint['swa_state_dict']
            self.swa_n = checkpoint.get('swa_n', 0)
            self._swa_epoch_start_absolute = checkpoint.get('swa_epoch_start', self.swa_epoch_start)
    
    def get_swa_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get SWA model state dict"""
        return self.swa_model
    
    def apply_swa_weights(self, pl_module: LightningModule) -> None:
        """Manually apply SWA weights to model"""
        if self.swa_model is not None:
            pl_module.load_state_dict(self.swa_model, strict=False)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get callback state"""
        return {
            'swa_model': self.swa_model,
            'swa_n': self.swa_n,
            'swa_epoch_start_absolute': self._swa_epoch_start_absolute,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state"""
        self.swa_model = state_dict.get('swa_model')
        self.swa_n = state_dict.get('swa_n', 0)
        self._swa_epoch_start_absolute = state_dict.get('swa_epoch_start_absolute')


def create_swa_callback(
    swa_lr: float = 1e-2,
    swa_epoch_start: Union[int, float] = 0.8,
    annealing_epochs: int = 10,
    **kwargs
) -> SWACallback:
    """Create SWA callback with common settings"""
    return SWACallback(
        swa_lrs=swa_lr,
        swa_epoch_start=swa_epoch_start,
        annealing_epochs=annealing_epochs,
        **kwargs
    )