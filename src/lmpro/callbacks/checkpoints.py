# File: src/lmpro/callbacks/checkpoints.py

"""
Enhanced model checkpoint callback with additional features
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


class EnhancedModelCheckpoint(ModelCheckpoint):
    """
    Enhanced ModelCheckpoint with additional features:
    - Save model architecture
    - Save training hyperparameters  
    - Save additional metadata
    - Automatic cleanup of old checkpoints
    """
    
    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[str] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        save_architecture: bool = True,
        save_hyperparameters: bool = True,
        save_optimizer_state: bool = True,
        max_checkpoints_to_keep: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end
        )
        
        self.save_architecture = save_architecture
        self.save_hyperparameters = save_hyperparameters
        self.save_optimizer_state = save_optimizer_state
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        
        # Track saved checkpoints for cleanup
        self.saved_checkpoints = []
    
    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        """Save checkpoint with enhanced features"""
        # Call parent method
        super()._save_checkpoint(trainer, filepath)
        
        # Add to tracking list
        self.saved_checkpoints.append(filepath)
        
        # Save additional artifacts
        if self.save_architecture or self.save_hyperparameters:
            self._save_additional_artifacts(trainer, filepath)
        
        # Cleanup old checkpoints if needed
        if self.max_checkpoints_to_keep is not None:
            self._cleanup_old_checkpoints()
        
        rank_zero_info(f"Enhanced checkpoint saved: {filepath}")
    
    def _save_additional_artifacts(self, trainer: Trainer, filepath: str) -> None:
        """Save additional artifacts alongside checkpoint"""
        base_path = Path(filepath).parent
        base_name = Path(filepath).stem
        
        pl_module = trainer.lightning_module
        
        # Save model architecture
        if self.save_architecture:
            arch_path = base_path / f"{base_name}_architecture.txt"
            try:
                with open(arch_path, 'w') as f:
                    f.write(str(pl_module))
                    f.write("\n\n" + "="*50 + "\n")
                    f.write("Model Summary:\n")
                    
                    # Add parameter count
                    total_params = sum(p.numel() for p in pl_module.parameters())
                    trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
                    f.write(f"Total parameters: {total_params:,}\n")
                    f.write(f"Trainable parameters: {trainable_params:,}\n")
                    f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
                    
            except Exception as e:
                rank_zero_warn(f"Failed to save model architecture: {e}")
        
        # Save hyperparameters
        if self.save_hyperparameters:
            hparams_path = base_path / f"{base_name}_hparams.yaml"
            try:
                import yaml
                hparams = dict(pl_module.hparams)
                
                # Add training info
                hparams['training_info'] = {
                    'current_epoch': trainer.current_epoch,
                    'global_step': trainer.global_step,
                    'total_epochs': trainer.max_epochs,
                }
                
                # Add system info
                hparams['system_info'] = {
                    'torch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
                
                with open(hparams_path, 'w') as f:
                    yaml.dump(hparams, f, default_flow_style=False, indent=2)
                    
            except Exception as e:
                rank_zero_warn(f"Failed to save hyperparameters: {e}")
        
        # Save optimizer state separately if requested
        if self.save_optimizer_state and not self.save_weights_only:
            opt_path = base_path / f"{base_name}_optimizer.pth"
            try:
                optimizer_states = {}
                for i, optimizer in enumerate(trainer.optimizers):
                    optimizer_states[f'optimizer_{i}'] = optimizer.state_dict()
                
                # Save scheduler states
                scheduler_states = {}
                for i, scheduler_config in enumerate(trainer.lr_scheduler_configs):
                    scheduler_states[f'scheduler_{i}'] = scheduler_config.scheduler.state_dict()
                
                torch.save({
                    'optimizers': optimizer_states,
                    'schedulers': scheduler_states,
                    'epoch': trainer.current_epoch,
                    'global_step': trainer.global_step
                }, opt_path)
                
            except Exception as e:
                rank_zero_warn(f"Failed to save optimizer state: {e}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond the limit"""
        if len(self.saved_checkpoints) > self.max_checkpoints_to_keep:
            # Sort by modification time (oldest first)
            self.saved_checkpoints.sort(key=lambda x: os.path.getmtime(x))
            
            # Remove oldest checkpoints
            checkpoints_to_remove = self.saved_checkpoints[:-self.max_checkpoints_to_keep]
            
            for checkpoint_path in checkpoints_to_remove:
                try:
                    # Remove checkpoint and associated files
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                        rank_zero_info(f"Removed old checkpoint: {checkpoint_path}")
                    
                    # Remove associated files
                    base_path = Path(checkpoint_path).parent
                    base_name = Path(checkpoint_path).stem
                    
                    for suffix in ['_architecture.txt', '_hparams.yaml', '_optimizer.pth']:
                        assoc_file = base_path / f"{base_name}{suffix}"
                        if assoc_file.exists():
                            assoc_file.unlink()
                    
                except Exception as e:
                    rank_zero_warn(f"Failed to remove checkpoint {checkpoint_path}: {e}")
            
            # Update tracking list
            self.saved_checkpoints = self.saved_checkpoints[-self.max_checkpoints_to_keep:]
    
    def load_checkpoint_with_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint with associated metadata"""
        base_path = Path(checkpoint_path).parent  
        base_name = Path(checkpoint_path).stem
        
        # Load main checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        metadata = {}
        
        # Load architecture if available
        arch_path = base_path / f"{base_name}_architecture.txt"
        if arch_path.exists():
            with open(arch_path, 'r') as f:
                metadata['architecture'] = f.read()
        
        # Load hyperparameters if available
        hparams_path = base_path / f"{base_name}_hparams.yaml"
        if hparams_path.exists():
            try:
                import yaml
                with open(hparams_path, 'r') as f:
                    metadata['hyperparameters'] = yaml.safe_load(f)
            except Exception as e:
                rank_zero_warn(f"Failed to load hyperparameters: {e}")
        
        # Load optimizer state if available
        opt_path = base_path / f"{base_name}_optimizer.pth"
        if opt_path.exists():
            try:
                metadata['optimizer_state'] = torch.load(opt_path, map_location='cpu')
            except Exception as e:
                rank_zero_warn(f"Failed to load optimizer state: {e}")
        
        return {
            'checkpoint': checkpoint,
            'metadata': metadata
        }
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about saved checkpoints"""
        info = {
            'total_checkpoints': len(self.saved_checkpoints),
            'checkpoint_dir': self.dirpath,
            'best_model_path': self.best_model_path,
            'last_model_path': self.last_model_path,
            'monitor': self.monitor,
            'mode': self.mode,
        }
        
        if self.saved_checkpoints:
            info['oldest_checkpoint'] = min(self.saved_checkpoints, key=os.path.getmtime)
            info['newest_checkpoint'] = max(self.saved_checkpoints, key=os.path.getmtime)
        
        return info
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training starts"""
        super().on_train_start(trainer, pl_module)
        
        # Create checkpoint directory if it doesn't exist
        if self.dirpath:
            Path(self.dirpath).mkdir(parents=True, exist_ok=True)
            
            # Save initial model info
            info_path = Path(self.dirpath) / "checkpoint_info.txt"
            with open(info_path, 'w') as f:
                f.write("Enhanced Model Checkpoint Info\n")
                f.write("="*40 + "\n\n")
                f.write(f"Monitor: {self.monitor}\n")
                f.write(f"Mode: {self.mode}\n")
                f.write(f"Save top k: {self.save_top_k}\n")
                f.write(f"Save last: {self.save_last}\n")
                f.write(f"Save weights only: {self.save_weights_only}\n")
                f.write(f"Max checkpoints to keep: {self.max_checkpoints_to_keep}\n")
                f.write(f"Save architecture: {self.save_architecture}\n")
                f.write(f"Save hyperparameters: {self.save_hyperparameters}\n")
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training ends"""
        super().on_train_end(trainer, pl_module)
        
        # Update checkpoint info file
        if self.dirpath:
            info = self.get_checkpoint_info()
            info_path = Path(self.dirpath) / "final_checkpoint_info.yaml"
            
            try:
                import yaml
                with open(info_path, 'w') as f:
                    yaml.dump(info, f, default_flow_style=False, indent=2)
            except Exception as e:
                rank_zero_warn(f"Failed to save final checkpoint info: {e}")
        
        rank_zero_info(f"Training completed. Best checkpoint: {self.best_model_path}")


# Convenience functions for common checkpoint configurations
def get_best_checkpoint_callback(
    monitor: str = "val/loss",
    mode: str = "min",
    save_top_k: int = 1,
    dirpath: str = "checkpoints/best",
    **kwargs
) -> EnhancedModelCheckpoint:
    """Get checkpoint callback for saving best model"""
    return EnhancedModelCheckpoint(
        dirpath=dirpath,
        filename="best-{epoch:02d}-{" + monitor.replace('/', '-') + ":.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=False,
        **kwargs
    )


def get_periodic_checkpoint_callback(
    every_n_epochs: int = 5,
    save_top_k: int = 3,
    dirpath: str = "checkpoints/periodic",
    **kwargs
) -> EnhancedModelCheckpoint:
    """Get checkpoint callback for periodic saving"""
    return EnhancedModelCheckpoint(
        dirpath=dirpath,
        filename="epoch-{epoch:02d}",
        every_n_epochs=every_n_epochs,
        save_top_k=save_top_k,
        save_last=True,
        **kwargs
    )


def get_last_checkpoint_callback(
    dirpath: str = "checkpoints/last",
    **kwargs
) -> EnhancedModelCheckpoint:
    """Get checkpoint callback for saving last model"""
    return EnhancedModelCheckpoint(
        dirpath=dirpath,
        filename="last",
        save_last=True,
        save_top_k=0,
        **kwargs
    )