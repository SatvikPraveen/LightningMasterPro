# File: src/lmpro/loops/curriculum_loop.py

"""
Curriculum Learning Loop for progressive training difficulty
"""

import torch
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.loops import Loop
from lightning import LightningModule, Trainer
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import numpy as np
from abc import ABC, abstractmethod
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import math


class CurriculumStrategy(ABC):
    """Abstract base class for curriculum learning strategies"""
    
    @abstractmethod
    def get_difficulty_scores(self, dataset, model: Optional[LightningModule] = None) -> np.ndarray:
        """Get difficulty scores for each sample in the dataset"""
        pass
    
    @abstractmethod
    def get_curriculum_schedule(self, total_epochs: int, dataset_size: int) -> List[Tuple[int, float]]:
        """Get curriculum schedule: list of (epoch, difficulty_threshold) pairs"""
        pass


class LengthBasedCurriculum(CurriculumStrategy):
    """Curriculum based on input sequence/sample length"""
    
    def __init__(self, reverse: bool = False):
        self.reverse = reverse  # If True, start with longer samples
    
    def get_difficulty_scores(self, dataset, model: Optional[LightningModule] = None) -> np.ndarray:
        """Get difficulty scores based on input length"""
        lengths = []
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                if isinstance(sample, tuple):
                    input_data = sample[0]
                else:
                    input_data = sample
                
                if isinstance(input_data, torch.Tensor):
                    if input_data.dim() == 1:  # Sequence length
                        length = input_data.shape[0]
                    elif input_data.dim() == 2:  # Image or 2D data
                        length = input_data.shape[0] * input_data.shape[1]
                    else:
                        length = input_data.numel()
                else:
                    length = len(input_data) if hasattr(input_data, '__len__') else 1
                
                lengths.append(length)
                
            except Exception:
                lengths.append(0)  # Default for problematic samples
        
        lengths = np.array(lengths)
        
        # Normalize to [0, 1] range
        if lengths.max() > lengths.min():
            normalized_lengths = (lengths - lengths.min()) / (lengths.max() - lengths.min())
        else:
            normalized_lengths = np.ones_like(lengths)
        
        return 1 - normalized_lengths if self.reverse else normalized_lengths
    
    def get_curriculum_schedule(self, total_epochs: int, dataset_size: int) -> List[Tuple[int, float]]:
        """Linear curriculum schedule"""
        schedule = []
        for epoch in range(total_epochs):
            # Linear progression from 0.1 to 1.0
            threshold = 0.1 + (0.9 * epoch / max(1, total_epochs - 1))
            schedule.append((epoch, threshold))
        
        return schedule


class LossBasedCurriculum(CurriculumStrategy):
    """Curriculum based on sample loss/difficulty from previous training"""
    
    def __init__(self, warmup_epochs: int = 5):
        self.warmup_epochs = warmup_epochs
        self.sample_losses = None
    
    def get_difficulty_scores(self, dataset, model: Optional[LightningModule] = None) -> np.ndarray:
        """Get difficulty scores based on model loss"""
        if self.sample_losses is not None:
            return self.sample_losses
        
        if model is None:
            # Fallback to random scoring for first iteration
            return np.random.random(len(dataset))
        
        # Compute loss for each sample
        model.eval()
        losses = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                try:
                    sample = dataset[i]
                    if isinstance(sample, tuple):
                        x, y = sample
                    else:
                        x, y = sample, None
                    
                    # Add batch dimension
                    if isinstance(x, torch.Tensor):
                        x = x.unsqueeze(0).to(model.device)
                    if y is not None and isinstance(y, torch.Tensor):
                        y = y.unsqueeze(0).to(model.device)
                    
                    # Compute loss
                    if hasattr(model, 'training_step'):
                        if y is not None:
                            loss = model.training_step((x, y), 0)
                        else:
                            loss = model.training_step(x, 0)
                    else:
                        # Simple forward pass
                        logits = model(x)
                        if y is not None and hasattr(model, 'criterion'):
                            loss = model.criterion(logits, y)
                        else:
                            loss = torch.tensor(0.0)
                    
                    losses.append(loss.item())
                    
                except Exception as e:
                    losses.append(1.0)  # High loss for problematic samples
        
        losses = np.array(losses)
        
        # Normalize to [0, 1] range
        if losses.max() > losses.min():
            normalized_losses = (losses - losses.min()) / (losses.max() - losses.min())
        else:
            normalized_losses = np.ones_like(losses) * 0.5
        
        self.sample_losses = normalized_losses
        return normalized_losses
    
    def get_curriculum_schedule(self, total_epochs: int, dataset_size: int) -> List[Tuple[int, float]]:
        """Exponential curriculum schedule"""
        schedule = []
        for epoch in range(total_epochs):
            if epoch < self.warmup_epochs:
                # Start with easier samples
                threshold = 0.3
            else:
                # Exponential progression
                progress = (epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs - 1)
                threshold = 0.3 + 0.7 * (1 - math.exp(-3 * progress))
            
            schedule.append((epoch, threshold))
        
        return schedule


class RandomCurriculum(CurriculumStrategy):
    """Random curriculum (baseline) - gradually increase dataset size"""
    
    def get_difficulty_scores(self, dataset, model: Optional[LightningModule] = None) -> np.ndarray:
        """Random difficulty scores"""
        return np.random.random(len(dataset))
    
    def get_curriculum_schedule(self, total_epochs: int, dataset_size: int) -> List[Tuple[int, float]]:
        """Random curriculum with increasing dataset size"""
        schedule = []
        for epoch in range(total_epochs):
            # Start with 20% of data, gradually increase to 100%
            threshold = 0.2 + 0.8 * epoch / max(1, total_epochs - 1)
            schedule.append((epoch, threshold))
        
        return schedule


class CurriculumLoop(Loop):
    """
    Curriculum Learning Loop
    
    Implements curriculum learning by progressively introducing
    more difficult training samples based on various strategies.
    """
    
    def __init__(
        self,
        strategy: Union[str, CurriculumStrategy] = "length",
        update_frequency: int = 5,  # Update curriculum every N epochs
        min_samples_per_epoch: int = None,  # Minimum samples per epoch
        recompute_difficulty: bool = True,  # Recompute difficulty scores
        curriculum_warmup: int = 0,  # Epochs before curriculum starts
        **kwargs
    ):
        super().__init__()
        
        # Initialize strategy
        if isinstance(strategy, str):
            if strategy == "length":
                self.strategy = LengthBasedCurriculum()
            elif strategy == "loss":
                self.strategy = LossBasedCurriculum()
            elif strategy == "random":
                self.strategy = RandomCurriculum()
            else:
                raise ValueError(f"Unknown curriculum strategy: {strategy}")
        else:
            self.strategy = strategy
        
        self.update_frequency = update_frequency
        self.min_samples_per_epoch = min_samples_per_epoch
        self.recompute_difficulty = recompute_difficulty
        self.curriculum_warmup = curriculum_warmup
        
        # Internal state
        self.original_train_dataloader = None
        self.original_dataset = None
        self.difficulty_scores = None
        self.curriculum_schedule = None
        self.current_epoch = 0
        self.current_subset_indices = None
        
        # Tracking
        self.curriculum_stats = []
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup curriculum learning"""
        super().setup(trainer, pl_module, stage)
        
        # Store original dataloader and dataset
        self.original_train_dataloader = trainer.datamodule.train_dataloader()
        self.original_dataset = self.original_train_dataloader.dataset
        
        # Compute initial difficulty scores
        rank_zero_info("Computing initial difficulty scores...")
        self.difficulty_scores = self.strategy.get_difficulty_scores(self.original_dataset)
        
        # Create curriculum schedule
        total_epochs = trainer.max_epochs
        dataset_size = len(self.original_dataset)
        self.curriculum_schedule = self.strategy.get_curriculum_schedule(total_epochs, dataset_size)
        
        # Set minimum samples per epoch if not specified
        if self.min_samples_per_epoch is None:
            self.min_samples_per_epoch = max(32, dataset_size // 20)  # At least 5% of data
        
        rank_zero_info(f"Curriculum learning setup: {type(self.strategy).__name__}")
        rank_zero_info(f"Dataset size: {dataset_size}, Min samples per epoch: {self.min_samples_per_epoch}")
    
    @property
    def done(self) -> bool:
        """Curriculum learning continues throughout training"""
        return False  # Let the parent fit loop handle completion
    
    def reset(self) -> None:
        """Reset loop state"""
        self.current_epoch = 0
        self.curriculum_stats = []
    
    def advance(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Update curriculum for current epoch"""
        # Skip curriculum during warmup
        if self.current_epoch < self.curriculum_warmup:
            rank_zero_info(f"Epoch {self.current_epoch + 1}: Curriculum warmup - using full dataset")
            self.current_epoch += 1
            return
        
        # Recompute difficulty scores periodically
        if (self.recompute_difficulty and 
            self.current_epoch > 0 and 
            self.current_epoch % self.update_frequency == 0):
            rank_zero_info("Recomputing difficulty scores...")
            self.difficulty_scores = self.strategy.get_difficulty_scores(self.original_dataset, pl_module)
        
        # Get current difficulty threshold from schedule
        curriculum_epoch = self.current_epoch - self.curriculum_warmup
        if curriculum_epoch < len(self.curriculum_schedule):
            _, difficulty_threshold = self.curriculum_schedule[curriculum_epoch]
        else:
            # Use full dataset after curriculum completion
            difficulty_threshold = 1.0
        
        # Select samples based on difficulty threshold
        selected_indices = self._select_curriculum_samples(difficulty_threshold)
        
        # Create curriculum dataloader
        curriculum_dataloader = self._create_curriculum_dataloader(selected_indices)
        
        # Replace trainer's train dataloader
        trainer.datamodule.train_dataloader = lambda: curriculum_dataloader
        
        # Log curriculum statistics
        self._log_curriculum_stats(difficulty_threshold, len(selected_indices))
        
        self.current_epoch += 1
    
    def _select_curriculum_samples(self, difficulty_threshold: float) -> List[int]:
        """Select samples based on difficulty threshold"""
        # Include samples with difficulty <= threshold
        candidate_indices = np.where(self.difficulty_scores <= difficulty_threshold)[0]
        
        # Ensure minimum number of samples
        if len(candidate_indices) < self.min_samples_per_epoch:
            # Add easiest samples to meet minimum requirement
            all_indices = np.argsort(self.difficulty_scores)
            candidate_indices = all_indices[:self.min_samples_per_epoch]
        
        # Convert to list and shuffle
        selected_indices = list(candidate_indices)
        np.random.shuffle(selected_indices)
        
        self.current_subset_indices = selected_indices
        return selected_indices
    
    def _create_curriculum_dataloader(self, selected_indices: List[int]) -> DataLoader:
        """Create curriculum dataloader with selected samples"""
        # Create subset dataset
        curriculum_dataset = Subset(self.original_dataset, selected_indices)
        
        # Create dataloader with same parameters as original
        curriculum_dataloader = DataLoader(
            curriculum_dataset,
            batch_size=self.original_train_dataloader.batch_size,
            shuffle=True,
            num_workers=self.original_train_dataloader.num_workers,
            pin_memory=getattr(self.original_train_dataloader, 'pin_memory', False),
            persistent_workers=getattr(self.original_train_dataloader, 'persistent_workers', False),
            drop_last=getattr(self.original_train_dataloader, 'drop_last', False),
        )
        
        return curriculum_dataloader
    
    def _log_curriculum_stats(self, difficulty_threshold: float, num_samples: int) -> None:
        """Log curriculum statistics"""
        total_samples = len(self.original_dataset)
        percentage = (num_samples / total_samples) * 100
        
        stats = {
            'epoch': self.current_epoch + 1,
            'difficulty_threshold': difficulty_threshold,
            'num_samples': num_samples,
            'total_samples': total_samples,
            'percentage': percentage
        }
        
        self.curriculum_stats.append(stats)
        
        rank_zero_info(
            f"Epoch {self.current_epoch + 1}: Curriculum - "
            f"threshold={difficulty_threshold:.3f}, "
            f"samples={num_samples}/{total_samples} ({percentage:.1f}%)"
        )
    
    def on_run_end(self, trainer: Trainer, pl_module: LightningModule) -> Any:
        """Called when curriculum loop completes"""
        # Restore original dataloader
        trainer.datamodule.train_dataloader = lambda: self.original_train_dataloader
        
        # Log final curriculum statistics
        self._log_final_stats()
        
        return self.curriculum_stats
    
    def _log_final_stats(self) -> None:
        """Log final curriculum statistics"""
        if not self.curriculum_stats:
            return
        
        rank_zero_info("\n" + "="*50)
        rank_zero_info("CURRICULUM LEARNING SUMMARY")
        rank_zero_info("="*50)
        
        total_epochs = len(self.curriculum_stats)
        avg_samples = np.mean([s['num_samples'] for s in self.curriculum_stats])
        total_samples = self.curriculum_stats[0]['total_samples'] if self.curriculum_stats else 0
        
        rank_zero_info(f"Strategy: {type(self.strategy).__name__}")
        rank_zero_info(f"Total epochs with curriculum: {total_epochs}")
        rank_zero_info(f"Average samples per epoch: {avg_samples:.1f}/{total_samples} ({avg_samples/total_samples*100:.1f}%)")
        
        # Show progression
        if total_epochs >= 5:
            epochs_to_show = [0, total_epochs//4, total_epochs//2, 3*total_epochs//4, total_epochs-1]
            rank_zero_info("\nCurriculum progression:")
            for i in epochs_to_show:
                stats = self.curriculum_stats[i]
                rank_zero_info(
                    f"  Epoch {stats['epoch']:3d}: "
                    f"{stats['num_samples']:5d} samples "
                    f"({stats['percentage']:5.1f}%) "
                    f"threshold={stats['difficulty_threshold']:.3f}"
                )
        
        rank_zero_info("="*50 + "\n")
    
    def get_curriculum_stats(self) -> List[Dict[str, Any]]:
        """Get curriculum statistics"""
        return self.curriculum_stats
    
    def get_current_difficulty_scores(self) -> Optional[np.ndarray]:
        """Get current difficulty scores"""
        return self.difficulty_scores
    
    def plot_curriculum_progression(self, save_path: Optional[str] = None) -> None:
        """Plot curriculum progression"""
        if not self.curriculum_stats:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            epochs = [s['epoch'] for s in self.curriculum_stats]
            percentages = [s['percentage'] for s in self.curriculum_stats]
            thresholds = [s['difficulty_threshold'] for s in self.curriculum_stats]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot sample percentage
            ax1.plot(epochs, percentages, 'b-', linewidth=2, marker='o', markersize=3)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Training Samples (%)')
            ax1.set_title('Curriculum Learning: Sample Progression')
            ax1.grid(True, alpha=0.3)
            
            # Plot difficulty threshold
            ax2.plot(epochs, thresholds, 'r-', linewidth=2, marker='s', markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Difficulty Threshold')
            ax2.set_title('Curriculum Learning: Difficulty Threshold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                rank_zero_info(f"Curriculum progression plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            rank_zero_info("Matplotlib not available for plotting")


# Convenience functions
def create_length_curriculum_loop(reverse: bool = False, **kwargs) -> CurriculumLoop:
    """Create curriculum loop based on input length"""
    strategy = LengthBasedCurriculum(reverse=reverse)
    return CurriculumLoop(strategy=strategy, **kwargs)


def create_loss_curriculum_loop(warmup_epochs: int = 5, **kwargs) -> CurriculumLoop:
    """Create curriculum loop based on sample loss"""
    strategy = LossBasedCurriculum(warmup_epochs=warmup_epochs)
    return CurriculumLoop(strategy=strategy, **kwargs)


def create_random_curriculum_loop(**kwargs) -> CurriculumLoop:
    """Create random curriculum loop (baseline)"""
    strategy = RandomCurriculum()
    return CurriculumLoop(strategy=strategy, **kwargs)