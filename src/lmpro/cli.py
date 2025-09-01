# File: src/lmpro/cli.py

"""
Enhanced LightningCLI with custom configurations and features
"""

import os
from typing import Any, Dict, Optional, Union
from lightning import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from pathlib import Path


class LightningMasterCLI(LightningCLI):
    """Enhanced LightningCLI with additional features and configurations"""
    
    def __init__(
        self,
        model_class: Optional[Union[type, callable]] = None,
        datamodule_class: Optional[Union[type, callable]] = None,
        save_config_callback: Optional[type] = None,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: type = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Optional[int] = None,
        description: str = "LightningMasterPro CLI",
        env_prefix: str = "LMPRO",
        env_parse: bool = True,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        subcommands: Optional[Dict[str, Dict[str, Any]]] = None,
        args: Optional[list] = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
    ):
        # Set default trainer configurations
        if trainer_defaults is None:
            trainer_defaults = {
                "max_epochs": 10,
                "enable_progress_bar": True,
                "enable_model_summary": True,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": 1,
            }
        
        # Set default save config kwargs
        if save_config_kwargs is None:
            save_config_kwargs = {
                "config_filename": "config.yaml",
                "overwrite": True,
            }
        
        super().__init__(
            model_class=model_class,
            datamodule_class=datamodule_class,
            save_config_callback=save_config_callback,
            save_config_kwargs=save_config_kwargs,
            trainer_class=trainer_class,
            trainer_defaults=trainer_defaults,
            seed_everything_default=seed_everything_default or 42,
            description=description,
            env_prefix=env_prefix,
            env_parse=env_parse,
            parser_kwargs=parser_kwargs,
            subcommands=subcommands,
            args=args,
            run=run,
            auto_configure_optimizers=auto_configure_optimizers,
        )
    
    def before_instantiate_classes(self) -> None:
        """Called before instantiating classes. Used for setup and validation."""
        super().before_instantiate_classes()
        
        # Create necessary directories
        self._setup_directories()
        
        # Log configuration info
        self._log_config_info()
    
    def _setup_directories(self) -> None:
        """Create necessary directories for logging, checkpoints, etc."""
        directories = [
            "logs",
            "models", 
            "data/synthetic",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        rank_zero_info(f"Created directories: {directories}")
    
    def _log_config_info(self) -> None:
        """Log important configuration information"""
        if hasattr(self.config, 'trainer'):
            trainer_config = self.config.trainer
            rank_zero_info(f"Training configuration:")
            rank_zero_info(f"  - Max epochs: {getattr(trainer_config, 'max_epochs', 'default')}")
            rank_zero_info(f"  - Accelerator: {getattr(trainer_config, 'accelerator', 'auto')}")
            rank_zero_info(f"  - Devices: {getattr(trainer_config, 'devices', 'auto')}")
            rank_zero_info(f"  - Precision: {getattr(trainer_config, 'precision', '32-true')}")
    
    def add_arguments_to_parser(self, parser) -> None:
        """Add custom arguments to the parser"""
        super().add_arguments_to_parser(parser)
        
        # Add custom arguments
        parser.add_argument(
            "--experiment_name",
            type=str,
            default="lmpro_experiment",
            help="Name of the experiment for logging"
        )
        
        parser.add_argument(
            "--tags",
            nargs="+",
            default=None,
            help="Tags for experiment tracking"
        )
        
        parser.add_argument(
            "--notes",
            type=str,
            default="",
            help="Notes for the experiment"
        )
    
    @staticmethod
    def configure_optimizers_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to configure optimizers from config"""
        optimizer_config = {
            "optimizer": {
                "class_path": "torch.optim.AdamW",
                "init_args": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                }
            }
        }
        
        if "lr_scheduler" in config:
            optimizer_config["lr_scheduler"] = config["lr_scheduler"]
        
        return optimizer_config


# Convenience functions for different domains
def vision_cli(args=None):
    """CLI for vision tasks"""
    from .modules.vision.classifier import VisionClassifier
    from .datamodules.vision_dm import VisionDataModule
    
    return LightningMasterCLI(
        model_class=VisionClassifier,
        datamodule_class=VisionDataModule,
        args=args,
        description="LightningMasterPro Vision CLI"
    )


def nlp_cli(args=None):
    """CLI for NLP tasks"""
    from .modules.nlp.char_lm import CharacterLanguageModel
    from .datamodules.nlp_dm import NLPDataModule
    
    return LightningMasterCLI(
        model_class=CharacterLanguageModel,
        datamodule_class=NLPDataModule,
        args=args,
        description="LightningMasterPro NLP CLI"
    )


def tabular_cli(args=None):
    """CLI for tabular tasks"""
    from .modules.tabular.mlp_reg_cls import MLPRegressorClassifier
    from .datamodules.tabular_dm import TabularDataModule
    
    return LightningMasterCLI(
        model_class=MLPRegressorClassifier,
        datamodule_class=TabularDataModule,
        args=args,
        description="LightningMasterPro Tabular CLI"
    )


def timeseries_cli(args=None):
    """CLI for time series tasks"""
    from .modules.timeseries.forecaster import TimeSeriesForecaster
    from .datamodules.ts_dm import TimeSeriesDataModule
    
    return LightningMasterCLI(
        model_class=TimeSeriesForecaster,
        datamodule_class=TimeSeriesDataModule,
        args=args,
        description="LightningMasterPro Time Series CLI"
    )


if __name__ == "__main__":
    # General CLI that can handle all domains
    cli = LightningMasterCLI(description="LightningMasterPro General CLI")