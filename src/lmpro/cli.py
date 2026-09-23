# File: src/lmpro/cli.py

"""LightningCLI subclass shared by every script and the ``lmpro`` console entry point.

Design notes
------------
* ``LightningMasterCLI`` is a thin, well-behaved subclass of
  :class:`lightning.pytorch.cli.LightningCLI`.  Everything that the base class
  does not accept as a keyword (description, environment parsing) is routed to
  ``parser_kwargs`` where jsonargparse expects it.
* With ``run=True`` (the default) the CLI exposes the standard Lightning
  subcommands ``fit`` / ``validate`` / ``test`` / ``predict``.
* With ``run=False`` the CLI only parses and instantiates ``trainer``, ``model``
  and ``datamodule`` so that scripts (LR finder, batch-size scaler, export,
  evaluation) can reuse a training config without duplicating the wiring.
* ``add_arguments_to_parser`` adds experiment metadata and shows
  ``link_arguments`` in action: ``--experiment_name`` is forwarded into the
  TensorBoard logger name so the two never drift apart.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union

from jsonargparse import Namespace
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from lightning.pytorch.utilities.rank_zero import rank_zero_info

ModelType = Union[Type[LightningModule], Callable[..., LightningModule], None]
DataType = Union[Type[LightningDataModule], Callable[..., LightningDataModule], None]

DEFAULT_TRAINER_DEFAULTS: Dict[str, Any] = {
    "log_every_n_steps": 10,
    "enable_model_summary": True,
}

DEFAULT_SAVE_CONFIG_KWARGS: Dict[str, Any] = {
    "config_filename": "config.yaml",
    "overwrite": True,
}


class LightningMasterCLI(LightningCLI):
    """LightningCLI with project defaults, experiment metadata and argument linking."""

    def __init__(
        self,
        model_class: ModelType = None,
        datamodule_class: DataType = None,
        *,
        description: str = "LightningMasterPro CLI",
        env_prefix: str = "LMPRO",
        env_parse: bool = False,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = 42,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        argument_links: Optional[Dict[str, str]] = None,
        args: Optional[Union[List[str], Dict[str, Any], Namespace]] = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
        **kwargs: Any,
    ) -> None:
        # {"data.num_classes": "model.num_classes"}: after the datamodule is instantiated its
        # attribute is read and injected into the model constructor (apply_on="instantiate").
        self.argument_links: Dict[str, str] = dict(argument_links or {})
        merged_parser_kwargs: Dict[str, Any] = {
            "description": description,
            "env_prefix": env_prefix,
            "default_env": env_parse,
        }
        if parser_kwargs:
            merged_parser_kwargs.update(parser_kwargs)

        super().__init__(
            model_class=model_class,
            datamodule_class=datamodule_class,
            save_config_callback=save_config_callback,
            save_config_kwargs={**DEFAULT_SAVE_CONFIG_KWARGS, **(save_config_kwargs or {})},
            trainer_class=trainer_class,
            trainer_defaults={**DEFAULT_TRAINER_DEFAULTS, **(trainer_defaults or {})},
            seed_everything_default=seed_everything_default,
            parser_kwargs=merged_parser_kwargs,
            args=args,
            run=run,
            auto_configure_optimizers=auto_configure_optimizers,
            **kwargs,
        )

    # ------------------------------------------------------------------ parser
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Register experiment metadata and link it into the trainer config."""
        parser.add_argument(
            "--experiment_name",
            type=Optional[str],
            default=None,
            help="Experiment name. When a TensorBoardLogger is configured its `name` is set to this value.",
        )
        parser.add_argument("--tags", type=Optional[List[str]], default=None, help="Free-form tags for bookkeeping.")
        parser.add_argument("--notes", type=str, default="", help="Free-form notes stored in the saved config.")

        for source, target in self.argument_links.items():
            parser.link_arguments(source, target, apply_on="instantiate")

    # -------------------------------------------------------------- lifecycle
    def before_instantiate_classes(self) -> None:
        """Apply ``--experiment_name`` to the logger and print a short summary."""
        config = self._active_config()
        experiment_name = config.get("experiment_name")
        if experiment_name:
            self._apply_experiment_name(config, experiment_name)

        trainer_cfg = config.get("trainer")
        if trainer_cfg is not None:
            rank_zero_info(
                "Trainer config: max_epochs=%s accelerator=%s devices=%s precision=%s"
                % (
                    trainer_cfg.get("max_epochs"),
                    trainer_cfg.get("accelerator"),
                    trainer_cfg.get("devices"),
                    trainer_cfg.get("precision"),
                )
            )

    def _active_config(self) -> Namespace:
        """Return the config namespace for the running subcommand (or the flat one when run=False)."""
        if self.subcommand is not None:
            return self.config[self.subcommand]
        return self.config

    @staticmethod
    def _apply_experiment_name(config: Namespace, experiment_name: str) -> None:
        logger_cfg = config.get("trainer.logger")
        loggers = logger_cfg if isinstance(logger_cfg, list) else [logger_cfg]
        for logger in loggers:
            if isinstance(logger, Namespace) and "init_args" in logger:
                logger["init_args.name"] = experiment_name

    # ---------------------------------------------------------------- helpers
    @classmethod
    def from_config(
        cls,
        config_path: str,
        *overrides: str,
        model_class: ModelType = None,
        datamodule_class: DataType = None,
        **kwargs: Any,
    ) -> "LightningMasterCLI":
        """Instantiate trainer/model/datamodule from a YAML config without running anything.

        Example::

            cli = LightningMasterCLI.from_config("configs/vision/classifier.yaml", "--trainer.max_epochs=1")
            cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        """
        args = ["--config", str(config_path), *overrides]
        # Scripts call this after parsing their own argv; hide it so LightningCLI does not warn about
        # receiving both `args` and command-line arguments.
        saved_argv = sys.argv
        sys.argv = saved_argv[:1]
        try:
            return cls(model_class=model_class, datamodule_class=datamodule_class, args=args, run=False, **kwargs)
        finally:
            sys.argv = saved_argv


# --------------------------------------------------------------------- domain CLIs
# Each domain CLI pins the classes and links data-derived sizes into the model, so a
# user only configures the data and never repeats num_classes / vocab_size / input_dim:
#
#   vision_cli(["fit", "--data.data_config.num_classes=5", "--trainer.fast_dev_run=true"])
#
# Because the target is filled by the link, it must NOT also be set in the config.


def vision_cli(args: Optional[List[str]] = None, **kwargs: Any) -> LightningMasterCLI:
    """CLI pinned to the vision classifier; ``model.num_classes`` comes from the datamodule."""
    from .datamodules.vision_dm import VisionDataModule
    from .modules.vision.classifier import VisionClassifier

    kwargs.setdefault("argument_links", {"data.num_classes": "model.num_classes"})
    return LightningMasterCLI(
        VisionClassifier, VisionDataModule, args=args, description="LightningMasterPro Vision CLI", **kwargs
    )


def nlp_cli(args: Optional[List[str]] = None, **kwargs: Any) -> LightningMasterCLI:
    """CLI pinned to the character language model; ``model.vocab_size`` comes from the datamodule."""
    from .datamodules.nlp_dm import NLPDataModule
    from .modules.nlp.char_lm import CharacterLanguageModel

    kwargs.setdefault("argument_links", {"data.vocab_size": "model.vocab_size"})
    return LightningMasterCLI(
        CharacterLanguageModel, NLPDataModule, args=args, description="LightningMasterPro NLP CLI", **kwargs
    )


def tabular_cli(args: Optional[List[str]] = None, **kwargs: Any) -> LightningMasterCLI:
    """CLI pinned to the tabular MLP; ``model.input_dim`` comes from the datamodule."""
    from .datamodules.tabular_dm import TabularDataModule
    from .modules.tabular.mlp_reg_cls import MLPRegressorClassifier

    kwargs.setdefault("argument_links", {"data.num_features": "model.input_dim"})
    return LightningMasterCLI(
        MLPRegressorClassifier, TabularDataModule, args=args, description="LightningMasterPro Tabular CLI", **kwargs
    )


def timeseries_cli(args: Optional[List[str]] = None, **kwargs: Any) -> LightningMasterCLI:
    """CLI pinned to the time-series forecaster."""
    from .datamodules.ts_dm import TimeSeriesDataModule
    from .modules.timeseries.forecaster import TimeSeriesForecaster

    return LightningMasterCLI(
        TimeSeriesForecaster,
        TimeSeriesDataModule,
        args=args,
        description="LightningMasterPro Time Series CLI",
        **kwargs,
    )


def main(args: Optional[List[str]] = None) -> LightningMasterCLI:
    """General entry point: ``lmpro fit --config configs/vision/classifier.yaml``.

    The model and datamodule classes are read from ``class_path`` entries in the config.
    """
    return LightningMasterCLI(args=args, description="LightningMasterPro CLI")


if __name__ == "__main__":
    main()
