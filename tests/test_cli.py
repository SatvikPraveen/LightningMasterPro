# tests/test_cli.py
"""Behavioural tests for LightningMasterCLI: real parsing, instantiation, linking and a fast_dev_run fit."""

from pathlib import Path

import pytest
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI

from lmpro.cli import LightningMasterCLI, main, nlp_cli, tabular_cli, timeseries_cli, vision_cli

ROOT = Path(__file__).parent.parent
CONFIGS = ROOT / "configs"

# Overrides that make every config cheap and side-effect free on CI.
CPU_OVERRIDES = [
    "--data.init_args.num_workers=0",
    "--data.init_args.persistent_workers=false",
    "--trainer.logger=false",
    "--trainer.accelerator=cpu",
    "--trainer.devices=1",
]


def test_subclass_of_lightning_cli():
    assert issubclass(LightningMasterCLI, LightningCLI)


def test_from_config_instantiates_classes_without_running():
    cli = LightningMasterCLI.from_config(CONFIGS / "vision" / "classifier.yaml", *CPU_OVERRIDES)
    assert isinstance(cli.model, LightningModule)
    assert isinstance(cli.datamodule, LightningDataModule)
    assert isinstance(cli.trainer, Trainer)
    assert cli.model.hparams.num_classes == 10
    assert cli.datamodule.num_workers == 0


def test_cli_overrides_win_over_config():
    cli = LightningMasterCLI.from_config(
        CONFIGS / "tabular" / "mlp.yaml", *CPU_OVERRIDES, "--model.init_args.learning_rate=0.123"
    )
    assert cli.model.hparams.learning_rate == pytest.approx(0.123)


def test_experiment_name_is_linked_into_logger_name():
    cli = LightningMasterCLI.from_config(
        CONFIGS / "vision" / "classifier.yaml",
        "--data.init_args.num_workers=0",
        "--data.init_args.persistent_workers=false",
        "--experiment_name=my_exp",
    )
    assert cli.trainer.logger.name == "my_exp"


def test_fit_fast_dev_run_through_subcommand(tmp_path, monkeypatch):
    # fast_dev_run replaces the configured logger with a DummyLogger, so LearningRateMonitor still works.
    monkeypatch.chdir(tmp_path)  # the config's relative checkpoint dirpath must not land in the repo
    cli = main(
        [
            "fit",
            "--config",
            str(CONFIGS / "vision" / "classifier.yaml"),
            "--data.init_args.num_workers=0",
            "--data.init_args.persistent_workers=false",
            "--trainer.accelerator=cpu",
            "--trainer.devices=1",
            "--trainer.fast_dev_run=true",
            "--trainer.default_root_dir",
            str(tmp_path),
        ]
    )
    assert cli.trainer.state.finished
    assert "val/loss" in cli.trainer.callback_metrics


def test_unknown_init_arg_is_rejected():
    with pytest.raises(SystemExit):
        LightningMasterCLI.from_config(
            CONFIGS / "vision" / "classifier.yaml", *CPU_OVERRIDES, "--model.init_args.backbone=resnet18"
        )


@pytest.mark.parametrize(
    ("domain_cli", "data_args", "expected_attr", "source_attr"),
    [
        (vision_cli, ["--data.data_config.num_classes=5"], "num_classes", "num_classes"),
        (nlp_cli, ["--data.task=language_modeling"], "vocab_size", "vocab_size"),
        (tabular_cli, ["--data.data_config.num_features=7", "--model.output_dim=3"], "input_dim", "num_features"),
    ],
)
def test_domain_cli_links_data_to_model(domain_cli, data_args, expected_attr, source_attr):
    """link_arguments(apply_on="instantiate") fills the model size from the instantiated datamodule."""
    cli = domain_cli(
        [
            *data_args,
            "--data.num_workers=0",
            "--data.persistent_workers=false",
            "--trainer.logger=false",
            "--trainer.accelerator=cpu",
        ],
        run=False,
    )
    assert getattr(cli.model.hparams, expected_attr) == getattr(cli.datamodule, source_attr)


def test_timeseries_cli_instantiates():
    cli = timeseries_cli(
        [
            "--model.input_dim=1",
            "--model.output_dim=1",
            "--model.sequence_length=100",
            "--model.prediction_horizon=10",
            "--data.num_workers=0",
            "--data.persistent_workers=false",
            "--trainer.logger=false",
        ],
        run=False,
    )
    assert isinstance(cli.model, LightningModule)
