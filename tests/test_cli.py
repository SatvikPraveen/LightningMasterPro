# tests/test_cli.py
"""Integration tests for CLI module."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─── Import Tests ─────────────────────────────────────────────────────────────

class TestCLIImports:
    def test_import_lightning_master_cli(self):
        from lmpro.cli import LightningMasterCLI
        assert LightningMasterCLI is not None

    def test_import_main(self):
        from lmpro.cli import main
        assert callable(main)

    def test_import_vision_cli(self):
        from lmpro.cli import vision_cli
        assert callable(vision_cli)

    def test_import_nlp_cli(self):
        from lmpro.cli import nlp_cli
        assert callable(nlp_cli)

    def test_import_tabular_cli(self):
        from lmpro.cli import tabular_cli
        assert callable(tabular_cli)

    def test_import_timeseries_cli(self):
        from lmpro.cli import timeseries_cli
        assert callable(timeseries_cli)


# ─── LightningMasterCLI Class ─────────────────────────────────────────────────

class TestLightningMasterCLI:
    def test_is_subclass_of_lightningcli(self):
        from lmpro.cli import LightningMasterCLI
        from lightning.pytorch.cli import LightningCLI
        assert issubclass(LightningMasterCLI, LightningCLI)

    def test_has_configure_optimizers_helper(self):
        from lmpro.cli import LightningMasterCLI
        assert hasattr(LightningMasterCLI, "configure_optimizers_from_config")
        assert callable(LightningMasterCLI.configure_optimizers_from_config)

    def test_configure_optimizers_returns_dict(self):
        from lmpro.cli import LightningMasterCLI
        result = LightningMasterCLI.configure_optimizers_from_config({})
        assert isinstance(result, dict)
        assert "optimizer" in result

    def test_configure_optimizers_with_scheduler(self):
        from lmpro.cli import LightningMasterCLI
        config = {"lr_scheduler": {"class_path": "torch.optim.lr_scheduler.StepLR"}}
        result = LightningMasterCLI.configure_optimizers_from_config(config)
        assert "lr_scheduler" in result


# ─── CLI Initialization Defaults ─────────────────────────────────────────────

class TestCLIDefaults:
    """Test CLI class attributes and statics without actually running training."""

    def test_description_default(self):
        """Check the description kwarg is accepted by the constructor signature."""
        import inspect
        from lmpro.cli import LightningMasterCLI
        sig = inspect.signature(LightningMasterCLI.__init__)
        assert "description" in sig.parameters

    def test_env_prefix_default(self):
        import inspect
        from lmpro.cli import LightningMasterCLI
        sig = inspect.signature(LightningMasterCLI.__init__)
        assert sig.parameters["env_prefix"].default == "LMPRO"

    def test_auto_configure_optimizers_default(self):
        import inspect
        from lmpro.cli import LightningMasterCLI
        sig = inspect.signature(LightningMasterCLI.__init__)
        assert sig.parameters["auto_configure_optimizers"].default is True

    def test_run_default_true(self):
        import inspect
        from lmpro.cli import LightningMasterCLI
        sig = inspect.signature(LightningMasterCLI.__init__)
        assert sig.parameters["run"].default is True


# ─── Domain CLI Function Signatures ──────────────────────────────────────────

class TestDomainCLISignatures:
    def test_vision_cli_accepts_args_kwarg(self):
        import inspect
        from lmpro.cli import vision_cli
        sig = inspect.signature(vision_cli)
        assert "args" in sig.parameters

    def test_nlp_cli_accepts_args_kwarg(self):
        import inspect
        from lmpro.cli import nlp_cli
        sig = inspect.signature(nlp_cli)
        assert "args" in sig.parameters

    def test_tabular_cli_accepts_args_kwarg(self):
        import inspect
        from lmpro.cli import tabular_cli
        sig = inspect.signature(tabular_cli)
        assert "args" in sig.parameters

    def test_timeseries_cli_accepts_args_kwarg(self):
        import inspect
        from lmpro.cli import timeseries_cli
        sig = inspect.signature(timeseries_cli)
        assert "args" in sig.parameters
