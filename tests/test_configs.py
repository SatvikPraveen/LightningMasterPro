# tests/test_configs.py
"""Every YAML config must resolve to real classes and match their constructor signatures exactly."""

import importlib
import inspect
from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).parent.parent / "configs"
TRAINING_CONFIGS = sorted(
    p for p in CONFIG_DIR.rglob("*.yaml") if p.parent.name not in ("tuning",) and p.name != "defaults.yaml"
)
TUNING_CONFIGS = sorted((CONFIG_DIR / "tuning").glob("*.yaml"))


def load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def check_init_args(cls, init_args: dict, where: str) -> None:
    sig = inspect.signature(cls.__init__)
    params = {name: p for name, p in sig.parameters.items() if name != "self"}
    if cls.__module__.startswith("lmpro."):
        # Our own classes must fail loudly on a typo instead of swallowing it.
        assert not any(p.kind is p.VAR_KEYWORD for p in params.values()), f"{cls.__name__} must not accept **kwargs"

    unknown = set(init_args) - set(params)
    assert not unknown, f"{where}: unknown init_args for {cls.__name__}: {sorted(unknown)}"

    missing = [
        name
        for name, p in params.items()
        if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and name not in init_args
    ]
    assert not missing, f"{where}: missing required init_args for {cls.__name__}: {missing}"


@pytest.mark.parametrize("path", TRAINING_CONFIGS, ids=[str(p.relative_to(CONFIG_DIR)) for p in TRAINING_CONFIGS])
def test_training_config_matches_class_signatures(path):
    cfg = load(path)
    for section in ("model", "data"):
        assert "class_path" in cfg[section], f"{path.name}: {section} needs a class_path"
        cls = resolve(cfg[section]["class_path"])
        check_init_args(cls, cfg[section].get("init_args", {}), f"{path.name}:{section}")

    for callback in cfg["trainer"].get("callbacks", []):
        cls = resolve(callback["class_path"])
        check_init_args(cls, callback.get("init_args", {}), f"{path.name}:callbacks")

    logger = cfg["trainer"].get("logger")
    if isinstance(logger, dict):
        check_init_args(resolve(logger["class_path"]), logger.get("init_args", {}), f"{path.name}:logger")


@pytest.mark.parametrize("path", TRAINING_CONFIGS, ids=[str(p.relative_to(CONFIG_DIR)) for p in TRAINING_CONFIGS])
def test_training_config_uses_jsonargparse_layout(path):
    cfg = load(path)
    assert set(cfg) <= {
        "seed_everything",
        "model",
        "data",
        "trainer",
        "experiment_name",
        "tags",
        "notes",
    }, f"{path.name}: top-level keys must be LightningCLI keys, got {sorted(cfg)}"
    assert "defaults" not in cfg, "Hydra-style `defaults:` is not supported by LightningCLI"
    for callback in cfg["trainer"].get("callbacks", []):
        if callback["class_path"].endswith("ModelCheckpoint"):
            filename = callback["init_args"].get("filename", "")
            assert "/" not in filename, "a '/' in ModelCheckpoint.filename creates sub-directories"


@pytest.mark.parametrize("path", TRAINING_CONFIGS, ids=[str(p.relative_to(CONFIG_DIR)) for p in TRAINING_CONFIGS])
def test_config_trains_and_logs_every_monitored_metric(path, tmp_path, monkeypatch):
    """Each config must complete a fast_dev_run fit, and every `monitor` key must really be logged."""
    from lmpro.cli import main

    monkeypatch.chdir(tmp_path)  # relative dirpath/save_dir in the configs must not land in the repo
    cfg = load(path)
    monitors = {
        cb["init_args"]["monitor"] for cb in cfg["trainer"].get("callbacks", []) if "monitor" in cb.get("init_args", {})
    }

    cli = main(
        [
            "fit",
            "--config",
            str(path),
            "--data.init_args.num_workers=0",
            "--data.init_args.persistent_workers=false",
            "--trainer.accelerator=cpu",
            "--trainer.devices=1",
            "--trainer.fast_dev_run=true",
            "--trainer.default_root_dir",
            str(tmp_path),
        ]
    )
    logged = set(cli.trainer.callback_metrics)
    assert cli.trainer.state.finished
    assert monitors <= logged, f"{path.name} monitors {sorted(monitors - logged)} which the module never logs"


def test_defaults_config_is_a_valid_layer():
    cfg = load(CONFIG_DIR / "defaults.yaml")
    assert set(cfg) <= {"seed_everything", "trainer"}
    for callback in cfg["trainer"].get("callbacks", []):
        check_init_args(resolve(callback["class_path"]), callback.get("init_args", {}), "defaults.yaml:callbacks")


def test_tuning_configs_match_tuner_signatures():
    from lightning.pytorch.tuner import Tuner

    lr = load(CONFIG_DIR / "tuning" / "lr_finder.yaml")["lr_finder"]
    assert set(lr) <= set(inspect.signature(Tuner.lr_find).parameters)
    assert lr["min_lr"] < lr["max_lr"]

    scaler = load(CONFIG_DIR / "tuning" / "batch_scaler.yaml")["batch_scaler"]
    assert set(scaler) <= set(inspect.signature(Tuner.scale_batch_size).parameters)
    assert scaler["mode"] in ("power", "binsearch")


def test_ablation_config_targets_real_keys():
    ablation = load(CONFIG_DIR / "tuning" / "ablation_study.yaml")["ablation"]
    base = load(CONFIG_DIR / "vision" / "classifier.yaml")
    for dotted, values in ablation["parameters"].items():
        node = base
        for key in dotted.split("."):
            assert key in node, f"{dotted} does not exist in classifier.yaml"
            node = node[key]
        assert isinstance(values, list) and values
    for key in ablation.get("trainer_overrides", {}):
        assert key in inspect.signature(resolve("lightning.pytorch.Trainer").__init__).parameters
