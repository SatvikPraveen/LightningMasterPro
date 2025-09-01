# tests/test_configs.py
"""Tests for validating YAML configuration files."""

import pytest
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Get config directory
CONFIG_DIR = Path(__file__).parent.parent / "configs"


def test_config_directory_exists():
    """Test that config directory exists."""
    assert CONFIG_DIR.exists(), f"Config directory not found: {CONFIG_DIR}"


def test_defaults_config_valid():
    """Test that defaults.yaml is valid."""
    defaults_path = CONFIG_DIR / "defaults.yaml"
    assert defaults_path.exists(), "defaults.yaml not found"
    
    with open(defaults_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    assert 'seed_everything' in config
    assert 'trainer' in config
    assert 'model' in config
    assert 'data' in config
    assert 'callbacks' in config
    assert 'logger' in config
    
    # Validate trainer config
    trainer_config = config['trainer']
    assert 'max_epochs' in trainer_config
    assert 'accelerator' in trainer_config
    assert 'devices' in trainer_config
    assert isinstance(trainer_config['max_epochs'], int)
    assert trainer_config['max_epochs'] > 0


def test_vision_configs_valid():
    """Test that vision configs are valid."""
    vision_dir = CONFIG_DIR / "vision"
    assert vision_dir.exists(), "vision config directory not found"
    
    # Test classifier config
    classifier_path = vision_dir / "classifier.yaml"
    assert classifier_path.exists(), "classifier.yaml not found"
    
    with open(classifier_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'model' in config
    assert 'data' in config
    assert 'trainer' in config
    
    model_config = config['model']
    assert 'class_path' in model_config
    assert 'lmpro.modules.vision.classifier.VisionClassifier' in model_config['class_path']
    
    # Test segmenter config
    segmenter_path = vision_dir / "segmenter.yaml"
    assert segmenter_path.exists(), "segmenter.yaml not found"
    
    with open(segmenter_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'model' in config
    model_config = config['model']
    assert 'lmpro.modules.vision.segmenter.VisionSegmenter' in model_config['class_path']


def test_nlp_configs_valid():
    """Test that NLP configs are valid."""
    nlp_dir = CONFIG_DIR / "nlp"
    assert nlp_dir.exists(), "nlp config directory not found"
    
    # Test char_lm config
    char_lm_path = nlp_dir / "char_lm.yaml"
    assert char_lm_path.exists(), "char_lm.yaml not found"
    
    with open(char_lm_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'model' in config
    model_config = config['model']
    assert 'lmpro.modules.nlp.char_lm.CharacterLM' in model_config['class_path']
    
    # Test sentiment config
    sentiment_path = nlp_dir / "sentiment.yaml"
    assert sentiment_path.exists(), "sentiment.yaml not found"
    
    with open(sentiment_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'model' in config
    model_config = config['model']
    assert 'lmpro.modules.nlp.sentiment.SentimentClassifier' in model_config['class_path']


def test_tabular_config_valid():
    """Test that tabular config is valid."""
    tabular_dir = CONFIG_DIR / "tabular"
    assert tabular_dir.exists(), "tabular config directory not found"
    
    mlp_path = tabular_dir / "mlp.yaml"
    assert mlp_path.exists(), "mlp.yaml not found"
    
    with open(mlp_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'model' in config
    model_config = config['model']
    assert 'lmpro.modules.tabular.mlp_reg_cls.MLPRegCls' in model_config['class_path']


def test_timeseries_config_valid():
    """Test that timeseries config is valid."""
    ts_dir = CONFIG_DIR / "timeseries"
    assert ts_dir.exists(), "timeseries config directory not found"
    
    forecaster_path = ts_dir / "forecaster.yaml"
    assert forecaster_path.exists(), "forecaster.yaml not found"
    
    with open(forecaster_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'model' in config
    model_config = config['model']
    assert 'lmpro.modules.timeseries.forecaster.TimeSeriesForecaster' in model_config['class_path']


def test_tuning_configs_valid():
    """Test that tuning configs are valid."""
    tuning_dir = CONFIG_DIR / "tuning"
    assert tuning_dir.exists(), "tuning config directory not found"
    
    # Test lr_finder config
    lr_finder_path = tuning_dir / "lr_finder.yaml"
    assert lr_finder_path.exists(), "lr_finder.yaml not found"
    
    with open(lr_finder_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'lr_finder' in config
    lr_config = config['lr_finder']
    assert 'min_lr' in lr_config
    assert 'max_lr' in lr_config
    assert isinstance(lr_config['min_lr'], float)
    assert isinstance(lr_config['max_lr'], float)
    assert lr_config['min_lr'] < lr_config['max_lr']
    
    # Test batch_scaler config
    batch_scaler_path = tuning_dir / "batch_scaler.yaml"
    assert batch_scaler_path.exists(), "batch_scaler.yaml not found"
    
    with open(batch_scaler_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'batch_scaler' in config
    batch_config = config['batch_scaler']
    assert 'mode' in batch_config
    assert batch_config['mode'] in ['power_scaling', 'binsearch']
    
    # Test ablation_study config
    ablation_path = tuning_dir / "ablation_study.yaml"
    assert ablation_path.exists(), "ablation_study.yaml not found"
    
    with open(ablation_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'ablation' in config
    ablation_config = config['ablation']
    assert 'experiment_name' in ablation_config
    assert 'parameters' in ablation_config
    assert 'metrics' in ablation_config
    assert isinstance(ablation_config['parameters'], dict)
    assert isinstance(ablation_config['metrics'], list)


def test_all_configs_yaml_syntax():
    """Test that all YAML files have valid syntax."""
    config_files = list(CONFIG_DIR.rglob("*.yaml"))
    assert len(config_files) > 0, "No YAML config files found"
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML syntax in {config_file}: {e}")


def test_config_inheritance():
    """Test that configs properly inherit from defaults."""
    defaults_path = CONFIG_DIR / "defaults.yaml"
    vision_classifier_path = CONFIG_DIR / "vision" / "classifier.yaml"
    
    with open(defaults_path, 'r') as f:
        defaults = yaml.safe_load(f)
    
    with open(vision_classifier_path, 'r') as f:
        classifier_config = yaml.safe_load(f)
    
    # Check that classifier config has defaults reference
    assert 'defaults' in classifier_config
    assert '/defaults.yaml' in classifier_config['defaults']


def test_checkpoint_paths_consistency():
    """Test that checkpoint paths are consistent across configs."""
    config_files = list(CONFIG_DIR.rglob("*.yaml"))
    
    for config_file in config_files:
        if config_file.name in ['lr_finder.yaml', 'batch_scaler.yaml']:
            continue  # Skip tuning configs
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'callbacks' in config:
            for callback in config['callbacks']:
                if 'ModelCheckpoint' in str(callback.get('class_path', '')):
                    init_args = callback.get('init_args', {})
                    if 'dirpath' in init_args:
                        dirpath = init_args['dirpath']
                        assert isinstance(dirpath, str)
                        assert len(dirpath) > 0
                        # Should contain checkpoints/ prefix
                        assert dirpath.startswith('checkpoints/')


def test_logger_consistency():
    """Test that logger configs are consistent."""
    config_files = list(CONFIG_DIR.rglob("*.yaml"))
    
    for config_file in config_files:
        if config_file.name in ['lr_finder.yaml', 'batch_scaler.yaml']:
            continue  # Skip tuning configs
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'logger' in config:
            loggers = config['logger']
            assert isinstance(loggers, list)
            
            for logger in loggers:
                assert 'class_path' in logger
                assert 'TensorBoardLogger' in logger['class_path']
                
                init_args = logger.get('init_args', {})
                assert 'save_dir' in init_args
                assert init_args['save_dir'] == 'logs/'


def test_learning_rate_ranges():
    """Test that learning rates are in reasonable ranges."""
    config_files = [
        CONFIG_DIR / "vision" / "classifier.yaml",
        CONFIG_DIR / "vision" / "segmenter.yaml",
        CONFIG_DIR / "nlp" / "char_lm.yaml",
        CONFIG_DIR / "nlp" / "sentiment.yaml",
        CONFIG_DIR / "tabular" / "mlp.yaml",
        CONFIG_DIR / "timeseries" / "forecaster.yaml"
    ]
    
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'model' in config and 'init_args' in config['model']:
            init_args = config['model']['init_args']
            if 'learning_rate' in init_args:
                lr = init_args['learning_rate']
                assert isinstance(lr, (int, float))
                assert 1e-6 <= lr <= 1e-1, f"LR {lr} out of range in {config_file}"