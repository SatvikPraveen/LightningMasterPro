# scripts/predict.py
"""Script for running predictions with trained models."""

import sys
from pathlib import Path
import torch
import lightning.pytorch as L
from lightning.pytorch.cli import LightningArgumentParser

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.cli import LightningMasterProCLI


def main():
    """Main prediction function."""
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(L.Trainer, "trainer")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--data_config", type=str, required=True,
                       help="Path to data configuration file")
    parser.add_argument("--output_dir", type=str, default="predictions/",
                       help="Directory to save predictions")
    
    args = parser.parse_args()
    
    # Load model from checkpoint
    print(f"Loading model from {args.model_path}")
    
    # Create trainer for prediction
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    # Use CLI to load data module from config
    cli = LightningMasterProCLI(
        args=["--config", args.data_config, "--trainer.max_epochs", "1"],
        run=False
    )
    
    # Get model class from checkpoint
    model_class = None
    checkpoint = torch.load(args.model_path, map_location="cpu")
    
    # Determine model class from checkpoint metadata
    if "hyper_parameters" in checkpoint:
        hp = checkpoint["hyper_parameters"]
        if "backbone" in hp:
            if "num_classes" in hp and hp["num_classes"] > 1:
                from lmpro.modules.vision.classifier import VisionClassifier
                model_class = VisionClassifier
            else:
                from lmpro.modules.vision.segmenter import VisionSegmenter
                model_class = VisionSegmenter
        elif "vocab_size" in hp:
            if "sequence_length" in hp:
                from lmpro.modules.nlp.char_lm import CharacterLM
                model_class = CharacterLM
            else:
                from lmpro.modules.nlp.sentiment import SentimentClassifier
                model_class = SentimentClassifier
        elif "input_dim" in hp and "hidden_dims" in hp:
            from lmpro.modules.tabular.mlp_reg_cls import MLPRegCls
            model_class = MLPRegCls
        elif "prediction_length" in hp:
            from lmpro.modules.timeseries.forecaster import TimeSeriesForecaster
            model_class = TimeSeriesForecaster
    
    if model_class is None:
        raise ValueError("Could not determine model class from checkpoint")
    
    # Load model
    model = model_class.load_from_checkpoint(args.model_path)
    model.eval()
    
    # Run predictions
    predictions = trainer.predict(model, datamodule=cli.datamodule)
    
    # Save predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(predictions, output_dir / "predictions.pt")
    print(f"Predictions saved to {output_dir / 'predictions.pt'}")


if __name__ == "__main__":
    main()