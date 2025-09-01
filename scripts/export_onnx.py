# scripts/export_onnx.py
"""Script for exporting trained models to ONNX format."""

import sys
from pathlib import Path
import torch
import onnx
from lightning.pytorch.cli import LightningArgumentParser

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Main ONNX export function."""
    parser = LightningArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save the ONNX model")
    parser.add_argument("--input_shape", type=str, default=None,
                       help="Input shape as comma-separated values (e.g., '1,3,224,224')")
    parser.add_argument("--dynamic_axes", action="store_true",
                       help="Use dynamic axes for variable input sizes")
    
    args = parser.parse_args()
    
    # Load model from checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    
    # Determine model class and create dummy input
    model_class = None
    dummy_input = None
    
    if "hyper_parameters" in checkpoint:
        hp = checkpoint["hyper_parameters"]
        
        if "backbone" in hp:
            # Vision models
            if "num_classes" in hp and hp["num_classes"] > 1:
                from lmpro.modules.vision.classifier import VisionClassifier
                model_class = VisionClassifier
                dummy_input = torch.randn(1, 3, 224, 224)  # Standard image size
            else:
                from lmpro.modules.vision.segmenter import VisionSegmenter
                model_class = VisionSegmenter
                dummy_input = torch.randn(1, 3, 256, 256)  # Segmentation size
                
        elif "vocab_size" in hp:
            # NLP models
            if "sequence_length" in hp:
                from lmpro.modules.nlp.char_lm import CharacterLM
                model_class = CharacterLM
                seq_len = hp.get("sequence_length", 256)
                dummy_input = torch.randint(0, hp["vocab_size"], (1, seq_len))
            else:
                from lmpro.modules.nlp.sentiment import SentimentClassifier
                model_class = SentimentClassifier
                dummy_input = torch.randint(0, hp["vocab_size"], (1, 512))
                
        elif "input_dim" in hp and "hidden_dims" in hp:
            # Tabular models
            from lmpro.modules.tabular.mlp_reg_cls import MLPRegCls
            model_class = MLPRegCls
            dummy_input = torch.randn(1, hp["input_dim"])
            
        elif "prediction_length" in hp:
            # Time series models
            from lmpro.modules.timeseries.forecaster import TimeSeriesForecaster
            model_class = TimeSeriesForecaster
            seq_len = hp.get("sequence_length", 100)
            input_dim = hp.get("input_dim", 1)
            dummy_input = torch.randn(1, seq_len, input_dim)
    
    if model_class is None:
        raise ValueError("Could not determine model class from checkpoint")
    
    # Override dummy input if shape provided
    if args.input_shape:
        shape = [int(x) for x in args.input_shape.split(",")]
        dummy_input = torch.randn(*shape)
    
    # Load model
    model = model_class.load_from_checkpoint(args.model_path)
    model.eval()
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up dynamic axes
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    print(f"Input shape: {dummy_input.shape}")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    # Verify the exported model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    print("ONNX export successful!")
    print(f"Model saved to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()