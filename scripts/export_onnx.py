# scripts/export_onnx.py
"""Export a trained LightningModule to ONNX and verify it with onnxruntime.

Usage::

    python scripts/export_onnx.py --config configs/vision/classifier.yaml \
        --checkpoint checkpoints/vision/classifier/last.ckpt --output exports/classifier.onnx

What it demonstrates
--------------------
* ``LightningModule.to_onnx`` with an input sample taken from the real datamodule
  (so dtypes and shapes are always right, including integer token ids for NLP).
* A dynamic batch axis so the exported graph accepts any batch size.
* ``onnx.checker`` structural validation.
* Numerical parity: PyTorch vs onnxruntime outputs are compared with ``np.allclose``.
"""

import argparse
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import torch

from lmpro.cli import LightningMasterCLI


def parse_args() -> "tuple[argparse.Namespace, List[str]]":
    parser = argparse.ArgumentParser(description="Export a LightningMasterPro checkpoint to ONNX")
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .ckpt file.")
    parser.add_argument("--output", required=True, help="Destination .onnx path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--static_batch", action="store_true", help="Disable the dynamic batch axis.")
    parser.add_argument("--no_verify", action="store_true", help="Skip the onnxruntime parity check.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for the parity check.")
    return parser.parse_known_args()


def _first_batch_input(datamodule: Any) -> torch.Tensor:
    """Return the model input from the first validation batch (batch size 1)."""
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))
    x = batch[0] if isinstance(batch, (tuple, list)) else batch["input"] if isinstance(batch, dict) else batch
    return x[:1].clone()


def _as_tuple(outputs: Any) -> Sequence[torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, (tuple, list)):
        flat: List[torch.Tensor] = []
        for item in outputs:
            flat.extend(_as_tuple(item))
        return tuple(flat)
    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def export(config: str, checkpoint: str, output: str, opset: int = 17, dynamic_batch: bool = True) -> Path:
    cli = LightningMasterCLI.from_config(config, "--trainer.logger=false", "--trainer.callbacks=[]")
    model = type(cli.model).load_from_checkpoint(checkpoint, map_location="cpu").eval()

    input_sample = model.example_input_array
    if input_sample is None:
        input_sample = _first_batch_input(cli.datamodule)

    with torch.no_grad():
        outputs = _as_tuple(model(input_sample))
    output_names = [f"output_{i}" for i in range(len(outputs))] if len(outputs) > 1 else ["output"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, **{name: {0: "batch"} for name in output_names}}

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.to_onnx(
        out_path,
        input_sample,
        export_params=True,
        opset_version=opset,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    return out_path


def verify(onnx_path: Path, config: str, checkpoint: str, atol: float = 1e-4) -> float:
    """Structural check plus PyTorch/onnxruntime parity. Returns the max abs difference."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    cli = LightningMasterCLI.from_config(config, "--trainer.logger=false", "--trainer.callbacks=[]")
    model = type(cli.model).load_from_checkpoint(checkpoint, map_location="cpu").eval()
    input_sample = model.example_input_array
    if input_sample is None:
        input_sample = _first_batch_input(cli.datamodule)

    with torch.no_grad():
        torch_out = _as_tuple(model(input_sample))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"input": input_sample.numpy()})

    max_diff = 0.0
    for expected, actual in zip(torch_out, ort_out):
        diff = float(np.max(np.abs(expected.numpy() - actual)))
        max_diff = max(max_diff, diff)
        if not np.allclose(expected.numpy(), actual, atol=atol):
            raise AssertionError(f"ONNX output differs from PyTorch (max abs diff {diff:.3e} > atol {atol})")
    return max_diff


def main() -> Path:
    args, _ = parse_args()
    out_path = export(args.config, args.checkpoint, args.output, args.opset, dynamic_batch=not args.static_batch)
    print(f"Exported to {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    if not args.no_verify:
        max_diff = verify(out_path, args.config, args.checkpoint, args.atol)
        print(f"onnxruntime parity OK (max abs diff {max_diff:.3e})")
    return out_path


if __name__ == "__main__":
    main()
