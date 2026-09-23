# scripts/predict.py
"""Run ``Trainer.predict`` with a trained checkpoint and save the outputs.

The training YAML is reused so the datamodule (and its preprocessing) is built
exactly as during training; only the weights come from the checkpoint.

Usage::

    python scripts/predict.py --config configs/vision/classifier.yaml \
        --checkpoint checkpoints/vision/classifier/last.ckpt --output_dir predictions/vision

Any extra ``--key value`` arguments are forwarded to the CLI as overrides, e.g.
``--data.init_args.batch_size 256``.  The equivalent one-liner without this
script is ``python scripts/train.py predict --config ... --ckpt_path ...``.
"""

import argparse
from pathlib import Path
from typing import List

import torch
from lightning.pytorch.utilities.model_helpers import is_overridden

from lmpro.cli import LightningMasterCLI


def parse_args() -> "tuple[argparse.Namespace, List[str]]":
    parser = argparse.ArgumentParser(description="Predict with a LightningMasterPro checkpoint")
    parser.add_argument("--config", required=True, help="Training YAML config (model + data + trainer).")
    parser.add_argument("--checkpoint", required=True, help="Path to the .ckpt file.")
    parser.add_argument("--output_dir", default="predictions", help="Where to save predictions.pt")
    parser.add_argument("--split", default="predict", choices=["predict", "test", "val"], help="Dataloader to use.")
    return parser.parse_known_args()


def main() -> Path:
    args, overrides = parse_args()

    cli = LightningMasterCLI.from_config(
        args.config,
        *overrides,
        "--trainer.logger=false",
        "--trainer.callbacks=[]",
    )
    model_cls = type(cli.model)
    model = model_cls.load_from_checkpoint(args.checkpoint, map_location="cpu")

    datamodule = cli.datamodule
    if args.split == "predict" and is_overridden("predict_dataloader", datamodule):
        predictions = cli.trainer.predict(model, datamodule=datamodule)
    else:
        stage = "test" if args.split in ("predict", "test") else "fit"
        datamodule.setup(stage)
        loader = datamodule.test_dataloader() if stage == "test" else datamodule.val_dataloader()
        predictions = cli.trainer.predict(model, dataloaders=loader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "predictions.pt"
    torch.save(predictions, out_file)
    print(f"Saved {len(predictions or [])} prediction batches to {out_file}")
    return out_file


if __name__ == "__main__":
    main()
