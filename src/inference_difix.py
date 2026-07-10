"""Deterministic directory inference for rain removal."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

try:
    from dataset import _files_by_stem
    from model import model_from_checkpoint
except ImportError:
    from .dataset import _files_by_stem
    from .model import model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rainy_dir", required=True)
    parser.add_argument("--preliminary_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=None, help="Optional override; defaults to the checkpoint prompt")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main(args):
    rainy = _files_by_stem(args.rainy_dir)
    preliminary = _files_by_stem(args.preliminary_dir)
    if set(rainy) != set(preliminary):
        raise ValueError(
            f"Rainy/preliminary stems differ; missing preliminary={sorted(set(rainy)-set(preliminary))}, "
            f"extra preliminary={sorted(set(preliminary)-set(rainy))}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = model_from_checkpoint(args.checkpoint, device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in tqdm(sorted(rainy), desc="Deraining"):
        with Image.open(rainy[stem]) as rainy_image, Image.open(preliminary[stem]) as preliminary_image:
            if rainy_image.mode != "RGB" or preliminary_image.mode != "RGB":
                raise ValueError(f"Images for '{stem}' must be RGB")
            if rainy_image.size != preliminary_image.size:
                raise ValueError(f"Images for '{stem}' have different sizes")
            output = model.sample(rainy_image, preliminary_image, prompt=args.prompt)
        output.save(output_dir / f"{stem}.png")


if __name__ == "__main__":
    main(parse_args())
