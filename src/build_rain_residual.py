"""Build signed ``rainy - preliminary`` residual images for view 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from dataset import _files_by_stem
except ImportError:
    from .dataset import _files_by_stem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rainy_dir", required=True)
    parser.add_argument("--preliminary_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_count", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing residual PNG files instead of failing",
    )
    return parser.parse_args()


def encode_signed_residual(
    rainy: np.ndarray, preliminary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 centered encoding and the exact signed int16 residual.

    The encoding maps ``[-255, 255]`` to ``[0, 255]`` with zero near 128.
    After the project's standard ``2*x/255 - 1`` normalization, the tensor
    approximates ``(rainy - preliminary) / 255`` with at most 1/255 error.
    """
    if rainy.shape != preliminary.shape:
        raise ValueError(f"Array shapes differ: {rainy.shape} vs {preliminary.shape}")
    if rainy.ndim != 3 or rainy.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB arrays, got {rainy.shape}")
    difference = rainy.astype(np.int16) - preliminary.astype(np.int16)
    encoded = ((difference + 256) // 2).astype(np.uint8)
    return encoded, difference


def build_residuals(args) -> dict:
    rainy_files = _files_by_stem(args.rainy_dir)
    preliminary_files = _files_by_stem(args.preliminary_dir)
    rainy_stems = set(rainy_files)
    preliminary_stems = set(preliminary_files)
    if rainy_stems != preliminary_stems:
        missing = sorted(rainy_stems - preliminary_stems)
        extra = sorted(preliminary_stems - rainy_stems)
        raise ValueError(
            "Rainy/preliminary stems differ; "
            f"missing preliminary={missing[:10]}, extra preliminary={extra[:10]}"
        )
    if args.expected_count is not None and len(rainy_stems) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} pairs, found {len(rainy_stems)}"
        )

    rainy_dir = Path(args.rainy_dir).resolve()
    preliminary_dir = Path(args.preliminary_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir in {rainy_dir, preliminary_dir}:
        raise ValueError("output_dir must differ from both input directories")
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = {path.stem: path for path in output_dir.glob("*.png")}
    unexpected = sorted(set(existing) - rainy_stems)
    if unexpected:
        raise ValueError(f"Output directory contains unexpected PNG stems: {unexpected[:10]}")
    already_present = sorted(set(existing) & rainy_stems)
    if already_present and not args.overwrite:
        raise FileExistsError(
            f"{len(already_present)} output PNG files already exist; pass --overwrite to replace them"
        )

    absolute_sum = 0
    value_count = 0
    residual_min = 255
    residual_max = -255
    for stem in tqdm(sorted(rainy_stems), desc="Building signed rain residuals"):
        with Image.open(rainy_files[stem]) as rainy_image, Image.open(
            preliminary_files[stem]
        ) as preliminary_image:
            if rainy_image.mode != "RGB" or preliminary_image.mode != "RGB":
                raise ValueError(f"Images for '{stem}' must both be RGB")
            if rainy_image.size != preliminary_image.size:
                raise ValueError(
                    f"Image sizes differ for '{stem}': "
                    f"{rainy_image.size} vs {preliminary_image.size}"
                )
            if rainy_image.size != (512, 512):
                raise ValueError(f"Images for '{stem}' must be 512x512, got {rainy_image.size}")
            encoded, difference = encode_signed_residual(
                np.asarray(rainy_image), np.asarray(preliminary_image)
            )

        absolute_sum += int(np.abs(difference).sum(dtype=np.int64))
        value_count += difference.size
        residual_min = min(residual_min, int(difference.min()))
        residual_max = max(residual_max, int(difference.max()))
        destination = output_dir / f"{stem}.png"
        temporary = output_dir / f".{stem}.png.tmp"
        Image.fromarray(encoded).save(temporary, format="PNG")
        os.replace(temporary, destination)

    result = {
        "count": len(rainy_stems),
        "output_dir": str(output_dir),
        "encoding": "uint8(floor((rainy-preliminary+256)/2)); zero is 128",
        "residual_min": residual_min,
        "residual_max": residual_max,
        "mean_absolute_residual": absolute_sum / value_count,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    build_residuals(parse_args())
