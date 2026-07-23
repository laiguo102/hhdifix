"""Generate same-stem preliminary backgrounds for DiFix from a frozen UNet.

This script intentionally lives in the DiFix project and does not modify or use
the UNet project's folder-based inference Dataset. It reads one flat directory
of rainy RGB images, loads the GeneralDecompositionNet implementation from an
explicit ``--unet_root``, and writes only ``<stem>.png`` background predictions.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def files_by_stem(directory: str | Path) -> Dict[str, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files: Dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in files:
            raise ValueError(
                f"Duplicate image stem '{path.stem}' in {directory}: "
                f"{files[path.stem].name}, {path.name}"
            )
        files[path.stem] = path
    if not files:
        raise ValueError(f"No supported images found in: {directory}")
    return files


class FlatRainDataset(Dataset):
    """Load a flat, same-resolution directory without reading any GT."""

    def __init__(
        self,
        input_dir: str | Path,
        expected_size: tuple[int, int],
        expected_count: int | None = None,
    ) -> None:
        self.files = files_by_stem(input_dir)
        self.stems = sorted(self.files)
        self.expected_size = expected_size
        if expected_count is not None and len(self.stems) != expected_count:
            raise ValueError(
                f"Input directory contains {len(self.stems)} images, expected "
                f"{expected_count}: {input_dir}"
            )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict:
        stem = self.stems[index]
        path = self.files[stem]
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"Input image must be RGB, got {image.mode}: {path}")
            if image.size != self.expected_size:
                raise ValueError(
                    f"Input image must be {self.expected_size[0]}x"
                    f"{self.expected_size[1]}, got {image.size}: {path}"
                )
            tensor = TF.to_tensor(image)
        return {"input": tensor, "stem": stem}


def import_unet_api(unet_root: str | Path):
    """Import the exact GeneralDecompositionNet implementation requested."""

    unet_root = Path(unet_root).resolve()
    model_file = unet_root / "general_decomposition_model.py"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"UNet implementation does not exist: {model_file}"
        )
    sys.path.insert(0, str(unet_root.parent))
    module = importlib.import_module(
        f"{unet_root.name}.general_decomposition_model"
    )
    imported_file = Path(module.__file__).resolve()
    if imported_file != model_file:
        raise RuntimeError(
            f"Imported UNet implementation from {imported_file}, expected {model_file}"
        )
    return module.GeneralDecompositionNet, module.validate_checkpoint_architecture


def load_model(
    unet_root: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    GeneralDecompositionNet, validate_architecture = import_unet_api(unet_root)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    config = checkpoint.get("config", {})
    validate_architecture(config)
    if config.get("use_orient_block", False) and device.type != "cuda":
        raise RuntimeError("This DCNv4 checkpoint requires --device cuda")

    model = GeneralDecompositionNet(
        in_channels=config.get("in_channels", 3),
        base_channels=config.get("base_channels", 64),
        bottleneck_type=config.get("bottleneck_type", "conv"),
        use_orient_block=config.get("use_orient_block", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, config


def validate_existing_outputs(
    output_dir: str | Path,
    input_stems: set[str],
    expected_size: tuple[int, int],
    resume: bool,
) -> set[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = files_by_stem(output_dir) if any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in output_dir.iterdir()
    ) else {}
    extra = sorted(set(existing) - input_stems)
    if extra:
        raise ValueError(
            f"Output directory contains stems absent from input: {extra[:8]}"
        )
    if existing and not resume:
        raise FileExistsError(
            f"Output directory already contains {len(existing)} images: {output_dir}. "
            "Pass --resume to validate and keep completed outputs."
        )
    for stem, path in existing.items():
        if path.suffix.lower() != ".png":
            raise ValueError(f"Existing background output must be PNG: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != expected_size:
                raise ValueError(
                    f"Existing output is not RGB {expected_size}: {path} "
                    f"(mode={image.mode}, size={image.size})"
                )
    return set(existing)


def save_png_atomic(tensor: torch.Tensor, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.stem}.tmp.png")
    image = TF.to_pil_image(tensor.detach().float().cpu().clamp(0, 1))
    image.save(temporary, format="PNG")
    os.replace(temporary, destination)


def autocast_context(device: torch.device, precision: str):
    if precision == "no":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"--precision {precision} requires CUDA")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.autocast(device_type="cuda", dtype=dtype)


def generate_backgrounds(
    model,
    dataloader: DataLoader,
    output_dir: str | Path,
    device: torch.device,
    precision: str = "no",
    completed_stems: set[str] | None = None,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_stems = completed_stems or set()
    written = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Generating DiFix backgrounds"):
            stems = list(batch["stem"])
            pending_indices = [
                index for index, stem in enumerate(stems)
                if stem not in completed_stems
            ]
            if not pending_indices:
                continue
            inputs = batch["input"][pending_indices].to(
                device,
                non_blocking=True,
            )
            with autocast_context(device, precision):
                _pattern, background, _orthogonal_loss = model(inputs)
            for local_index, batch_index in enumerate(pending_indices):
                stem = stems[batch_index]
                save_png_atomic(
                    background[local_index],
                    output_dir / f"{stem}.png",
                )
                written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate same-stem UNet backgrounds for DiFix training."
    )
    parser.add_argument("--unet_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_count", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Autocast precision; 'no' matches the original FP32 inference path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and skip already generated same-stem RGB PNGs.",
    )
    args = parser.parse_args()
    if args.expected_count <= 0:
        parser.error("--expected_count must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    return args


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("--input_dir and --output_dir must be different")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    input_files = files_by_stem(input_dir)
    if len(input_files) != args.expected_count:
        raise ValueError(
            f"Input directory contains {len(input_files)} images, expected "
            f"{args.expected_count}: {input_dir}"
        )
    model, config = load_model(args.unet_root, args.checkpoint, device)
    width = int(config.get("image_width", 512))
    height = int(config.get("image_height", 512))
    expected_size = (width, height)
    dataset = FlatRainDataset(
        input_dir,
        expected_size=expected_size,
        expected_count=args.expected_count,
    )
    completed = validate_existing_outputs(
        output_dir,
        set(dataset.stems),
        expected_size,
        resume=args.resume,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    written = generate_backgrounds(
        model,
        dataloader,
        output_dir,
        device,
        precision=args.precision,
        completed_stems=completed,
    )
    outputs = files_by_stem(output_dir)
    if set(outputs) != set(dataset.stems):
        missing = sorted(set(dataset.stems) - set(outputs))
        extra = sorted(set(outputs) - set(dataset.stems))
        raise RuntimeError(
            f"Output/input stem mismatch after inference; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    if len(outputs) != args.expected_count:
        raise RuntimeError(
            f"Generated {len(outputs)} outputs, expected {args.expected_count}"
        )
    print(
        f"Complete: total={len(outputs)}, newly_written={written}, "
        f"resumed={len(completed)}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main(parse_args())
