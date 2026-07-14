"""Strict paired dataset for two-view rain removal."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from PIL import Image
from torchvision.transforms import functional as TF


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_PROMPT = "remove rain streaks and restore a clean natural image"
PRELIMINARY_VIEW_INDEX = 0
RAINY_VIEW_INDEX = 1


def _files_by_stem(directory: str | Path) -> Dict[str, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    result: Dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(
                    f"Duplicate image stem '{path.stem}' in {directory}: "
                    f"{result[path.stem].name}, {path.name}"
                )
            result[path.stem] = path
    if not result:
        raise ValueError(f"No supported images found in: {directory}")
    return result


def _format_stems(stems: Iterable[str]) -> str:
    values = sorted(stems)
    preview = ", ".join(values[:8])
    return preview + (f" ... ({len(values)} total)" if len(values) > 8 else "")


class PairedDataset(torch.utils.data.Dataset):
    """Return aligned ``[preliminary, rainy] -> clean`` training samples.

    The JSON split is directory based and contains ``image``, ``ref_image``,
    ``target_image`` and optionally ``prompt``. Files are joined by stem, never
    by sorted-list position.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        height: int = 512,
        width: int = 512,
        tokenizer=None,
        horizontal_flip_prob: float = 0.5,
        reference_dropout_prob: float = 0.2,
        clean_identity_prob: float = 0.1,
        prompt_override: Optional[str] = None,
    ) -> None:
        super().__init__()
        if (height, width) != (512, 512):
            raise ValueError("Rain-removal inputs are fixed at RGB 512x512; resizing is disabled")
        for name, value in {
            "horizontal_flip_prob": horizontal_flip_prob,
            "reference_dropout_prob": reference_dropout_prob,
            "clean_identity_prob": clean_identity_prob,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if reference_dropout_prob + clean_identity_prob > 1.0:
            raise ValueError("reference_dropout_prob + clean_identity_prob must not exceed 1")

        with Path(dataset_path).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if split not in config:
            raise KeyError(f"Split '{split}' not found in {dataset_path}")
        split_config = config[split]
        required = {"image", "ref_image", "target_image"}
        missing_keys = required - set(split_config)
        if missing_keys:
            raise KeyError(f"Split '{split}' is missing keys: {sorted(missing_keys)}")

        rainy = _files_by_stem(split_config["image"])
        preliminary = _files_by_stem(split_config["ref_image"])
        clean = _files_by_stem(split_config["target_image"])
        stem_sets = {
            "image": set(rainy),
            "ref_image": set(preliminary),
            "target_image": set(clean),
        }
        expected = stem_sets["image"]
        mismatches = []
        for name, stems in stem_sets.items():
            if stems != expected:
                missing = expected - stems
                extra = stems - expected
                mismatches.append(
                    f"{name}: missing=[{_format_stems(missing)}], extra=[{_format_stems(extra)}]"
                )
        if mismatches:
            raise ValueError("Image stems do not match across directories; " + "; ".join(mismatches))

        self.samples = [(rainy[s], preliminary[s], clean[s]) for s in sorted(expected)]
        self.configured_prompt = split_config.get("prompt", DEFAULT_PROMPT)
        self.prompt = prompt_override if prompt_override is not None else self.configured_prompt
        self.tokenizer = tokenizer
        self.horizontal_flip_prob = horizontal_flip_prob
        self.reference_dropout_prob = reference_dropout_prob
        self.clean_identity_prob = clean_identity_prob
        self._validate_images()

    def _validate_images(self) -> None:
        for rainy_path, preliminary_path, clean_path in self.samples:
            sizes = []
            for role, path in (
                ("image", rainy_path),
                ("ref_image", preliminary_path),
                ("target_image", clean_path),
            ):
                with Image.open(path) as image:
                    if image.mode != "RGB":
                        raise ValueError(f"{role} must be RGB, got {image.mode}: {path}")
                    sizes.append(image.size)
            if len(set(sizes)) != 1:
                raise ValueError(
                    f"Aligned images have different sizes for stem '{rainy_path.stem}': {sizes}"
                )
            if sizes[0] != (512, 512):
                raise ValueError(
                    f"Images must be 512x512 for stem '{rainy_path.stem}', got {sizes[0]}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return TF.to_tensor(image)

    def __getitem__(self, index: int):
        rainy_path, preliminary_path, clean_path = self.samples[index]
        rainy = self._load(rainy_path)
        preliminary = self._load(preliminary_path)
        clean = self._load(clean_path)

        augmentation_draw = random.random()
        if augmentation_draw < self.clean_identity_prob:
            rainy = clean.clone()
            preliminary = clean.clone()
        elif augmentation_draw < self.clean_identity_prob + self.reference_dropout_prob:
            preliminary = rainy.clone()

        if random.random() < self.horizontal_flip_prob:
            rainy = TF.hflip(rainy)
            preliminary = TF.hflip(preliminary)
            clean = TF.hflip(clean)

        # View 0 is the image to refine, so the supervised output is decoded
        # from the preliminary image latent. The rainy image remains view 1 and
        # contributes degradation evidence through the multi-view attention.
        conditioning = torch.stack((preliminary, rainy), dim=0).mul(2.0).sub(1.0)
        target = clean.mul(2.0).sub(1.0)
        output = {
            "conditioning_pixel_values": conditioning,
            "target_pixel_values": target,
            "caption": self.prompt,
        }
        if self.tokenizer is not None:
            output["input_ids"] = self.tokenizer(
                self.prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
        return output
