import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.dataset import PairedDataset


class TokenizerStub:
    model_max_length = 77

    def __call__(self, *_args, **_kwargs):
        return type("Tokens", (), {"input_ids": torch.zeros((1, 77), dtype=torch.long)})()


def _write_rgb(path: Path, color=(10, 20, 30), size=(512, 512)):
    Image.new("RGB", size, color).save(path)


def _dataset_config(tmp_path: Path):
    dirs = {}
    for key in ("image", "ref_image", "target_image"):
        dirs[key] = tmp_path / key
        dirs[key].mkdir()
        _write_rgb(dirs[key] / "sample.png")
    config = {"train": {**{k: str(v) for k, v in dirs.items()}, "prompt": "derain"}}
    path = tmp_path / "data.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, dirs


def test_shapes_ranges_and_tokens(tmp_path):
    config, _ = _dataset_config(tmp_path)
    dataset = PairedDataset(
        config,
        "train",
        tokenizer=TokenizerStub(),
        horizontal_flip_prob=0,
        reference_dropout_prob=0,
        clean_identity_prob=0,
    )
    sample = dataset[0]
    assert sample["conditioning_pixel_values"].shape == (2, 3, 512, 512)
    assert sample["target_pixel_values"].shape == (3, 512, 512)
    assert sample["input_ids"].shape == (77,)
    assert -1 <= sample["conditioning_pixel_values"].min() <= 1
    assert -1 <= sample["target_pixel_values"].max() <= 1


def test_stem_mismatch_fails(tmp_path):
    config, dirs = _dataset_config(tmp_path)
    (dirs["ref_image"] / "sample.png").rename(dirs["ref_image"] / "wrong.png")
    with pytest.raises(ValueError, match="stems do not match"):
        PairedDataset(config, "train")


def test_size_mismatch_fails(tmp_path):
    config, dirs = _dataset_config(tmp_path)
    _write_rgb(dirs["target_image"] / "sample.png", size=(256, 512))
    with pytest.raises(ValueError, match="different sizes"):
        PairedDataset(config, "train")


def test_non_rgb_fails(tmp_path):
    config, dirs = _dataset_config(tmp_path)
    Image.new("L", (512, 512), 10).save(dirs["image"] / "sample.png")
    with pytest.raises(ValueError, match="must be RGB"):
        PairedDataset(config, "train")
