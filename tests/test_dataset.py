import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.dataset import (
    PRELIMINARY_VIEW_INDEX,
    RAINY_VIEW_INDEX,
    RESIDUAL_VIEW_INDEX,
    PairedDataset,
)


class TokenizerStub:
    model_max_length = 77

    def __call__(self, *_args, **_kwargs):
        return type("Tokens", (), {"input_ids": torch.zeros((1, 77), dtype=torch.long)})()


def _write_rgb(path: Path, color=(10, 20, 30), size=(512, 512)):
    Image.new("RGB", size, color).save(path)


def _dataset_config(tmp_path: Path, with_residual=False):
    dirs = {}
    colors = {
        "image": (10, 20, 30),
        "ref_image": (40, 50, 60),
        "target_image": (70, 80, 90),
        "residual_image": (128, 178, 103),
    }
    keys = ["image", "ref_image", "target_image"]
    if with_residual:
        keys.append("residual_image")
    for key in keys:
        dirs[key] = tmp_path / key
        dirs[key].mkdir()
        _write_rgb(dirs[key] / "sample.png", color=colors[key])
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
    expected_preliminary = torch.tensor([40, 50, 60]).div(255).mul(2).sub(1)
    expected_rainy = torch.tensor([10, 20, 30]).div(255).mul(2).sub(1)
    torch.testing.assert_close(
        sample["conditioning_pixel_values"][PRELIMINARY_VIEW_INDEX, :, 0, 0],
        expected_preliminary,
    )
    torch.testing.assert_close(
        sample["conditioning_pixel_values"][RAINY_VIEW_INDEX, :, 0, 0],
        expected_rainy,
    )
    torch.testing.assert_close(sample["rainy_pixel_values"][:, 0, 0], expected_rainy)


def test_residual_view_keeps_original_rainy_for_metrics(tmp_path):
    config, _ = _dataset_config(tmp_path, with_residual=True)
    dataset = PairedDataset(
        config,
        "train",
        horizontal_flip_prob=0,
        reference_dropout_prob=0,
        clean_identity_prob=0,
    )

    sample = dataset[0]

    expected_residual = torch.tensor([128, 178, 103]).div(255).mul(2).sub(1)
    expected_rainy = torch.tensor([10, 20, 30]).div(255).mul(2).sub(1)
    torch.testing.assert_close(
        sample["conditioning_pixel_values"][RESIDUAL_VIEW_INDEX, :, 0, 0],
        expected_residual,
    )
    torch.testing.assert_close(sample["rainy_pixel_values"][:, 0, 0], expected_rainy)


@pytest.mark.parametrize(
    ("reference_dropout", "clean_identity", "expected_view0"),
    [
        (1.0, 0.0, torch.tensor([40, 50, 60]).div(255).mul(2).sub(1)),
        (0.0, 1.0, torch.tensor([70, 80, 90]).div(255).mul(2).sub(1)),
    ],
)
def test_residual_dropout_and_identity_use_zero_auxiliary(
    tmp_path, reference_dropout, clean_identity, expected_view0
):
    config, _ = _dataset_config(tmp_path, with_residual=True)
    dataset = PairedDataset(
        config,
        "train",
        horizontal_flip_prob=0,
        reference_dropout_prob=reference_dropout,
        clean_identity_prob=clean_identity,
    )

    conditioning = dataset[0]["conditioning_pixel_values"]

    torch.testing.assert_close(
        conditioning[PRELIMINARY_VIEW_INDEX, :, 0, 0], expected_view0
    )
    torch.testing.assert_close(
        conditioning[RESIDUAL_VIEW_INDEX], torch.zeros_like(conditioning[RESIDUAL_VIEW_INDEX])
    )


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
