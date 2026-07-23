from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.generate_difix_background import (
    FlatRainDataset,
    generate_backgrounds,
    validate_existing_outputs,
)


def _write_rgb(path: Path, color=(10, 20, 30), size=(512, 512)) -> None:
    Image.new("RGB", size, color).save(path)


class BackgroundStub(torch.nn.Module):
    def forward(self, inputs):
        background = torch.full_like(inputs, 0.25)
        return torch.zeros_like(inputs), background, torch.tensor(0.0)


def test_generates_same_stem_pngs_from_flat_input_without_gt(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "background"
    input_dir.mkdir()
    _write_rgb(input_dir / "00010.jpg")
    _write_rgb(input_dir / "00020.jpg")
    dataset = FlatRainDataset(
        input_dir,
        expected_size=(512, 512),
        expected_count=2,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    written = generate_backgrounds(
        BackgroundStub(),
        loader,
        output_dir,
        torch.device("cpu"),
    )

    assert written == 2
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "00010.png",
        "00020.png",
    ]
    with Image.open(output_dir / "00010.png") as image:
        assert image.mode == "RGB"
        assert image.size == (512, 512)
        assert image.getpixel((0, 0)) == (64, 64, 64)


def test_resume_validates_and_skips_completed_outputs(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "background"
    input_dir.mkdir()
    output_dir.mkdir()
    _write_rgb(input_dir / "00010.jpg")
    _write_rgb(input_dir / "00020.jpg")
    _write_rgb(output_dir / "00010.png")
    dataset = FlatRainDataset(input_dir, (512, 512), expected_count=2)
    completed = validate_existing_outputs(
        output_dir,
        set(dataset.stems),
        (512, 512),
        resume=True,
    )

    written = generate_backgrounds(
        BackgroundStub(),
        DataLoader(dataset, batch_size=2, num_workers=0),
        output_dir,
        torch.device("cpu"),
        completed_stems=completed,
    )

    assert completed == {"00010"}
    assert written == 1
    assert (output_dir / "00020.png").is_file()


def test_rejects_wrong_count_size_and_contaminated_output(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_rgb(input_dir / "00010.jpg")
    with pytest.raises(ValueError, match="contains 1 images, expected 2"):
        FlatRainDataset(input_dir, (512, 512), expected_count=2)

    _write_rgb(input_dir / "00020.jpg", size=(256, 512))
    dataset = FlatRainDataset(input_dir, (512, 512), expected_count=2)
    with pytest.raises(ValueError, match="must be 512x512"):
        dataset[1]

    output_dir = tmp_path / "background"
    output_dir.mkdir()
    _write_rgb(output_dir / "unrelated.png")
    with pytest.raises(ValueError, match="absent from input"):
        validate_existing_outputs(
            output_dir,
            set(dataset.stems),
            (512, 512),
            resume=True,
        )


def test_existing_outputs_require_explicit_resume(tmp_path):
    output_dir = tmp_path / "background"
    output_dir.mkdir()
    _write_rgb(output_dir / "00010.png")

    with pytest.raises(FileExistsError, match="Pass --resume"):
        validate_existing_outputs(
            output_dir,
            {"00010"},
            (512, 512),
            resume=False,
        )
