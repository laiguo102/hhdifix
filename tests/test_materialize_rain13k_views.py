import json
import os
from pathlib import Path

import pytest
from PIL import Image

from src.materialize_rain13k_views import (
    materialize_views,
    read_split_ids,
    write_difix_config,
)


def _write_rgb(path: Path, color) -> None:
    Image.new("RGB", (512, 512), color).save(path)


def _source_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "processed"
    (root / "input").mkdir(parents=True)
    (root / "target").mkdir()
    for stem, color in (("1", (10, 20, 30)), ("2", (40, 50, 60))):
        _write_rgb(root / "input" / f"{stem}.jpg", color)
        _write_rgb(root / "target" / f"{stem}.png", color)
    return root


def test_materializes_difix_and_validation_views_and_config(tmp_path):
    source = _source_dataset(tmp_path)
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "difix_train.txt").write_text("1\n", encoding="utf-8")
    (splits / "validation.txt").write_text("2\n", encoding="utf-8")
    output = tmp_path / "views"

    counts = materialize_views(source, splits, output)

    assert counts == {"difix_train": 1, "validation": 1}
    assert os.path.samefile(
        source / "input" / "1.jpg", output / "difix_train" / "input" / "1.jpg"
    )
    assert os.path.samefile(
        source / "target" / "2.png", output / "validation" / "target" / "2.png"
    )

    config_path = write_difix_config(output, tmp_path / "derain.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["train"]["ref_image"] == str(
        output.resolve() / "difix_train" / "background"
    )
    assert "residual_image" not in config["train"]
    assert config["test"]["image"] == str(output.resolve() / "validation" / "input")


def test_rerun_is_safe_but_different_existing_file_fails(tmp_path):
    source = _source_dataset(tmp_path)
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "difix_train.txt").write_text("1\n", encoding="utf-8")
    output = tmp_path / "views"

    materialize_views(source, splits, output, splits=["difix_train"])
    materialize_views(source, splits, output, splits=["difix_train"])
    destination = output / "difix_train" / "input" / "1.jpg"
    destination.unlink()
    _write_rgb(destination, (255, 0, 0))

    with pytest.raises(FileExistsError, match="different contents"):
        materialize_views(source, splits, output, splits=["difix_train"])


def test_split_ids_must_be_unique_bare_stems(tmp_path):
    split = tmp_path / "bad.txt"
    split.write_text("one.png\none.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_split_ids(split)

    split.write_text("../one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bare filename stems"):
        read_split_ids(split)


def test_materialization_rejects_cross_split_overlap(tmp_path):
    source = _source_dataset(tmp_path)
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "difix_train.txt").write_text("1\n", encoding="utf-8")
    (splits / "validation.txt").write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        materialize_views(source, splits, tmp_path / "views")
