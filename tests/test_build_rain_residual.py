from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.build_rain_residual import build_residuals, encode_signed_residual


def test_signed_residual_encoding_preserves_both_signs():
    rainy = np.array([[[0, 128, 255]]], dtype=np.uint8)
    preliminary = np.array([[[255, 128, 0]]], dtype=np.uint8)

    encoded, difference = encode_signed_residual(rainy, preliminary)

    np.testing.assert_array_equal(difference, [[[-255, 0, 255]]])
    np.testing.assert_array_equal(encoded, [[[0, 128, 255]]])
    normalized = encoded.astype(np.float32) / 127.5 - 1.0
    np.testing.assert_allclose(
        normalized,
        difference.astype(np.float32) / 255.0,
        atol=1.0 / 255.0 + 1e-7,
    )


def test_signed_residual_encoding_rejects_bad_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        encode_signed_residual(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((3, 2, 3), dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="HWC RGB"):
        encode_signed_residual(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
        )


def test_build_residuals_writes_centered_png_and_refuses_stale_output(tmp_path):
    rainy_dir = tmp_path / "rain"
    preliminary_dir = tmp_path / "preliminary"
    output_dir = tmp_path / "residual"
    rainy_dir.mkdir()
    preliminary_dir.mkdir()
    Image.new("RGB", (512, 512), (200, 100, 50)).save(rainy_dir / "sample.png")
    Image.new("RGB", (512, 512), (100, 100, 100)).save(
        preliminary_dir / "sample.png"
    )
    args = SimpleNamespace(
        rainy_dir=str(rainy_dir),
        preliminary_dir=str(preliminary_dir),
        output_dir=str(output_dir),
        expected_count=1,
        overwrite=False,
    )

    result = build_residuals(args)

    assert result["count"] == 1
    with Image.open(output_dir / "sample.png") as residual:
        assert residual.mode == "RGB"
        assert residual.size == (512, 512)
        assert residual.getpixel((0, 0)) == (178, 128, 103)
    with pytest.raises(FileExistsError, match="--overwrite"):
        build_residuals(args)
