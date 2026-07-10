"""GPU/model-download integration checks.

Run explicitly with ``HHDIFIX_RUN_GPU_TESTS=1 pytest -q tests/test_model_integration.py``.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

if os.environ.get("HHDIFIX_RUN_GPU_TESTS") != "1":
    pytest.skip("set HHDIFIX_RUN_GPU_TESTS=1 to run model integration tests", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from src.model import Difix, model_from_checkpoint, save_checkpoint


def _tokens(model, batch=1):
    return model.tokenizer(
        ["remove rain streaks and restore a clean natural image"] * batch,
        max_length=model.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.cuda()


@pytest.fixture(scope="module")
def model():
    return Difix().cuda()


def test_two_view_forward_shape_and_finiteness(model):
    pixels = torch.zeros(1, 2, 3, 512, 512, device="cuda")
    output = model(pixels, _tokens(model), deterministic=True)
    assert output.shape == pixels.shape
    assert torch.isfinite(output).all()


def test_stage_a_and_b_trainable_parameters(model):
    model.set_stage("A")
    assert any(p.requires_grad for p in model.unet.parameters())
    assert not any(p.requires_grad for p in model.vae.parameters())
    model.set_stage("B")
    assert any(p.requires_grad for p in model.unet.parameters())
    assert any(p.requires_grad for p in model.vae.decoder.parameters())
    assert not any(p.requires_grad for p in model.vae.encoder.parameters())


def test_wrong_view_shape_fails(model):
    with pytest.raises(ValueError, match=r"\[B,2,3,512,512\]"):
        model(torch.zeros(1, 1, 3, 512, 512, device="cuda"), _tokens(model))


def test_weights_checkpoint_roundtrip_is_deterministic(model, tmp_path):
    pixels = torch.zeros(1, 2, 3, 512, 512, device="cuda")
    tokens = _tokens(model)
    with torch.no_grad():
        before = model(pixels, tokens, deterministic=True).cpu()
    path = tmp_path / "roundtrip.pt"
    save_checkpoint(path, model, None, None, 0, checkpoint_kind="weights")
    restored = model_from_checkpoint(path, device="cuda")
    with torch.no_grad():
        after = restored(pixels, tokens, deterministic=True).cpu()
    torch.testing.assert_close(before, after)
