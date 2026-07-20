from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.model import (
    DEFAULT_VIEW_ORDER,
    RESIDUAL_VIEW_ORDER,
    VAE_SKIP_NAMES,
    _vae_decoder_forward_with_skips,
    _vae_encoder_forward_with_skips,
    load_checkpoint,
    read_checkpoint_metadata,
    save_checkpoint,
)
from src.train_difix import (
    checkpoint_to_resume,
    prune_checkpoints,
    resolve_prompt,
    resolve_view_order,
    transition_to_stage_b,
    validate,
)


class TinyBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([1.0]))


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.decoder = torch.nn.Module()
        for index in range(1, 5):
            skip_conv = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
            torch.nn.init.zeros_(skip_conv.weight)
            setattr(self.decoder, f"skip_conv_{index}", skip_conv)


class TinyDifix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = TinyBranch()
        self.vae = TinyVAE()
        self.stage = "A"
        self.set_stage("A")

    def set_stage(self, stage):
        self.stage = stage
        self.unet.lora_weight.requires_grad_(True)
        self.vae.lora_weight.requires_grad_(stage == "B")
        self.vae.decoder.requires_grad_(stage == "B")

    def trainable_vae_parameters(self):
        return [parameter for parameter in self.vae.parameters() if parameter.requires_grad]

    def checkpoint_config(self):
        return {
            "base_model": "tiny",
            "num_views": 2,
            "lora_rank_unet": 1,
            "lora_rank_vae": 1,
            "timestep": 199,
            "prompt": "derain",
            "cfg_scale": 1.0,
            "view_order": "preliminary_rainy",
            "vae_skip": True,
            "stage": self.stage,
        }


class DummyLPIPS(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().flatten(1).mean(1).view(-1, 1, 1, 1)


class AddOne(torch.nn.Module):
    def forward(self, sample, *_args):
        return sample + 1


class PassThroughBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype_anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, sample, *_args):
        return sample


class DummyCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips_model = DummyLPIPS()

    def forward(self, prediction, target):
        return (prediction - target).square().mean(), {}


class ValidationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = torch.nn.Identity()
        self.vae = torch.nn.Module()
        self.vae.encoder = torch.nn.Identity()
        self.vae.decoder = torch.nn.Identity()
        self.text_encoder = torch.nn.Identity()
        self.stage = "A"
        self.set_stage("A")

    def set_stage(self, stage):
        self.stage = stage
        self.training = True
        self.unet.train()
        self.vae.eval()
        if stage == "B":
            self.vae.decoder.train()
        self.text_encoder.eval()

    def forward(self, conditioning, _tokens, deterministic=False):
        assert deterministic
        return conditioning


class DummyAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model):
        return model


def _args():
    return SimpleNamespace(stage_b_unet_lr=1e-5, vae_lr=1e-5)


def _stage_a_optimizer(model):
    return torch.optim.AdamW([{
        "params": [model.unet.lora_weight],
        "lr": 5e-5,
        "name": "unet_lora",
    }])


def _initialize_adam_state(model, optimizer):
    loss = model.unet.lora_weight.square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_vae_skip_forward_caches_and_injects_features_in_reverse_order():
    encoder = torch.nn.Module()
    encoder.conv_in = torch.nn.Identity()
    encoder.down_blocks = torch.nn.ModuleList([AddOne() for _ in VAE_SKIP_NAMES])
    encoder.mid_block = torch.nn.Identity()
    encoder.conv_norm_out = torch.nn.Identity()
    encoder.conv_act = torch.nn.Identity()
    encoder.conv_out = torch.nn.Identity()
    encoded = _vae_encoder_forward_with_skips(encoder, torch.zeros(1, 1, 2, 2))

    assert torch.equal(encoded, torch.full_like(encoded, 4))
    assert [activation[0, 0, 0, 0].item() for activation in encoder.current_down_blocks] == [
        0, 1, 2, 3,
    ]

    decoder = torch.nn.Module()
    decoder.conv_in = torch.nn.Identity()
    decoder.mid_block = PassThroughBlock()
    decoder.up_blocks = torch.nn.ModuleList([PassThroughBlock() for _ in VAE_SKIP_NAMES])
    decoder.conv_norm_out = PassThroughBlock()
    decoder.conv_act = torch.nn.Identity()
    decoder.conv_out = torch.nn.Identity()
    decoder.incoming_skip_acts = encoder.current_down_blocks
    for index, name in enumerate(VAE_SKIP_NAMES, start=1):
        skip_conv = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            skip_conv.weight.fill_(index)
        setattr(decoder, name, skip_conv)

    decoded = _vae_decoder_forward_with_skips(decoder, torch.zeros(1, 1, 2, 2))
    # Reversed encoder values [3, 2, 1, 0] use skip weights [1, 2, 3, 4].
    assert torch.equal(decoded, torch.full_like(decoded, 10))


def test_stage_b_transition_preserves_unet_adam_state():
    model = TinyDifix()
    optimizer = _stage_a_optimizer(model)
    _initialize_adam_state(model, optimizer)
    exp_avg = optimizer.state[model.unet.lora_weight]["exp_avg"].clone()
    optimizer_id = id(optimizer)

    returned = transition_to_stage_b(model, optimizer, _args())

    assert id(returned) == optimizer_id
    assert [group["name"] for group in optimizer.param_groups] == [
        "unet_lora", "vae_decoder_lora_skip",
    ]
    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["lr"] == 1e-5
    assert torch.equal(optimizer.state[model.unet.lora_weight]["exp_avg"], exp_avg)
    assert model.vae.lora_weight not in optimizer.state
    assert {id(parameter) for parameter in optimizer.param_groups[1]["params"]} == {
        id(parameter) for parameter in model.vae.parameters() if parameter.requires_grad
    }


def test_stage_a_checkpoint_roundtrip_and_boundary_resume(tmp_path):
    model = TinyDifix()
    optimizer = _stage_a_optimizer(model)
    _initialize_adam_state(model, optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    path = tmp_path / "stage-a-final.pt"
    save_checkpoint(
        path, model, optimizer, scheduler, 15_000,
        {"global_step": 15_000, "stage": "A", "stage_step": 15_000, "best_metrics": None},
    )

    restored = TinyDifix()
    restored_optimizer = _stage_a_optimizer(restored)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _step: 1.0)
    state = load_checkpoint(path, restored, restored_optimizer, restored_scheduler)

    assert state["global_step"] == 15_000
    assert state["stage"] == "A"
    assert torch.equal(restored.unet.lora_weight, model.unet.lora_weight)
    transition_to_stage_b(restored, restored_optimizer, _args())
    assert restored.stage == "B"
    assert [group["name"] for group in restored_optimizer.param_groups] == [
        "unet_lora", "vae_decoder_lora_skip",
    ]
    assert restored_optimizer.state[restored.unet.lora_weight]["step"].item() == 1

    stage_b_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _step: 1.0)
    stage_b_path = tmp_path / "checkpoint-15000.pt"
    save_checkpoint(
        stage_b_path, restored, restored_optimizer, stage_b_scheduler, 15_000,
        {"global_step": 15_000, "stage": "B", "stage_step": 0, "best_metrics": None},
    )
    stage_b_restored = TinyDifix()
    stage_b_restored.set_stage("B")
    stage_b_optimizer = torch.optim.AdamW([
        {"params": [stage_b_restored.unet.lora_weight], "lr": 1e-5, "name": "unet_lora"},
        {
            "params": stage_b_restored.trainable_vae_parameters(),
            "lr": 1e-5,
            "name": "vae_decoder_lora_skip",
        },
    ])
    stage_b_scheduler_restored = torch.optim.lr_scheduler.LambdaLR(stage_b_optimizer, lambda _step: 1.0)
    stage_b_state = load_checkpoint(
        stage_b_path, stage_b_restored, stage_b_optimizer, stage_b_scheduler_restored
    )
    assert stage_b_state["stage"] == "B"
    assert stage_b_state["stage_step"] == 0
    assert len(stage_b_optimizer.param_groups) == 2


def test_weights_checkpoint_has_no_optimizer(tmp_path):
    model = TinyDifix()
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, None, None, 42, checkpoint_kind="weights")
    metadata = read_checkpoint_metadata(path)
    payload = torch.load(path, map_location="cpu")
    assert metadata["checkpoint_kind"] == "weights"
    assert "optimizer" not in payload
    assert "scheduler" not in payload
    assert payload["config"]["vae_skip"] is True
    assert sorted(payload["vae_skip"]) == [f"skip_conv_{index}" for index in range(1, 5)]


def test_legacy_view_order_checkpoint_cannot_resume_new_training(tmp_path):
    model = TinyDifix()
    optimizer = _stage_a_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, model, optimizer, scheduler, 1)
    payload = torch.load(path, map_location="cpu")
    del payload["config"]["view_order"]
    torch.save(payload, path)

    restored = TinyDifix()
    restored_optimizer = _stage_a_optimizer(restored)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda _step: 1.0
    )
    with pytest.raises(ValueError, match="view_order"):
        load_checkpoint(path, restored, restored_optimizer, restored_scheduler)


def test_checkpoint_retention_keeps_latest_five(tmp_path):
    for step in range(1_000, 9_000, 1_000):
        (tmp_path / f"checkpoint-{step}.pt").touch()
    (tmp_path / "checkpoint-broken.pt").touch()
    (tmp_path / "checkpoint-9000.pt.tmp").touch()

    prune_checkpoints(tmp_path, keep_last=5)

    remaining = sorted(path.name for path in tmp_path.glob("checkpoint-*.pt"))
    assert remaining == [
        "checkpoint-4000.pt", "checkpoint-5000.pt", "checkpoint-6000.pt",
        "checkpoint-7000.pt", "checkpoint-8000.pt", "checkpoint-broken.pt",
    ]


def test_auto_resume_ignores_non_numeric_checkpoints(tmp_path):
    resume = tmp_path / "checkpoints" / "resume"
    resume.mkdir(parents=True)
    (resume / "checkpoint-broken.pt").touch()
    (resume / "checkpoint-500.pt").touch()
    (resume / "checkpoint-1500.pt").touch()
    assert checkpoint_to_resume(str(tmp_path)).name == "checkpoint-1500.pt"


def test_prompt_resolution_and_cli_override(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        '{"train":{"prompt":"json prompt"},"test":{"prompt":"json prompt"}}',
        encoding="utf-8",
    )
    assert resolve_prompt(str(path), None) == "json prompt"
    assert resolve_prompt(str(path), "cli prompt") == "cli prompt"


def test_mismatched_split_prompts_require_override(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        '{"train":{"prompt":"train"},"test":{"prompt":"test"}}',
        encoding="utf-8",
    )
    try:
        resolve_prompt(str(path), None)
    except ValueError as error:
        assert "prompts differ" in str(error)
    else:
        raise AssertionError("Expected mismatched prompts to fail")


def test_view_order_is_resolved_from_both_dataset_splits(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"train":{},"test":{}}', encoding="utf-8")
    assert resolve_view_order(str(path)) == DEFAULT_VIEW_ORDER

    path.write_text(
        '{"train":{"residual_image":"train"},'
        '"test":{"residual_image":"test"}}',
        encoding="utf-8",
    )
    assert resolve_view_order(str(path)) == RESIDUAL_VIEW_ORDER

    path.write_text(
        '{"train":{"residual_image":"train"},"test":{}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="both define residual_image"):
        resolve_view_order(str(path))


def test_validation_restores_stage_specific_modes():
    model = ValidationModel()
    criterion = DummyCriterion()
    conditioning = torch.zeros(1, 2, 3, 16, 16)
    conditioning[:, 1] = 1
    batch = {
        "conditioning_pixel_values": conditioning,
        "rainy_pixel_values": torch.ones(1, 3, 16, 16),
        "target_pixel_values": torch.zeros(1, 3, 16, 16),
        "input_ids": torch.zeros(1, 77, dtype=torch.long),
    }
    metrics = validate(model, [batch], criterion, DummyAccelerator())
    assert "val/final_psnr" in metrics
    assert metrics["val/final_psnr"] == metrics["val/preliminary_psnr"]
    assert metrics["val/final_psnr"] > metrics["val/rainy_psnr"]
    assert model.training
    assert model.unet.training
    assert not model.vae.training
    assert not model.text_encoder.training
