"""Two-view, single-step Difix model for rain removal."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from einops import rearrange, repeat
from peft import LoraConfig
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer, CLIPTextModel


BASE_MODEL = "stabilityai/sd-turbo"
DEFAULT_PROMPT = "remove rain streaks and restore a clean natural image"
CHECKPOINT_SCHEMA_VERSION = 2
DEFAULT_VIEW_ORDER = "preliminary_rainy"
LEGACY_VIEW_ORDER = "rainy_preliminary"
RESIDUAL_VIEW_ORDER = "preliminary_residual"
SUPPORTED_VIEW_ORDERS = {DEFAULT_VIEW_ORDER, LEGACY_VIEW_ORDER, RESIDUAL_VIEW_ORDER}
UNET_LORA_TARGETS = [
    "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
    "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2",
    "ff.net.0.proj",
]
VAE_DECODER_LORA_SUFFIXES = [
    "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
    "to_k", "to_q", "to_v", "to_out.0",
]


def _lora_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items() if "lora_" in k}


def _load_lora_state(module: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    current = module.state_dict()
    unknown = sorted(set(state) - set(current))
    if unknown:
        raise ValueError(f"Checkpoint contains unknown LoRA parameters: {unknown[:5]}")
    current.update(state)
    module.load_state_dict(current)


class Difix(torch.nn.Module):
    num_views = 2

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        lora_rank_unet: int = 16,
        lora_rank_vae: int = 4,
        timestep: int = 199,
        prompt: str = DEFAULT_PROMPT,
        cfg_scale: float = 1.0,
        view_order: str = DEFAULT_VIEW_ORDER,
    ) -> None:
        super().__init__()
        if view_order not in SUPPORTED_VIEW_ORDERS:
            raise ValueError(
                f"Unsupported view_order={view_order!r}; expected one of {sorted(SUPPORTED_VIEW_ORDERS)}"
            )
        self.base_model = base_model
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.prompt = prompt
        self.cfg_scale = cfg_scale
        self.view_order = view_order
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
        try:
            from mv_unet import UNet2DConditionModel
        except ImportError:  # package-style imports used by tests and notebooks
            from .mv_unet import UNet2DConditionModel
        self.unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
        self.scheduler.set_timesteps(1)
        self.register_buffer("fixed_timestep", torch.tensor([timestep], dtype=torch.long), persistent=False)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.target_modules_unet = list(UNET_LORA_TARGETS)
        unet_config = LoraConfig(
            r=lora_rank_unet,
            init_lora_weights="gaussian",
            target_modules=self.target_modules_unet,
        )
        self.unet.add_adapter(unet_config, adapter_name="unet_derain")

        self.target_modules_vae = [
            name for name, _module in self.vae.named_modules()
            if name.startswith("decoder.") and any(name.endswith(suffix) for suffix in VAE_DECODER_LORA_SUFFIXES)
        ]
        if not self.target_modules_vae:
            raise RuntimeError("No VAE decoder modules matched the LoRA target list")
        vae_config = LoraConfig(
            r=lora_rank_vae,
            init_lora_weights="gaussian",
            target_modules=self.target_modules_vae,
        )
        self.vae.add_adapter(vae_config, adapter_name="vae_decoder")
        self.set_stage("A")

    @property
    def timestep(self) -> int:
        return int(self.fixed_timestep.item())

    def set_stage(self, stage: str) -> None:
        if stage not in {"A", "B"}:
            raise ValueError(f"stage must be 'A' or 'B', got {stage}")
        self.stage = stage
        self.training = True
        self.unet.train()
        self.vae.eval()
        if stage == "B":
            self.vae.decoder.train()
        self.text_encoder.eval()
        for name, parameter in self.unet.named_parameters():
            parameter.requires_grad_("lora_" in name)
        for name, parameter in self.vae.named_parameters():
            parameter.requires_grad_(stage == "B" and "lora_" in name and "decoder" in name)
        self.text_encoder.requires_grad_(False)

    def trainable_parameter_groups(self, unet_lr: float, vae_lr: float):
        unet = [p for p in self.unet.parameters() if p.requires_grad]
        vae = [p for p in self.vae.parameters() if p.requires_grad]
        groups = [{"params": unet, "lr": unet_lr, "name": "unet_lora"}]
        if vae:
            groups.append({"params": vae, "lr": vae_lr, "name": "vae_decoder_lora"})
        if not unet or any(not p.requires_grad for group in groups for p in group["params"]):
            raise RuntimeError("Optimizer groups must contain only trainable LoRA parameters")
        return groups

    def trainable_vae_lora_parameters(self):
        return [
            parameter for name, parameter in self.vae.named_parameters()
            if parameter.requires_grad and "lora_" in name and "decoder" in name
        ]

    def _prompt_embeddings(self, prompt_tokens: torch.Tensor, views: int):
        embeddings = self.text_encoder(prompt_tokens)[0]
        return repeat(embeddings, "b n c -> (b v) n c", v=views)

    def forward(
        self,
        conditioning_pixel_values: torch.Tensor,
        prompt_tokens: torch.Tensor,
        cfg_scale: float = 1.0,
        empty_prompt_tokens: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if conditioning_pixel_values.ndim != 5 or conditioning_pixel_values.shape[1:] != (2, 3, 512, 512):
            raise ValueError(
                "conditioning_pixel_values must have shape [B,2,3,512,512], got "
                f"{tuple(conditioning_pixel_values.shape)}"
            )
        if cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative")

        batch, views = conditioning_pixel_values.shape[:2]
        pixels = rearrange(conditioning_pixel_values, "b v c h w -> (b v) c h w")
        posterior = self.vae.encode(pixels).latent_dist
        latents = posterior.mode() if deterministic else posterior.sample()
        latents = latents * self.vae.config.scaling_factor
        # Scheduler tensors are not registered model buffers, so keep them on the
        # same device explicitly after Accelerate/model.to() moves the network.
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(latents.device)
        self.scheduler.one = self.scheduler.one.to(latents.device)
        prompt_embeddings = self._prompt_embeddings(prompt_tokens, views)
        timestep = self.fixed_timestep.to(latents.device)

        conditional = self.unet(latents, timestep, encoder_hidden_states=prompt_embeddings).sample
        if cfg_scale == 1.0:
            prediction = conditional
        else:
            if empty_prompt_tokens is None:
                raise ValueError("empty_prompt_tokens are required when cfg_scale != 1.0")
            empty_embeddings = self._prompt_embeddings(empty_prompt_tokens, views)
            unconditional = self.unet(latents, timestep, encoder_hidden_states=empty_embeddings).sample
            prediction = unconditional + cfg_scale * (conditional - unconditional)

        denoised = self.scheduler.step(prediction, timestep, latents, return_dict=True).prev_sample
        decoded = self.vae.decode(denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        return rearrange(decoded, "(b v) c h w -> b v c h w", b=batch, v=views)

    @torch.no_grad()
    def sample(
        self,
        rainy: Image.Image,
        preliminary: Image.Image,
        prompt: Optional[str] = None,
        cfg_scale: Optional[float] = None,
    ) -> Image.Image:
        if rainy.mode != "RGB" or preliminary.mode != "RGB":
            raise ValueError("Rainy and preliminary images must both be RGB")
        if rainy.size != preliminary.size:
            raise ValueError(f"Rainy/reference sizes differ: {rainy.size} vs {preliminary.size}")
        if rainy.size != (512, 512):
            raise ValueError(f"Rainy/reference images must be 512x512, got {rainy.size}")
        prompt = self.prompt if prompt is None else prompt
        cfg_scale = self.cfg_scale if cfg_scale is None else cfg_scale
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        rainy_tensor = TF.to_tensor(rainy)
        preliminary_tensor = TF.to_tensor(preliminary)
        if self.view_order == RESIDUAL_VIEW_ORDER:
            # Match build_rain_residual.py exactly so training-time PNG inputs
            # and inference-time online residuals have identical quantization.
            rainy_u8 = rainy_tensor.mul(255).round().to(torch.int16)
            preliminary_u8 = preliminary_tensor.mul(255).round().to(torch.int16)
            encoded = ((rainy_u8 - preliminary_u8 + 256) // 2).to(torch.float32).div(255)
            pixels = torch.stack((preliminary_tensor, encoded)).mul(2).sub(1)
        else:
            images = {
                LEGACY_VIEW_ORDER: (rainy_tensor, preliminary_tensor),
                DEFAULT_VIEW_ORDER: (preliminary_tensor, rainy_tensor),
            }[self.view_order]
            pixels = torch.stack(images).mul(2).sub(1)
        pixels = pixels.unsqueeze(0).to(device)
        empty_tokens = None
        if cfg_scale != 1.0:
            empty_tokens = self.tokenizer(
                "", max_length=self.tokenizer.model_max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
        output = self.forward(
            pixels, tokens, cfg_scale=cfg_scale,
            empty_prompt_tokens=empty_tokens, deterministic=True,
        )[:, 0]
        return TF.to_pil_image(output[0].float().cpu().add(1).div(2).clamp(0, 1))

    def checkpoint_config(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "num_views": self.num_views,
            "lora_rank_unet": self.lora_rank_unet,
            "lora_rank_vae": self.lora_rank_vae,
            "timestep": self.timestep,
            "prompt": self.prompt,
            "cfg_scale": self.cfg_scale,
            "view_order": self.view_order,
            "stage": self.stage,
        }


def save_checkpoint(
    path: str | Path,
    model: Difix,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    global_step: int,
    training_state: Optional[Dict[str, Any]] = None,
    checkpoint_kind: str = "training",
) -> None:
    if checkpoint_kind not in {"training", "weights"}:
        raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind}")
    if checkpoint_kind == "training" and (optimizer is None or scheduler is None):
        raise ValueError("Training checkpoints require optimizer and scheduler state")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "global_step": int(global_step),
        "stage": model.stage,
        "stage_step": 0,
        "optimizer_groups": [group.get("name", f"group_{index}") for index, group in enumerate(optimizer.param_groups)] if optimizer else [],
    }
    if training_state:
        state.update(training_state)
    if state["global_step"] != int(global_step):
        raise ValueError("training_state global_step differs from save_checkpoint global_step")
    if state["stage"] != model.stage:
        raise ValueError("training_state stage differs from model.stage")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "config": model.checkpoint_config(),
        "training_state": state,
        "unet_lora": _lora_state(model.unet),
        "vae_decoder_lora": _lora_state(model.vae),
    }
    if checkpoint_kind == "training":
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_checkpoint_metadata(path: str | Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {checkpoint.get('schema_version')}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    return {
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "config": checkpoint["config"],
        "training_state": checkpoint["training_state"],
    }


def load_checkpoint(
    path: str | Path,
    model: Difix,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Only checkpoint schema v2 is supported")
    config = checkpoint["config"]
    expected = model.checkpoint_config()
    saved_config = {**config, "view_order": config.get("view_order", LEGACY_VIEW_ORDER)}
    for key in (
        "base_model", "num_views", "lora_rank_unet", "lora_rank_vae",
        "timestep", "prompt", "cfg_scale", "view_order",
    ):
        if saved_config[key] != expected[key]:
            raise ValueError(
                f"Checkpoint {key}={saved_config[key]!r}, model expects {expected[key]!r}"
            )
    model.set_stage(config.get("stage", "A"))
    _load_lora_state(model.unet, checkpoint["unet_lora"])
    _load_lora_state(model.vae, checkpoint["vae_decoder_lora"])
    if optimizer is not None:
        if checkpoint["checkpoint_kind"] != "training":
            raise ValueError("A weights-only checkpoint cannot resume optimizer state")
        expected_groups = checkpoint["training_state"]["optimizer_groups"]
        actual_groups = [group.get("name", f"group_{index}") for index, group in enumerate(optimizer.param_groups)]
        if actual_groups != expected_groups:
            raise ValueError(f"Optimizer groups differ: checkpoint={expected_groups}, model={actual_groups}")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if checkpoint["checkpoint_kind"] != "training":
            raise ValueError("A weights-only checkpoint cannot resume scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["training_state"]


def model_from_checkpoint(path: str | Path, device: torch.device | str = "cuda") -> Difix:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = Difix(
        base_model=config["base_model"],
        lora_rank_unet=config["lora_rank_unet"],
        lora_rank_vae=config["lora_rank_vae"],
        timestep=config["timestep"],
        prompt=config["prompt"],
        cfg_scale=config.get("cfg_scale", 1.0),
        view_order=config.get("view_order", LEGACY_VIEW_ORDER),
    )
    load_checkpoint(path, model)
    model.eval().requires_grad_(False)
    return model.to(device)
