"""Two-stage LoRA training for two-view rain removal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import lpips
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

try:
    from dataset import (
        DEFAULT_PROMPT,
        PRELIMINARY_VIEW_INDEX,
        PairedDataset,
    )
    from loss import DerainLoss, _ssim_map
    from model import (
        DEFAULT_VIEW_ORDER,
        RESIDUAL_VIEW_ORDER,
        Difix,
        load_checkpoint,
        read_checkpoint_metadata,
        save_checkpoint,
    )
except ImportError:
    from .dataset import (
        DEFAULT_PROMPT,
        PRELIMINARY_VIEW_INDEX,
        PairedDataset,
    )
    from .loss import DerainLoss, _ssim_map
    from .model import (
        DEFAULT_VIEW_ORDER,
        RESIDUAL_VIEW_ORDER,
        Difix,
        load_checkpoint,
        read_checkpoint_metadata,
        save_checkpoint,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="stabilityai/sd-turbo")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt_dropout", type=float, default=0.0)
    parser.add_argument("--reference_dropout", type=float, default=0.2)
    parser.add_argument("--clean_identity", type=float, default=0.1)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--stage_a_steps", type=int, default=15_000)
    parser.add_argument("--stage_b_steps", type=int, default=5_000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--unet_lr", type=float, default=5e-5)
    parser.add_argument("--stage_b_unet_lr", type=float, default=1e-5)
    parser.add_argument("--vae_lr", type=float, default=1e-5)
    parser.add_argument("--lora_rank_unet", type=int, default=16)
    parser.add_argument("--lora_rank_vae", type=int, default=4)
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--tracker_project_name", default="hhdifix")
    parser.add_argument("--tracker_run_name", default="derain")
    args = parser.parse_args()
    for name in ("prompt_dropout", "reference_dropout", "clean_identity"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must be in [0,1]")
    for name in ("stage_a_steps", "stage_b_steps", "checkpointing_steps", "checkpoints_total_limit"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.cfg_scale != 1.0 and args.prompt_dropout == 0:
        parser.error("Non-default CFG requires explicit prompt dropout")
    return args


def resolve_prompt(dataset_path: str, cli_prompt: Optional[str]) -> str:
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    train_prompt = config.get("train", {}).get("prompt", DEFAULT_PROMPT)
    test_prompt = config.get("test", {}).get("prompt", DEFAULT_PROMPT)
    if cli_prompt is not None:
        return cli_prompt
    if train_prompt != test_prompt:
        raise ValueError("train/test prompts differ; pass --prompt to provide one explicit prompt")
    return train_prompt


def resolve_view_order(dataset_path: str) -> str:
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    residual_splits = {
        split: "residual_image" in config.get(split, {})
        for split in ("train", "test")
    }
    if len(set(residual_splits.values())) != 1:
        raise ValueError(
            "train/test must either both define residual_image or both omit it"
        )
    return RESIDUAL_VIEW_ORDER if residual_splits["train"] else DEFAULT_VIEW_ORDER


def make_optimizer(model: Difix, args, stage: str):
    model.set_stage(stage)
    unet_lr = args.unet_lr if stage == "A" else args.stage_b_unet_lr
    groups = model.trainable_parameter_groups(unet_lr, args.vae_lr)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)


def make_lr_scheduler(optimizer, args, stage: str):
    if stage == "A":
        return get_scheduler(
            "constant_with_warmup", optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.stage_a_steps,
        )
    return get_scheduler(
        "constant", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=args.stage_b_steps,
    )


def _base_optimizer(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def transition_to_stage_b(model: Difix, optimizer, args):
    """Enable VAE decoder LoRA and skip convs without replacing Adam state."""
    if model.stage != "A":
        raise ValueError(f"Can only transition from stage A, got {model.stage}")
    base_optimizer = _base_optimizer(optimizer)
    if [group.get("name") for group in base_optimizer.param_groups] != ["unet_lora"]:
        raise ValueError("Stage A optimizer must contain exactly the UNet LoRA group")
    unet_state_ids = {id(parameter): state for parameter, state in base_optimizer.state.items()}
    existing = {id(parameter) for group in base_optimizer.param_groups for parameter in group["params"]}

    model.set_stage("B")
    vae_parameters = model.trainable_vae_parameters()
    if not vae_parameters:
        raise RuntimeError("No trainable VAE decoder/skip parameters at stage-B transition")
    if any(id(parameter) in existing for parameter in vae_parameters):
        raise RuntimeError("VAE decoder/skip parameters are already present in the optimizer")

    base_optimizer.param_groups[0]["lr"] = args.stage_b_unet_lr
    base_optimizer.param_groups[0]["initial_lr"] = args.stage_b_unet_lr
    base_optimizer.add_param_group({
        "params": vae_parameters,
        "lr": args.vae_lr,
        "initial_lr": args.vae_lr,
        "name": "vae_decoder_lora_skip",
    })
    for parameter, state in base_optimizer.state.items():
        if id(parameter) in unet_state_ids and state is not unet_state_ids[id(parameter)]:
            raise RuntimeError("UNet optimizer state changed during stage transition")
    return base_optimizer


def checkpoint_to_resume(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    path = Path(path)
    if path.is_file():
        return path
    if (path / "checkpoints" / "resume").is_dir():
        search_dir = path / "checkpoints" / "resume"
    elif (path / "resume").is_dir():
        search_dir = path / "resume"
    else:
        search_dir = path
    candidates = []
    for candidate in search_dir.glob("checkpoint-*.pt"):
        try:
            step = int(candidate.stem.split("-")[-1])
        except ValueError:
            continue
        candidates.append((step, candidate))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-*.pt files in {search_dir}")
    return max(candidates)[1]


def prune_checkpoints(directory: Path, keep_last: int) -> None:
    checkpoints = []
    for path in directory.glob("checkpoint-*.pt"):
        try:
            checkpoints.append((int(path.stem.split("-")[-1]), path))
        except ValueError:
            continue
    for _step, path in sorted(checkpoints)[:-keep_last]:
        path.unlink()


def _metric_values(prediction, target, lpips_model):
    prediction_01 = prediction.add(1).div(2).clamp(0, 1)
    target_01 = target.add(1).div(2).clamp(0, 1)
    mse = (prediction_01 - target_01).pow(2).flatten(1).mean(1)
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    ssim = _ssim_map(prediction_01, target_01).flatten(1).mean(1)
    perceptual = lpips_model(prediction.float(), target.float()).flatten()
    return psnr, ssim, perceptual


@torch.no_grad()
def validate(model, dataloader, criterion, accelerator):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_criterion = accelerator.unwrap_model(criterion)
    previous_stage = unwrapped_model.stage
    previous_training = model.training
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    try:
        for batch in dataloader:
            conditioning = batch["conditioning_pixel_values"].to(accelerator.device)
            target = batch["target_pixel_values"].to(accelerator.device)
            tokens = batch["input_ids"].to(accelerator.device)
            rainy = batch["rainy_pixel_values"].to(accelerator.device)
            final = model(conditioning, tokens, deterministic=True)[:, PRELIMINARY_VIEW_INDEX]
            loss, _ = criterion(final, target)
            variants = {
                "rainy": rainy,
                "preliminary": conditioning[:, PRELIMINARY_VIEW_INDEX],
                "final": final,
            }
            batch_metrics = {}
            for name, prediction in variants.items():
                psnr, ssim, perceptual = _metric_values(
                    prediction, target, unwrapped_criterion.lpips_model
                )
                batch_metrics[name] = (psnr, ssim, perceptual)
                for metric_name, values in zip(("psnr", "ssim", "lpips"), batch_metrics[name]):
                    key = f"val/{name}_{metric_name}"
                    totals[key] = totals.get(key, 0.0) + values.sum().item()
            batch_size = target.shape[0]
            totals["val/loss"] = totals.get("val/loss", 0.0) + loss.item() * batch_size
            totals["val/psnr_wins"] = totals.get("val/psnr_wins", 0.0) + (
                batch_metrics["final"][0] > batch_metrics["preliminary"][0]
            ).sum().item()
            totals["val/lpips_wins"] = totals.get("val/lpips_wins", 0.0) + (
                batch_metrics["final"][2] < batch_metrics["preliminary"][2]
            ).sum().item()
            count += batch_size
    finally:
        if previous_training:
            unwrapped_model.set_stage(previous_stage)
        else:
            model.eval()
    metrics = {key: value / count for key, value in totals.items()}
    metrics["val/delta_psnr"] = metrics["val/final_psnr"] - metrics["val/preliminary_psnr"]
    metrics["val/delta_lpips"] = metrics["val/final_lpips"] - metrics["val/preliminary_lpips"]
    metrics["val/psnr_win_rate"] = metrics.pop("val/psnr_wins")
    metrics["val/lpips_win_rate"] = metrics.pop("val/lpips_wins")
    return metrics


def is_better(metrics: Dict[str, float], best: Optional[Dict[str, float]]) -> bool:
    if best is None:
        return True
    current_key = (metrics["val/final_psnr"], -metrics["val/final_lpips"], metrics["val/final_ssim"])
    best_key = (best["val/final_psnr"], -best["val/final_lpips"], best["val/final_ssim"])
    return current_key > best_key


def training_state(args, global_step: int, stage: str, best_metrics):
    return {
        "global_step": global_step,
        "stage": stage,
        "stage_step": global_step if stage == "A" else global_step - args.stage_a_steps,
        "stage_a_steps": args.stage_a_steps,
        "stage_b_steps": args.stage_b_steps,
        "best_metrics": best_metrics,
    }


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("Dynamic two-stage optimizer is currently supported only on one GPU")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    resume_dir = checkpoint_dir / "resume"
    if accelerator.is_main_process:
        resume_dir.mkdir(parents=True, exist_ok=True)

    resolved_prompt = resolve_prompt(args.dataset_path, args.prompt)
    resolved_view_order = resolve_view_order(args.dataset_path)
    resume_path = checkpoint_to_resume(args.resume)
    metadata = read_checkpoint_metadata(resume_path) if resume_path else None
    if metadata and metadata["checkpoint_kind"] != "training":
        raise ValueError("--resume requires a training checkpoint, not a weights-only checkpoint")
    if metadata and metadata["config"]["prompt"] != resolved_prompt:
        raise ValueError("Resolved dataset/CLI prompt differs from the checkpoint prompt")
    if metadata:
        saved_view_order = metadata["config"].get("view_order", "rainy_preliminary")
        if saved_view_order != resolved_view_order:
            raise ValueError(
                f"Checkpoint view_order={saved_view_order!r} differs from "
                f"dataset view_order={resolved_view_order!r}"
            )
    if metadata:
        saved_state = metadata["training_state"]
        for key in ("stage_a_steps", "stage_b_steps"):
            if int(saved_state[key]) != int(getattr(args, key)):
                raise ValueError(f"Checkpoint {key}={saved_state[key]} differs from requested {getattr(args, key)}")
        expected_groups = (
            ["unet_lora"]
            if saved_state["stage"] == "A"
            else ["unet_lora", "vae_decoder_lora_skip"]
        )
        if saved_state["optimizer_groups"] != expected_groups:
            raise ValueError(
                f"Checkpoint stage/group mismatch: stage={saved_state['stage']}, "
                f"groups={saved_state['optimizer_groups']}"
            )
    initial_stage = metadata["training_state"]["stage"] if metadata else "A"
    global_step = int(metadata["training_state"]["global_step"]) if metadata else 0
    best_metrics = metadata["training_state"].get("best_metrics") if metadata else None
    total_steps = args.stage_a_steps + args.stage_b_steps

    model = Difix(
        args.base_model, args.lora_rank_unet, args.lora_rank_vae, args.timestep,
        prompt=resolved_prompt, cfg_scale=args.cfg_scale,
        view_order=resolved_view_order,
    )
    optimizer = make_optimizer(model, args, initial_stage)
    lr_scheduler = make_lr_scheduler(optimizer, args, initial_stage)
    if resume_path:
        loaded_state = load_checkpoint(resume_path, model, optimizer, lr_scheduler)
        global_step = int(loaded_state["global_step"])
        best_metrics = loaded_state.get("best_metrics")

    transitioned_on_resume = False
    if model.stage == "A" and global_step >= args.stage_a_steps:
        optimizer = transition_to_stage_b(model, optimizer, args)
        lr_scheduler = make_lr_scheduler(optimizer, args, "B")
        transitioned_on_resume = True

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention:
        model.unet.enable_xformers_memory_efficient_attention()

    train_dataset = PairedDataset(
        args.dataset_path, "train", tokenizer=model.tokenizer,
        reference_dropout_prob=args.reference_dropout,
        clean_identity_prob=args.clean_identity, prompt_override=resolved_prompt,
    )
    val_dataset = PairedDataset(
        args.dataset_path, "test", tokenizer=model.tokenizer,
        horizontal_flip_prob=0, reference_dropout_prob=0, clean_identity_prob=0,
        prompt_override=resolved_prompt,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.dataloader_num_workers, pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    criterion = DerainLoss(lpips.LPIPS(net="vgg"))
    model, optimizer, lr_scheduler, train_loader, val_loader, criterion = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader, val_loader, criterion
    )
    if args.report_to != "none":
        accelerator.init_trackers(
            args.tracker_project_name,
            config={
                **vars(args),
                "resolved_prompt": resolved_prompt,
                "resolved_view_order": resolved_view_order,
            },
            init_kwargs={args.report_to: {"name": args.tracker_run_name}},
        )
    tokenizer = accelerator.unwrap_model(model).tokenizer
    empty_tokens = tokenizer(
        "", max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    ).input_ids.to(accelerator.device)

    if transitioned_on_resume and accelerator.is_main_process:
        save_checkpoint(
            resume_dir / f"checkpoint-{global_step}.pt", accelerator.unwrap_model(model),
            optimizer, lr_scheduler, global_step,
            training_state(args, global_step, "B", best_metrics),
        )
        prune_checkpoints(resume_dir, args.checkpoints_total_limit)

    progress = tqdm(total=total_steps, initial=global_step, disable=not accelerator.is_local_main_process)
    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break
            with accelerator.accumulate(model):
                conditioning = batch["conditioning_pixel_values"].to(accelerator.device)
                target = batch["target_pixel_values"].to(accelerator.device)
                tokens = batch["input_ids"].to(accelerator.device)
                if args.prompt_dropout > 0:
                    drop = torch.rand(tokens.shape[0], device=tokens.device) < args.prompt_dropout
                    tokens = tokens.clone()
                    tokens[drop] = empty_tokens.expand(tokens.shape[0], -1)[drop]
                prediction = model(
                    conditioning, tokens, cfg_scale=args.cfg_scale,
                    empty_prompt_tokens=empty_tokens.expand(tokens.shape[0], -1),
                )[:, PRELIMINARY_VIEW_INDEX]
                loss, terms = criterion(prediction, target)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            progress.update(1)
            logs = {"train/loss": loss.detach().item(), "train/stage": 0 if accelerator.unwrap_model(model).stage == "A" else 1}
            logs.update({f"train/{key}": value.detach().item() for key, value in terms.items()})
            for group in _base_optimizer(optimizer).param_groups:
                logs[f"train/lr_{group.get('name', 'unknown')}"] = group["lr"]
            if args.report_to != "none":
                accelerator.log(logs, step=global_step)

            metrics = None
            if args.eval_freq > 0 and global_step % args.eval_freq == 0:
                metrics = validate(model, val_loader, criterion, accelerator)
                if args.report_to != "none":
                    accelerator.log(metrics, step=global_step)
                if is_better(metrics, best_metrics):
                    best_metrics = {**metrics, "global_step": global_step}
                    if accelerator.is_main_process:
                        save_checkpoint(
                            checkpoint_dir / "best.pt", accelerator.unwrap_model(model),
                            None, None, global_step,
                            training_state(args, global_step, accelerator.unwrap_model(model).stage, best_metrics),
                            checkpoint_kind="weights",
                        )

            if global_step == args.stage_a_steps and accelerator.unwrap_model(model).stage == "A":
                if accelerator.is_main_process:
                    save_checkpoint(
                        checkpoint_dir / "stage-a-final.pt", accelerator.unwrap_model(model),
                        optimizer, lr_scheduler, global_step,
                        training_state(args, global_step, "A", best_metrics),
                    )
                base_optimizer = transition_to_stage_b(accelerator.unwrap_model(model), optimizer, args)
                new_scheduler = make_lr_scheduler(base_optimizer, args, "B")
                lr_scheduler = accelerator.prepare_scheduler(new_scheduler)
                if accelerator.is_main_process:
                    save_checkpoint(
                        resume_dir / f"checkpoint-{global_step}.pt", accelerator.unwrap_model(model),
                        optimizer, lr_scheduler, global_step,
                        training_state(args, global_step, "B", best_metrics),
                    )
                    prune_checkpoints(resume_dir, args.checkpoints_total_limit)
                continue

            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                stage = accelerator.unwrap_model(model).stage
                save_checkpoint(
                    resume_dir / f"checkpoint-{global_step}.pt", accelerator.unwrap_model(model),
                    optimizer, lr_scheduler, global_step,
                    training_state(args, global_step, stage, best_metrics),
                )
                prune_checkpoints(resume_dir, args.checkpoints_total_limit)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        stage = accelerator.unwrap_model(model).stage
        save_checkpoint(
            resume_dir / f"checkpoint-{global_step}.pt", accelerator.unwrap_model(model),
            optimizer, lr_scheduler, global_step,
            training_state(args, global_step, stage, best_metrics),
        )
        prune_checkpoints(resume_dir, args.checkpoints_total_limit)
        save_checkpoint(
            checkpoint_dir / "final.pt", accelerator.unwrap_model(model),
            None, None, global_step,
            training_state(args, global_step, stage, best_metrics),
            checkpoint_kind="weights",
        )
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
