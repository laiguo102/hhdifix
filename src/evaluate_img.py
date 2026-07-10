"""Evaluate rainy, preliminary and final outputs against clean ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lpips
import torch
from PIL import Image
from torchvision.transforms import functional as TF

try:
    from dataset import _files_by_stem
    from loss import _ssim_map
except ImportError:
    from .dataset import _files_by_stem
    from .loss import _ssim_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rainy_dir", required=True)
    parser.add_argument("--preliminary_dir", required=True)
    parser.add_argument("--final_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output", default="derain_metrics.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _tensor(path: Path, stem: str):
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (512, 512):
            raise ValueError(f"'{stem}' must be RGB 512x512, got {image.mode} {image.size}: {path}")
        return TF.to_tensor(image).unsqueeze(0)


def _metrics(prediction, target, lpips_metric):
    mse = torch.mean((prediction - target) ** 2).item()
    psnr = -10 * math.log10(max(mse, 1e-12))
    ssim = _ssim_map(prediction, target).mean().item()
    perceptual = lpips_metric(prediction.mul(2).sub(1), target.mul(2).sub(1)).mean().item()
    return {"psnr": psnr, "ssim": ssim, "lpips": perceptual}


def main(args):
    directories = {
        "rainy": _files_by_stem(args.rainy_dir),
        "preliminary": _files_by_stem(args.preliminary_dir),
        "final": _files_by_stem(args.final_dir),
        "gt": _files_by_stem(args.gt_dir),
    }
    expected = set(directories["gt"])
    for name, files in directories.items():
        if set(files) != expected:
            raise ValueError(f"{name} stems differ from GT")
    device = torch.device(args.device)
    metric = lpips.LPIPS(net="alex").eval().requires_grad_(False).to(device)
    samples = []
    with torch.no_grad():
        for stem in sorted(expected):
            target = _tensor(directories["gt"][stem], stem).to(device)
            row = {"stem": stem}
            for name in ("rainy", "preliminary", "final"):
                prediction = _tensor(directories[name][stem], stem).to(device)
                row[name] = _metrics(prediction, target, metric)
            row["delta_psnr_vs_preliminary"] = row["final"]["psnr"] - row["preliminary"]["psnr"]
            row["delta_lpips_vs_preliminary"] = row["final"]["lpips"] - row["preliminary"]["lpips"]
            samples.append(row)

    summary = {}
    for name in ("rainy", "preliminary", "final"):
        summary[name] = {
            key: sum(row[name][key] for row in samples) / len(samples)
            for key in ("psnr", "ssim", "lpips")
        }
    summary["delta_psnr_vs_preliminary"] = sum(row["delta_psnr_vs_preliminary"] for row in samples) / len(samples)
    summary["delta_lpips_vs_preliminary"] = sum(row["delta_lpips_vs_preliminary"] for row in samples) / len(samples)
    summary["psnr_win_rate_vs_preliminary"] = sum(
        row["delta_psnr_vs_preliminary"] > 0 for row in samples
    ) / len(samples)
    summary["lpips_win_rate_vs_preliminary"] = sum(
        row["delta_lpips_vs_preliminary"] < 0 for row in samples
    ) / len(samples)
    result = {"summary": summary, "samples": samples}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_args())
