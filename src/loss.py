"""Losses used by the rain-removal training objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier_loss(prediction, target, epsilon: float = 1e-3):
    return torch.sqrt((prediction - target).pow(2) + epsilon**2).mean()


def _ssim_map(x, y, window_size: int = 11):
    padding = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, padding) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y * y, window_size, 1, padding) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )


def ssim_loss(prediction, target):
    # SSIM constants assume [0, 1].
    prediction = prediction.add(1).div(2).clamp(0, 1)
    target = target.add(1).div(2).clamp(0, 1)
    return 1.0 - _ssim_map(prediction, target).mean()


def sobel_loss(prediction, target):
    dtype, device = prediction.dtype, prediction.device
    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    channels = prediction.shape[1]
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    pred_x = F.conv2d(prediction, kernel_x, padding=1, groups=channels)
    pred_y = F.conv2d(prediction, kernel_y, padding=1, groups=channels)
    target_x = F.conv2d(target, kernel_x, padding=1, groups=channels)
    target_y = F.conv2d(target, kernel_y, padding=1, groups=channels)
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


class DerainLoss(torch.nn.Module):
    def __init__(self, lpips_model):
        super().__init__()
        self.lpips_model = lpips_model.eval().requires_grad_(False)

    def forward(self, prediction, target):
        terms = {
            "charbonnier": charbonnier_loss(prediction, target),
            "ssim": ssim_loss(prediction, target),
            "lpips": self.lpips_model(prediction.float(), target.float()).mean(),
            "sobel": sobel_loss(prediction, target),
        }
        total = terms["charbonnier"] + 0.2 * terms["ssim"] + 0.1 * terms["lpips"] + 0.05 * terms["sobel"]
        return total, terms
