import torch

from src.loss import charbonnier_loss, sobel_loss, ssim_loss


def test_reconstruction_losses_are_finite_and_differentiable():
    prediction = torch.zeros(2, 3, 32, 32, requires_grad=True)
    target = torch.ones_like(prediction) * 0.2
    loss = charbonnier_loss(prediction, target) + ssim_loss(prediction, target) + sobel_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_matching_images_have_zero_structural_and_edge_loss():
    image = torch.rand(1, 3, 32, 32)
    assert torch.allclose(ssim_loss(image, image), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(sobel_loss(image, image), torch.tensor(0.0))
