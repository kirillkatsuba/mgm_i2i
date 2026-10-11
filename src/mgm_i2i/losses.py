"""The same variational objective is used by both MGM players.

Equation (10), Midpoint Generative Models, arXiv:2605.29920v2.
Conditioning on c is our adaptation; f never observes the flip variable.
"""
import torch


def observation(fake, real, t, b, noise, noise_strength=0.01):
    # t and b have shape [batch, 1] or [batch, 1, 1, 1].
    # The target real-fake does NOT change sign when b changes.
    real_weight = torch.where(b.bool(), 1 - t, t)
    sigma = (noise_strength * t * (1 - t)).sqrt()
    return (1 - real_weight) * fake + real_weight * real + sigma * noise


def variational_loss(field, displacement):
    # Mean over pixels rather than sum: fixed positive rescaling of divergence.
    return (field * (2 * displacement - field)).flatten(1).mean(1).mean()


def field_loss(field, displacement):
    return (field - displacement).square().mean()


def random_bridge(fake, midpoint=False, flip=True):
    shape = (fake.shape[0],) + (1,) * (fake.ndim - 1)
    # The deliberately invalid no-flip ablation integrates the full [0, 1].
    t = torch.full(shape, 0.5, device=fake.device) if midpoint else torch.rand(shape, device=fake.device) * (0.5 if flip else 1.)
    b = torch.rand(shape, device=fake.device) < 0.5 if flip else torch.zeros(shape, device=fake.device, dtype=torch.bool)
    return t, b, torch.randn_like(fake)
