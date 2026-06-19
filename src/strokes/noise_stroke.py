import torch


def gaussian_noise_layer_hook(
    value: torch.Tensor, hook, noise_level: float = 1.0
) -> torch.Tensor:
    """Inject Gaussian noise into an entire layer to simulate hemorrhage."""
    print(
        f"Injecting Gaussian noise (std={noise_level}) to layer at: {hook.name}"
    )
    noise = torch.randn_like(value) * noise_level
    return value + noise


def gaussian_noise_head_hook(
    value: torch.Tensor, hook, head_idx: int, noise_level: float = 1.0
) -> torch.Tensor:
    """Inject Gaussian noise into a specific attention head to corrupt signal."""
    print(
        f"Injecting Gaussian noise (std={noise_level}) to head {head_idx} at: {hook.name}"
    )
    # Target only the designated head index to corrupt its semantic vector
    noise = torch.randn_like(value[:, :, head_idx, :]) * noise_level
    value[:, :, head_idx, :] += noise
    return value
