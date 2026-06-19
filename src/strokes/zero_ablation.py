import torch


def zero_ablation_layer_hook(value: torch.Tensor, hook) -> torch.Tensor:
    """Ablate an entire layer by setting all its activations to zero."""
    print(f"Applying zero ablation to entire layer at: {hook.name}")
    value[...] = 0.0
    return value


def zero_ablation_head_hook(
    value: torch.Tensor, hook, head_idx: int
) -> torch.Tensor:
    """Ablate a specific attention head within a layer by setting it to zero."""
    print(f"Applying zero ablation to head {head_idx} at: {hook.name}")
    # Shape of value in TransformerLens attention hook is typically:
    # [batch, position, head_index, d_head]
    value[:, :, head_idx, :] = 0.0
    return value
