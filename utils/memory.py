import gc
import torch


def flush_memory():
    """Force garbage collection and empty the PyTorch CUDA cache to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_vram_usage():
    """Print the current allocated and reserved CUDA memory in megabytes."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(
            f"VRAM Usage - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
        )
    else:
        print("CUDA is not available. Running on CPU.")
