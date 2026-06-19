import contextlib
import sys
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForImageClassification, WhisperForConditionalGeneration

# Add project root to sys.path to enable absolute imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.memory import flush_memory, print_vram_usage


class ModelEnvironmentLoader:
    """Manages secure model loading and VRAM lifecycle for 8GB hardware constraints."""

    def __init__(self, config: dict):
        self.config = config
        self.device = (
            "cuda"
            if torch.cuda.is_available() and config["model"]["device"] == "cuda"
            else "cpu"
        )
        self.dtype = (
            torch.bfloat16
            if config["model"]["torch_dtype"] == "bfloat16"
            else torch.float32
        )

    def load_language_model(self) -> HookedTransformer:
        """Load HookedTransformer for mechanistic interpretability with optimized dtype."""
        model_name = self.config["model"]["model_name"]
        print(f"Loading language model: {model_name} in {self.dtype}...")

        # Clear existing cache before allocation
        flush_memory()

        model = HookedTransformer.from_pretrained(
            model_name, hf_model=None, torch_dtype=self.dtype
        )
        model.to(self.device)

        print("Language model loaded successfully.")
        print_vram_usage()
        return model

    def load_vision_model(self, model_name: str):
        """Load a standard Vision Transformer under strict memory budget."""
        print(f"Loading vision model: {model_name}...")
        flush_memory()

        model = AutoModelForImageClassification.from_pretrained(
            model_name, torch_dtype=self.dtype
        )
        model.to(self.device)

        print("Vision model loaded successfully.")
        print_vram_usage()
        return model

    def load_audio_model(self, model_name: str):
        """Load a standard Whisper model under strict memory budget."""
        print(f"Loading audio model: {model_name}...")
        flush_memory()

        model = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype
        )
        model.to(self.device)

        print("Audio model loaded successfully.")
        print_vram_usage()
        return model

    @contextlib.contextmanager
    def experiment_session(self):
        """Context manager to ensure VRAM is completely cleaned after an experiment session."""
        try:
            print_vram_usage()
            yield self
        finally:
            print("Closing session. Cleaning up VRAM resources...")
            flush_memory()
            print_vram_usage()
