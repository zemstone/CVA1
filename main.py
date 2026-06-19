import json
from pathlib import Path
from src.evaluation.language_eval import LanguageAphasiaEvaluator
from src.models.loader import ModelEnvironmentLoader
from utils.memory import flush_memory, print_vram_usage


def run_language_experiment_pipeline(loader: ModelEnvironmentLoader, data_path: Path):
    """Execute the full mechanistic language lesion pipeline inside a clean session."""
    print("Starting Language Lesion Experiment Session...")

    # Load the model inside the managed environment
    model = loader.load_language_model()
    evaluator = LanguageAphasiaEvaluator(model)

    print("Running baseline evaluation before inducing stroke...")
    baseline_results = evaluator.run_batch_evaluation(data_path)

    # 1. Induce stroke simulation via hooks (Simulating Broca's aphasia)
    print("Inducing target stroke simulation on linguistic processing units...")
    target_layer = 5
    target_head = 8

    # Apply zero ablation hook to the attention head using TransformerLens syntax
    model.add_hook(
        f"blocks.{target_layer}.attn.hook_z",
        lambda val, hook: val[:, :, target_head, :].fill_(0.0),
    )

    print("Running post-stroke evaluation to measure programmatic aphasia...")
    post_stroke_results = evaluator.run_batch_evaluation(data_path)

    # Remove all active hooks to restore base model state before memory release
    model.reset_hooks()


def main():
    """Main entrypoint orchestrating cross-modal stroke simulations under VRAM limits."""
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "text_samples.json"

    # Define minimal environment configuration manually to avoid breaking dependencies
    language_config = {
        "model": {"model_name": "gpt2", "torch_dtype": "bfloat16", "device": "cuda"}
    }

    print("Initializing Automated AI Stroke Simulation Pipeline")
    print_vram_usage()

    # Instantiate the master environment controller
    env_loader = ModelEnvironmentLoader(language_config)

    # Run language experiments under strict session isolation to prevent OOM
    with env_loader.experiment_session():
        run_language_experiment_pipeline(env_loader, data_path)

    print("All scheduled cross-modal stroke simulations completed successfully.")


if __name__ == "__main__":
    main()
