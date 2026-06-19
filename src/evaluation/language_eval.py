import json
from pathlib import Path
import torch
from transformer_lens import HookedTransformer


class LanguageAphasiaEvaluator:
    """Evaluates language model outputs to quantify aphasia-like symptoms via logit metrics."""

    def __init__(self, model: HookedTransformer):
        self.model = model

    def calculate_logit_difference(self, sample: dict) -> dict:
        """Compute logit scores and the relative difference for clean vs corrupted tokens."""
        prompt = sample["prompt"]
        clean_target = sample["clean_target"]
        corrupted_target = sample["corrupted_target"]

        # Run forward pass to extract terminal predictions without updating gradients
        with torch.no_grad():
            logits = self.model(prompt)

        # Extract the logits of the very last predicted token position
        # Shape: [batch, position, vocab_size] -> [vocab_size]
        last_token_logits = logits[0, -1, :]

        # Convert target strings to single token integers mapping to vocabulary
        clean_token_id = self.model.to_single_token(clean_target)
        corrupted_token_id = self.model.to_single_token(corrupted_target)

        clean_logit = last_token_logits[clean_token_id].item()
        corrupted_logit = last_token_logits[corrupted_id].item()
        logit_diff = clean_logit - corrupted_logit

        return {
            "prompt": prompt,
            "type": sample["type"],
            "clean_logit": clean_logit,
            "corrupted_logit": corrupted_logit,
            "logit_difference": logit_diff,
        }

    def run_batch_evaluation(self, json_data_path: Path) -> list:
        """Run standard evaluation over a batch of diagnostic text samples."""
        print(f"Loading diagnostic metrics from: {json_data_path}")
        with open(json_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for sample in data["samples"]:
            metrics = self.calculate_logit_difference(sample)
            results.append(metrics)
            print(
                f"Type: {metrics['type']} | Logit Diff: {metrics['logit_difference']:.4f}"
            )

        return results
