from pathlib import Path
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


class VisionNeglectEvaluator:
    """Evaluates Vision Transformer models to quantify spatial neglect and visual deficits."""

    def __init__(
        self, model: ViTForImageClassification, processor: ViTImageProcessor
    ):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

    def apply_spatial_mask(
        self, image: Image.Image, side: str = "left"
    ) -> Image.Image:
        """Physically mask half of the image to compare with mechanistic ablation results."""
        width, height = image.size
        masked_image = image.copy()
        pixels = masked_image.load()

        if side == "left":
            for x in range(0, width // 2):
                for y in range(0, height):
                    pixels[x, y] = (0, 0, 0)
        elif side == "right":
            for x in range(width // 2, width):
                for y in range(0, height):
                    pixels[x, y] = (0, 0, 0)

        return masked_image

    def evaluate_image(self, image_path: Path, mask_side: str = None) -> dict:
        """Evaluate model classification performance on a clean or spatially masked image."""
        print(f"Opening visual sample for evaluation: {image_path.name}")
        image = Image.open(image_path).convert("RGB")

        if mask_side in ["left", "right"]:
            print(
                f"Applying physical sensory neglect mask on the {mask_side} side."
            )
            image = self.apply_spatial_mask(image, side=mask_side)

        # Preprocess image and move tensors to target device
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_label = torch.max(probabilities, dim=-1)

        result = {
            "predicted_label": int(top_label.item()),
            "confidence": float(top_prob.item()),
        }

        print(
            f"Result - Label ID: {result['predicted_label']} | Confidence: {result['confidence']:.4f}"
        )
        return result
