import json
from pathlib import Path
from PIL import Image


def initialize_directories(root_path: Path):
    """Create necessary data and output directories if they do not exist."""
    required_dirs = [
        root_path / "data" / "vision_samples",
        root_path / "data" / "audio_samples",
        root_path / "outputs" / "plots",
        root_path / "outputs" / "metrics",
    ]

    for target_dir in required_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Verified directory existence: {target_dir}")


def create_dummy_vision_asset(root_path: Path):
    """Generate a basic dummy image for spatial neglect baseline testing."""
    target_path = root_path / "data" / "vision_samples" / "sample.jpg"

    if not target_path.exists():
        print(f"Generating dummy visual asset at: {target_path}")
        # Create a standard 224x224 RGB image matching ViT input expectations
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img.save(target_path)
        print("Dummy visual asset created successfully.")
    else:
        print(f"Visual asset already exists at: {target_path}")


def main():
    """Orchestrate the initialization of mock data assets for the repository."""
    print("Starting repository asset setup sequence...")
    project_root = Path(__file__).resolve().parents[1]

    initialize_directories(project_root)
    create_dummy_vision_asset(project_root)

    print("Asset setup sequence finalized successfully.")


if __name__ == "__main__":
    main()
