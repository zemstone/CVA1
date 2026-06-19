# AI Stroke Experiment Framework

This repository contains an experimental framework designed to simulate artificial localized strokes (neurolesions) inside deep learning neural networks across multiple modalities (Language, Vision, Audio). The framework is strictly optimized to run sequentially within a limited 8GB VRAM budget.

## Hardware Constraints and Mitigations

To prevent Out-Of-Memory (OOM) errors on 8GB VRAM hardware, this project implements:
1. **Dynamic Memory Flushing**: Explicit garbage collection and CUDA cache clearing between session transitions.
2. **Precision Management**: Models are loaded using `bfloat16` to halve memory footprint while maintaining activation scale.
3. **Activation-Level Hooks**: Interventions (ablation, noise injection) are applied directly to forward pass tensors without altering model weights or allocating additional parameter gradients.

---

## Installation

1. Clone the repository and navigate to the root directory.
2. Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
