import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class AudioDeafnessEvaluator:
    """Evaluates Whisper speech-to-text models to quantify cortical deafness and omission symptoms."""

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
    ):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

    def evaluate_audio_array(
        self, audio_array: np.ndarray, sampling_rate: int = 16000
    ) -> dict:
        """Transcribe an audio array and measure output characteristics post-stroke."""
        print("Processing audio waveform array for speech recognition.")

        # Extract features and convert to tensor mapped to the active device
        input_features = self.processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(self.device)

        # Enforce data type alignment matching the target model weights to avoid OOM
        target_dtype = next(self.model.parameters()).dtype
        input_features = input_features.to(target_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(input_features)

        # Decode token predictions back to human-readable text strings
        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        result = {
            "transcription": transcription,
            "token_count": len(generated_ids[0]),
        }

        print(f"Transcription result: '{result['transcription']}'")
        return result
