import os
import tempfile
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

class WhisperSTT:
    def __init__(self, model_name: str):
        self.model = WhisperModel(model_name, device="cuda")

    def transcribe(self, audio_float32, sample_rate: int) -> str:
        audio_int16 = np.clip(audio_float32 * 32767, -32768, 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wavfile.write(f.name, sample_rate, audio_int16)
            path = f.name

        try:
            segments, _ = self.model.transcribe(path, vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            os.unlink(path)
