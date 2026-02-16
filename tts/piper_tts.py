import os
import tempfile
import subprocess
import sounddevice as sd
from scipy.io import wavfile

class PiperTTS:
    def __init__(self, model_onnx: str):
        self.model_onnx = model_onnx

    def speak(self, text: str):
        if not text.strip():
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            out_wav = f.name

        try:
            subprocess.run(
                ["piper", "--model", self.model_onnx, "--output_file", out_wav],
                input=text.encode("utf-8"),
                check=True
            )
            sr, audio = wavfile.read(out_wav)
            sd.play(audio, sr)
            sd.wait()
        finally:
            try:
                os.unlink(out_wav)
            except OSError:
                pass
