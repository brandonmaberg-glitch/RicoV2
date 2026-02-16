import time
import numpy as np
import sounddevice as sd
import keyboard

def record_while_held(sample_rate: int, max_seconds: int, ptt_key: str):
    frames = []
    start = time.time()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while keyboard.is_pressed(ptt_key):
            if time.time() - start > max_seconds:
                break
            data, _ = stream.read(1024)
            frames.append(data.copy())

    if not frames:
        return None

    return np.concatenate(frames, axis=0).squeeze()
