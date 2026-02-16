from app.config import Config
from app.core.loop import run_loop
from app.llm.ollama_client import OllamaClient
from app.memory.service import MemoryService
from app.stt.whisper_stt import WhisperSTT
from app.tts.piper_tts import PiperTTS


def main():
    cfg = Config()

    stt = WhisperSTT(cfg.whisper_model)
    llm = OllamaClient(cfg.ollama_url, cfg.ollama_model)
    tts = PiperTTS(cfg.piper_model_onnx)
    memory_service = MemoryService(cfg, llm)

    run_loop(cfg, stt, llm, tts, memory_service)


if __name__ == "__main__":
    main()
