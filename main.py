from app.config import Config
from app.stt.whisper_stt import WhisperSTT
from app.llm.ollama_client import OllamaClient
from app.tts.piper_tts import PiperTTS
from app.memory.chat_memory import ChatMemory
from app.core.loop import run_loop

def main():
    cfg = Config()

    stt = WhisperSTT(cfg.whisper_model)
    llm = OllamaClient(cfg.ollama_url, cfg.ollama_model)
    tts = PiperTTS(cfg.piper_model_onnx)
    memory = ChatMemory(cfg.system_prompt)

    run_loop(cfg, stt, llm, tts, memory)

if __name__ == "__main__":
    main()
