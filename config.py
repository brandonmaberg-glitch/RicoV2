from dataclasses import dataclass

@dataclass
class Config:
    # Models
    ollama_model: str = "qwen2.5:7b-instruct"          # swap to 14B when you want
    piper_model_onnx: str = "en_GB-alan-medium.onnx"
    whisper_model: str = "distil-large-v3"

    # Endpoints
    ollama_url: str = "http://localhost:11434/api/chat"

    # Audio / PTT
    sample_rate: int = 16000
    max_seconds: int = 8
    ptt_key: str = "space"
    quit_key: str = "esc"

    # Prompt
    system_prompt: str = (
        "You are RICO (Really Intelligent Car Operator): a calm, precise British-butler AI. "
        "Be concise, ask for specifics when needed, and keep replies conversational."
    )
