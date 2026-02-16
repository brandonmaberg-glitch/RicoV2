from dataclasses import dataclass, field


@dataclass
class Config:
    # Models
    ollama_model: str = "qwen2.5:7b-instruct"
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

    # Local memory config
    memory_db_path: str = ".rico/memory.sqlite3"
    memory_stm_ttl_hours: int = 48
    memory_top_n: int = 12
    memory_use_llm_judge: bool = False
    memory_use_llm_summary: bool = False
    memory_summary_update_every: int = 4

    embeddings_backend: str = "sbert"  # sbert|ollama
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_model: str = "nomic-embed-text"

    memory_scoring_weights: dict[str, float] = field(
        default_factory=lambda: {
            "similarity": 0.55,
            "importance": 0.20,
            "recency": 0.15,
            "frequency": 0.10,
        }
    )

    # Compatibility aliases from requested env-style names
    @property
    def MEMORY_DB_PATH(self) -> str:
        return self.memory_db_path

    @property
    def MEMORY_STM_TTL_HOURS(self) -> int:
        return self.memory_stm_ttl_hours

    @property
    def MEMORY_TOP_N(self) -> int:
        return self.memory_top_n

    @property
    def MEMORY_USE_LLM_JUDGE(self) -> bool:
        return self.memory_use_llm_judge

    @property
    def MEMORY_USE_LLM_SUMMARY(self) -> bool:
        return self.memory_use_llm_summary

    @property
    def EMBEDDINGS_BACKEND(self) -> str:
        return self.embeddings_backend

    @property
    def MEMORY_SCORING_WEIGHTS(self) -> dict[str, float]:
        return self.memory_scoring_weights

    @property
    def ollama_base_url(self) -> str:
        if self.ollama_url.endswith("/api/chat"):
            return self.ollama_url[: -len("/api/chat")]
        return self.ollama_url.rsplit("/api", 1)[0] if "/api" in self.ollama_url else self.ollama_url
