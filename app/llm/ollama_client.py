import requests


class OllamaClient:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    @property
    def base_url(self) -> str:
        if self.url.endswith("/api/chat"):
            return self.url[: -len("/api/chat")]
        return self.url.rsplit("/api", 1)[0] if "/api" in self.url else self.url

    def chat(self, messages):
        r = requests.post(
            self.url,
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def complete(self, prompt: str) -> str:
        """Single-prompt completion helper using chat endpoint."""
        return self.chat([{"role": "user", "content": prompt}])
