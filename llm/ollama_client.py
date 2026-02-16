import requests

class OllamaClient:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def chat(self, messages):
        r = requests.post(
            self.url,
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
