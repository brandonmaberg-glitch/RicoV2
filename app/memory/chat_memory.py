class ChatMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def get(self):
        return self.messages
