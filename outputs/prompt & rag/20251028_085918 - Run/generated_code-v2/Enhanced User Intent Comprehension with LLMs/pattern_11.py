from collections import deque

class ContextManager:
    def __init__(self, max_history=5):
        self.conversations = {}
        self.max_history = max_history

    def add_message(self, user_id: str, message: str, sender: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = deque(maxlen=self.max_history)
        self.conversations[user_id].append({"sender": sender, "message": message})

    def get_history(self, user_id: str) -> list:
        return list(self.conversations.get(user_id, []))

    def clear_history(self, user_id: str):
        if user_id in self.conversations:
            del self.conversations[user_id]
