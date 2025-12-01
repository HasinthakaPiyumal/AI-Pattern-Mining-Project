
class WorkingMemory:
    def __init__(self):
        self.conversation_history = []
        self.current_context = {}

    def add_message(self, role, message):
        self.conversation_history.append({"role": role, "content": message})

    def get_history(self):
        return self.conversation_history

    def update_context(self, key, value):
        self.current_context[key] = value

    def get_context(self, key=None):
        if key:
            return self.current_context.get(key)
        return self.current_context

    def clear_memory(self):
        self.conversation_history = []
        self.current_context = {}

    def __str__(self):
        return f"History: {self.conversation_history}\nContext: {self.current_context}"
