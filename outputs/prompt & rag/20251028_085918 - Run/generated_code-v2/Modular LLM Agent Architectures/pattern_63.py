class MemoryModule:
    def __init__(self):
        self.short_term_memory = {}
        self.long_term_memory = []

    def add_short_term_memory(self, key, value):
        """Adds or updates an item in short-term memory."""
        self.short_term_memory[key] = value
        print(f"[Memory] Added/Updated short-term memory: {key} -> {value}")

    def get_short_term_memory(self, key):
        """Retrieves an item from short-term memory."""
        return self.short_term_memory.get(key)

    def add_long_term_memory(self, item):
        """Adds an item to long-term memory (e.g., past preferences, completed tasks)."""
        self.long_term_memory.append(item)
        print(f"[Memory] Added long-term memory: {item}")

    def get_long_term_memory(self):
        """Retrieves all items from long-term memory."""
        return self.long_term_memory

    def clear_short_term_memory(self):
        """Clears short-term memory."""
        self.short_term_memory = {}
        print("[Memory] Short-term memory cleared.")

    def __str__(self):
        return f"Short-term: {self.short_term_memory}\nLong-term: {self.long_term_memory}"