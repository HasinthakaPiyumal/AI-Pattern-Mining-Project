
class WorkingMemory:
    """
    A module to track and store all essential information related to the current interaction
    for an AI agent.
    """
    def __init__(self):
        self.memory = {
            "user_query": None,
            "external_evidence": [],
            "llm_responses": [],
            "utility_scores": [],
            "verbalized_feedback": [],
            "dialog_history": []
        }

    def update_memory(self, key, value):
        """
        Updates a specific key in the working memory.
        """
        if key in self.memory:
            if isinstance(self.memory[key], list) and not isinstance(value, list):
                self.memory[key].append(value)
            elif isinstance(self.memory[key], list) and isinstance(value, list):
                self.memory[key].extend(value)
            else:
                self.memory[key] = value
        else:
            print(f"Warning: Key '{key}' not a predefined memory slot. Adding as new key.")
            self.memory[key] = value

    def get_memory(self, key=None):
        """
        Retrieves information from the working memory.
        If no key is provided, returns the entire memory.
        """
        if key:
            return self.memory.get(key)
        return self.memory

    def add_dialog_turn(self, speaker, text):
        """
        Adds a new turn to the dialog history.
        """
        self.memory["dialog_history"].append({"speaker": speaker, "text": text})

    def clear_memory(self):
        """
        Clears the entire working memory, resetting it to its initial state.
        """
        self.memory = {
            "user_query": None,
            "external_evidence": [],
            "llm_responses": [],
            "utility_scores": [],
            "verbalized_feedback": [],
            "dialog_history": []
        }

    def __str__(self):
        return str(self.memory)

