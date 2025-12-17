from typing import Dict
from llm_interface import AbstractLLM

class LLMManager:
    """
    Manages multiple LLM providers and allows dynamic switching between them.
    """
    def __init__(self, llm_providers: Dict[str, AbstractLLM], default_llm: str):
        self._llm_providers = llm_providers
        if default_llm not in self._llm_providers:
            raise ValueError(f"Default LLM \'{default_llm}\' not found in providers.")
        self._current_llm_key = default_llm
        print(f"LLMManager initialized with default LLM: {self._current_llm_key}")

    def set_current_llm(self, model_key: str):
        """
        Sets the active LLM provider by its key.
        """
        if model_key in self._llm_providers:
            self._current_llm_key = model_key
            print(f"Switched current LLM to: {self._current_llm_key}")
        else:
            print(f"Warning: LLM provider \'{model_key}\' not found. Keeping current LLM: {self._current_llm_key}")

    def get_current_llm(self) -> AbstractLLM:
        """
        Returns the currently active LLM instance.
        """
        return self._llm_providers[self._current_llm_key]

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generates a response using the currently active LLM.
        """
        print(f"[Manager] Using {self._current_llm_key} for response...")
        return self.get_current_llm().generate_response(prompt, **kwargs)

    def switch_llm_dynamically(self, prompt: str, cost_preference: bool = False, speed_preference: bool = False):
        """
        A simplified example of dynamic LLM switching based on preferences or prompt content.
        In a real system, this would involve more complex logic (e.g., token count, sentiment, routing rules).
        """
        print(f"[Manager] Considering dynamic LLM switch for prompt: \'{prompt[:50]}...\'")
        current_model_name = self.get_current_llm().get_model_name().lower()

        if cost_preference or "cost" in prompt.lower() or "expensive" in prompt.lower():
            # Example: Prefer Llama for potential lower cost/local execution
            if "llama" not in current_model_name:
                print("[Manager] Switching to Llama for cost efficiency.")
                self.set_current_llm("llama")
                return

        if speed_preference or "fast" in prompt.lower() or "latency" in prompt.lower():
            # Example: Prefer Gemini for potentially faster responses
            if "gemini" not in current_model_name:
                print("[Manager] Switching to Gemini for speed.")
                self.set_current_llm("gemini")
                return

        # Default or fallback logic if no specific switch criteria met
        # For example, if the current model is already optimal for the criteria, or if no criteria are met.
        print(f"[Manager] No dynamic switch needed. Staying with {self._current_llm_key}.")
