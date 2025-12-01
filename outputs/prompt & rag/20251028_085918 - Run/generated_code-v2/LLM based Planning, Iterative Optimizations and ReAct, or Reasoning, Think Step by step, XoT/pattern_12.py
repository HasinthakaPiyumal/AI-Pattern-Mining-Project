
import random

class LLMReasoningModel:
    """
    A mock LLM that generates reasoning steps for medical diagnosis.
    In a real application, this would be an actual LLM (e.g., GPT-4, Llama).
    """
    def __init__(self):
        self.possible_steps = [
            "patient presents with fever",
            "patient presents with cough",
            "patient presents with headache",
            "patient presents with sore throat",
            "patient reports muscle aches",
            "consider influenza as a possibility",
            "consider common cold as a possibility",
            "rule out bacterial infection",
            "recommend a flu test",
            "prescribe antiviral medication if flu is confirmed",
            "advise rest and hydration",
            "bacterial pneumonia is unlikely",
            "viral pneumonia is a possibility",
            "antibiotics are not effective against viral infections",
            "patient has elevated temperature",
            "patient has respiratory symptoms",
            "patient has systemic symptoms",
            "inflammation markers are elevated",
        ]

    def generate_next_step(self, current_reasoning_path: list[str], symptoms: list[str]) -> str:
        """
        Generates a plausible next reasoning step based on the current path and symptoms.
        This is a highly simplified simulation.
        """
        # Prioritize steps related to current symptoms
        relevant_steps = []
        for symptom in symptoms:
            for step in self.possible_steps:
                if symptom.lower() in step.lower() and step not in current_reasoning_path:
                    relevant_steps.append(step)
        
        # Add some general diagnostic steps
        general_steps = [step for step in self.possible_steps if "consider" in step or "rule out" in step or "recommend" in step]
        
        # Combine and pick randomly, or with some basic logic
        if relevant_steps:
            return random.choice(relevant_steps)
        elif general_steps:
            return random.choice(general_steps)
        else:
            # If no specific relevant steps, pick a random one not already in the path
            available_steps = [step for step in self.possible_steps if step not in current_reasoning_path]
            return random.choice(available_steps) if available_steps else ""

    def generate_multiple_paths(self, symptoms: list[str], num_paths: int = 3, steps_per_path: int = 5) -> list[list[str]]:
        """
        Generates multiple divergent reasoning paths for Monte-Carlo planning.
        """
        all_paths = []
        for _ in range(num_paths):
            current_path = []
            for _ in range(random.randint(3, steps_per_path)): # Vary path length slightly
                next_step = self.generate_next_step(current_path, symptoms)
                if next_step and next_step not in current_path:
                    current_path.append(next_step)
                else:
                    break # Stop if no new valid step can be generated
            if current_path:
                all_paths.append(current_path)
        return all_paths
