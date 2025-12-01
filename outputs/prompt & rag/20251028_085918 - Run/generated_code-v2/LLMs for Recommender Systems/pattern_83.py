import os

class LLMAgent:
    def __init__(self, api_key: str = None, model_name: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found. LLM interactions will be simulated.")

    def _format_architecture_description(self, architecture: str) -> str:
        """Helper to format an architecture string for LLM prompts."""
        return f"""<architecture>
{architecture}
</architecture>"""

    def generate_architecture_prompt(self, problem_description: str) -> str:
        """Generates a prompt for the LLM to create an initial ML architecture.

        Args:
            problem_description: A description of the ML problem (e.g., "e-commerce product recommendation").
        Returns:
            A string prompt.
        """
        return f"""You are an expert in designing machine learning architectures for recommender systems.
Given the following problem description, propose a novel and effective ML model architecture.
Focus on components like embedding layers, feature interaction mechanisms, neural network structure (e.g., number of layers, activation functions), and output layers.
Represent the architecture as a concise string or a JSON-like structure. Be creative but practical.

Problem Description: {problem_description}

Proposed Architecture:"""

    def optimize_architecture_prompt(self, current_architecture: str, performance_data: dict, optimization_goal: str) -> str:
        """Generates a prompt for the LLM to suggest improvements to an existing architecture.

        Args:
            current_architecture: The current ML architecture being optimized.
            performance_data: A dictionary containing performance metrics (e.g., {'CTR': 0.05, 'Conversion': 0.01}).
            optimization_goal: A description of what to improve (e.g., "increase CTR and conversion rate").
        Returns:
            A string prompt.
        """
        formatted_arch = self._format_architecture_description(current_architecture)
        return f"""You are an expert ML architect tasked with optimizing recommender systems.
Analyze the current architecture and its performance data. Suggest specific modifications or an entirely new architecture to achieve the {optimization_goal}.
Explain your reasoning briefly.

Current Architecture:
{formatted_arch}

Performance Data: {performance_data}

Suggested Improvement / New Architecture:"""

    def genetic_operator_prompt(self, operation_type: str, architectures: list[str], performance_scores: list[float] = None) -> str:
        """Generates a prompt for the LLM to act as a genetic operator (mutation or crossover).

        Args:
            operation_type: "mutation" or "crossover".
            architectures: A list of parent architectures (1 for mutation, 2 for crossover).
            performance_scores: Optional performance scores for the architectures.
        Returns:
            A string prompt.
        """
        formatted_architectures = [self._format_architecture_description(arch) for arch in architectures]
        arch_info = "\n".join([f"Architecture {i+1}: {formatted_arch}" + (f" (Score: {performance_scores[i]:.4f})" if performance_scores else "") for i, formatted_arch in enumerate(formatted_architectures)])

        if operation_type == "mutation":
            return f"""You are an intelligent mutation operator for a genetic algorithm optimizing ML architectures.
Given the following parent architecture, introduce a small but impactful change to create a new offspring architecture.
Focus on architectural components, hyper-parameters, or feature engineering choices. The goal is to explore the search space intelligently.

Parent Architecture:
{arch_info}

Mutated Offspring Architecture:"""
        elif operation_type == "crossover":
            return f"""You are an intelligent crossover operator for a genetic algorithm optimizing ML architectures.
Given the following two parent architectures, combine their best features and ideas to create a superior offspring architecture.
Aim for a blend that leverages strengths from both parents.

Parent Architectures:
{arch_info}

Crossover Offspring Architecture:"""
        else:
            raise ValueError("operation_type must be 'mutation' or 'crossover'")

    def query_llm(self, prompt: str) -> str:
        """Simulates querying an LLM with the given prompt.
        In a real application, this would use an LLM API (e.g., OpenAI, Google Gemini).
        For now, it prints the prompt and returns a dummy response.

        Args:
            prompt: The prompt to send to the LLM.
        Returns:
            A simulated LLM response.
        """
        print(f"\n--- LLM Prompt ---\n{prompt}\n-------------------")
        # In a real scenario, integrate with an LLM API client here.
        # For demonstration, return a placeholder response based on the prompt type.
        if "Proposed Architecture:" in prompt:
            return "Simulated LLM Architecture: Embedding(32)->MultiHeadAttention(4)->FFN(128)->ReLU->Output(1)"
        elif "Suggested Improvement / New Architecture:" in prompt:
            return "Simulated LLM Improvement: Add a residual connection and increase embedding size to 64."
        elif "Mutated Offspring Architecture:" in prompt:
            return "Simulated LLM Mutation: Change activation from ReLU to GELU; add Dropout(0.2)."
        elif "Crossover Offspring Architecture:" in prompt:
            return "Simulated LLM Crossover: Blend of parent 1's embedding with parent 2's attention block."
        else:
            return "Simulated LLM Response: Default placeholder."
