import random
import json

class RecommenderArchitecture:
    """Represents a simplified recommender model architecture."""
    def __init__(self, embedding_dim: int, num_layers: int, activation: str, learning_rate: float, dropout_rate: float):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.id = hash(f"{embedding_dim}-{num_layers}-{activation}-{learning_rate}-{dropout_rate}")

    def to_dict(self):
        return {
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def from_dict(data: dict):
        return RecommenderArchitecture(
            data["embedding_dim"],
            data["num_layers"],
            data["activation"],
            data["learning_rate"],
            data["dropout_rate"],
        )

    def __repr__(self):
        return f"Arch(emb={self.embedding_dim}, layers={self.num_layers}, act={self.activation}, lr={self.learning_rate:.4f}, drop={self.dropout_rate:.2f})"

    def __eq__(self, other):
        if not isinstance(other, RecommenderArchitecture):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return self.id


class LLMClient:
    """Simulates interactions with a Large Language Model for architecture generation and optimization.
    In a real application, this would involve API calls to an actual LLM (e.g., GPT-4).
    """
    def __init__(self):
        self.available_activations = ["relu", "sigmoid", "tanh", "leaky_relu"]

    def _simulate_llm_response(self, prompt: str, content_type: str = "text"):
        """Helper to simulate LLM processing and response based on prompt context."""
        print(f"[LLM_Client] Simulating LLM response for prompt: {prompt[:100]}...")
        if "generate an initial candidate architecture" in prompt:
            # Simulate generating a reasonable initial architecture
            return json.dumps({
                "embedding_dim": random.choice([32, 64, 128]),
                "num_layers": random.choice([2, 3, 4]),
                "activation": random.choice(self.available_activations),
                "learning_rate": random.uniform(0.0001, 0.01),
                "dropout_rate": random.uniform(0.1, 0.5),
            })
        elif "analyze the following trial results and suggest a better architecture" in prompt:
            # Simulate analyzing past trials and suggesting an improvement
            # For simplicity, we'll make a small random tweak to a 'good' architecture if available
            # In a real scenario, LLM would parse trial data and reason.
            try:
                trials_str = prompt.split("Trial Results:\n")[-1]
                trials_data = json.loads(trials_str)
                best_trial = max(trials_data, key=lambda x: x['performance'])
                best_arch_dict = best_trial['architecture']

                # Make a slight modification based on the best performing one
                new_arch_dict = best_arch_dict.copy()
                change = random.choice(["embedding_dim", "num_layers", "learning_rate", "dropout_rate"])
                if change == "embedding_dim":
                    new_arch_dict["embedding_dim"] = random.choice([32, 64, 128, 256])
                elif change == "num_layers":
                    new_arch_dict["num_layers"] = random.choice([1, 2, 3, 4, 5])
                elif change == "learning_rate":
                    new_arch_dict["learning_rate"] = max(0.00005, min(0.05, new_arch_dict["learning_rate"] * random.uniform(0.8, 1.2)))
                elif change == "dropout_rate":
                    new_arch_dict["dropout_rate"] = max(0.0, min(0.6, new_arch_dict["dropout_rate"] + random.uniform(-0.1, 0.1)))
                new_arch_dict["activation"] = random.choice(self.available_activations) # Randomly change activation too
                return json.dumps(new_arch_dict)
            except Exception as e:
                print(f"[LLM_Client] Error parsing trials or generating improvement: {e}")
                # Fallback to random if parsing fails
                return self._simulate_llm_response("generate an initial candidate architecture")
        elif "generate a mutated architecture" in prompt or "generate a crossover architecture" in prompt:
            # Simulate genetic operators
            try:
                parents_info = json.loads(prompt.split("Parents: ")[-1].split("\nEvolutionary Context:")[0])
                parent1_dict = parents_info["parent1"]
                parent2_dict = parents_info.get("parent2") # Might be None for mutation

                if "mutated" in prompt:
                    # Simple mutation: randomly change one parameter of parent1
                    mutated_arch_dict = parent1_dict.copy()
                    change_param = random.choice(list(mutated_arch_dict.keys()))
                    if change_param == "embedding_dim":
                        mutated_arch_dict["embedding_dim"] = random.choice([32, 64, 128, 256])
                    elif change_param == "num_layers":
                        mutated_arch_dict["num_layers"] = random.choice([1, 2, 3, 4, 5])
                    elif change_param == "activation":
                        mutated_arch_dict["activation"] = random.choice(self.available_activations)
                    elif change_param == "learning_rate":
                        mutated_arch_dict["learning_rate"] = max(0.00005, min(0.05, mutated_arch_dict["learning_rate"] * random.uniform(0.5, 1.5)))
                    elif change_param == "dropout_rate":
                        mutated_arch_dict["dropout_rate"] = max(0.0, min(0.6, mutated_arch_dict["dropout_rate"] + random.uniform(-0.2, 0.2)))
                    return json.dumps(mutated_arch_dict)

                elif "crossover" in prompt and parent2_dict:
                    # Simple crossover: combine parameters from both parents
                    crossover_arch_dict = {}
                    for key in parent1_dict.keys():
                        crossover_arch_dict[key] = random.choice([parent1_dict[key], parent2_dict[key]])
                    
                    # Ensure learning rate and dropout are within reasonable bounds after crossover
                    crossover_arch_dict["learning_rate"] = max(0.00005, min(0.05, crossover_arch_dict["learning_rate"]))
                    crossover_arch_dict["dropout_rate"] = max(0.0, min(0.6, crossover_arch_dict["dropout_rate"]))

                    return json.dumps(crossover_arch_dict)
            except Exception as e:
                print(f"[LLM_Client] Error in genetic operator simulation: {e}")
                return self._simulate_llm_response("generate an initial candidate architecture") # Fallback
        
        return "{}" # Default empty response

    def generate_initial_architecture(self) -> RecommenderArchitecture:
        prompt = (
            "You are an expert in recommender system architecture design. "
            "Generate an initial candidate architecture for an e-commerce recommender model. "
            "Provide the architecture as a JSON object with keys: embedding_dim, num_layers, activation, learning_rate, dropout_rate. "
            "Consider common practices for collaborative filtering or content-based recommendation networks."
        )
        arch_json_str = self._simulate_llm_response(prompt)
        arch_dict = json.loads(arch_json_str)
        return RecommenderArchitecture.from_dict(arch_dict)

    def optimize_architecture_blackbox(self, past_trials: list[dict]) -> RecommenderArchitecture:
        """Uses LLM as a blackbox agent to analyze past trials and suggest an improved architecture."""
        trials_str = json.dumps(past_trials, indent=2)
        prompt = (
            "You are an AutoML optimization agent for recommender systems. "
            "Analyze the following trial results (architecture and performance) and suggest a better-performing architecture. "
            "Focus on improving the performance metric. Return the suggested architecture as a JSON object. "
            "Trial Results:\n" + trials_str
        )
        arch_json_str = self._simulate_llm_response(prompt)
        arch_dict = json.loads(arch_json_str)
        return RecommenderArchitecture.from_dict(arch_dict)

    def apply_genetic_operator(self, operator_type: str, parent1: RecommenderArchitecture, parent2: RecommenderArchitecture = None, current_population_info: list[dict] = None) -> RecommenderArchitecture:
        """Applies an LLM-driven genetic operator (mutation or crossover)."""
        parents_data = {"parent1": parent1.to_dict()}
        if parent2:
            parents_data["parent2"] = parent2.to_dict()
        
        population_context = json.dumps(current_population_info, indent=2) if current_population_info else "[]"

        if operator_type == "mutation":
            prompt = (
                "You are an LLM-driven mutation operator for a genetic algorithm in recommender system NAS. "
                f"Given the following parent architecture, generate a mutated architecture. "
                "Focus on making a small, intelligent change that could lead to performance improvement, considering the overall evolutionary context. "
                "Return the mutated architecture as a JSON object.\n" 
                f"Parents: {json.dumps(parents_data)}\n"
                f"Evolutionary Context: {population_context}"
            )
        elif operator_type == "crossover" and parent2:
            prompt = (
                "You are an LLM-driven crossover operator for a genetic algorithm in recommender system NAS. "
                f"Given the following two parent architectures, generate a child architecture by combining their features. "
                "Aim for a promising combination that leverages the strengths of both parents. "
                "Return the crossover architecture as a JSON object.\n" 
                f"Parents: {json.dumps(parents_data)}\n"
                f"Evolutionary Context: {population_context}"
            )
        else:
            raise ValueError("Invalid operator_type or missing parent2 for crossover.")

        arch_json_str = self._simulate_llm_response(prompt)
        arch_dict = json.loads(arch_json_str)
        return RecommenderArchitecture.from_dict(arch_dict)


def evaluate_architecture(architecture: RecommenderArchitecture) -> float:
    """Simulates the training and evaluation of a recommender model with the given architecture.
    In a real scenario, this would involve:
    1. Setting up a model with the defined architecture (e.g., using TensorFlow/PyTorch).
    2. Loading e-commerce dataset.
    3. Training the model.
    4. Evaluating performance (e.g., AUC, Recall@K, NDCG@K) on a validation set.
    Returns a dummy performance score for demonstration purposes.
    """
    print(f"  [Evaluation] Evaluating architecture: {architecture}...")
    # Simulate a noisy performance based on architecture parameters
    # Higher embedding_dim, more layers, lower learning_rate (within limits) tend to be better
    base_score = 0.5
    base_score += architecture.embedding_dim * 0.0005
    base_score += architecture.num_layers * 0.01
    base_score -= architecture.learning_rate * 10 # Penalize high LR
    base_score -= architecture.dropout_rate * 0.2 # Penalize high dropout slightly

    if architecture.activation == "relu":
        base_score += 0.03
    elif architecture.activation == "tanh":
        base_score += 0.02

    # Add some randomness to simulate real-world training variance
    performance = max(0.1, min(0.95, base_score + random.uniform(-0.05, 0.05)))
    print(f"  [Evaluation] Architecture {architecture.id} evaluated with performance: {performance:.4f}")
    return performance


class GeneticAlgorithmOptimizer:
    """Orchestrates the genetic algorithm for NAS, integrating LLM for operators."""
    def __init__(self, llm_client: LLMClient, population_size: int = 10, generations: int = 5, mutation_rate: float = 0.3):
        self.llm_client = llm_client
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population: list[tuple[RecommenderArchitecture, float]] = []
        self.history: list[dict] = []

    def initialize_population(self):
        print(f"[GA] Initializing population of {self.population_size} architectures...")
        initial_population = []
        for _ in range(self.population_size):
            arch = self.llm_client.generate_initial_architecture()
            performance = evaluate_architecture(arch)
            initial_population.append((arch, performance))
            self.history.append({"type": "initial", "architecture": arch.to_dict(), "performance": performance})
        self.population = sorted(initial_population, key=lambda x: x[1], reverse=True)
        print("[GA] Initial population generated and evaluated.")

    def select_parents(self) -> tuple[RecommenderArchitecture, RecommenderArchitecture]:
        """Tournament selection for parents."""
        # For simplicity, just pick top 2 for now
        return self.population[0][0], self.population[1][0]

    def run(self):
        self.initialize_population()

        for gen in range(self.generations):
            print(f"\n--- Generation {gen + 1}/{self.generations} ---")
            new_population = []

            # Elitism: Keep the best architecture(s) unchanged
            new_population.append(self.population[0]) 
            
            # Generate new offspring using LLM-driven genetic operators
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                
                current_pop_info = [{
                    "architecture": arch.to_dict(), 
                    "performance": perf
                } for arch, perf in self.population]

                if random.random() < self.mutation_rate:
                    # LLM-driven Mutation
                    offspring_arch = self.llm_client.apply_genetic_operator("mutation", parent1, current_population_info=current_pop_info)
                    print(f"[GA] LLM applied mutation to {parent1.id} -> {offspring_arch.id}")
                else:
                    # LLM-driven Crossover
                    offspring_arch = self.llm_client.apply_genetic_operator("crossover", parent1, parent2, current_population_info=current_pop_info)
                    print(f"[GA] LLM applied crossover to {parent1.id} and {parent2.id} -> {offspring_arch.id}")

                # Evaluate the new offspring
                offspring_performance = evaluate_architecture(offspring_arch)
                new_population.append((offspring_arch, offspring_performance))
                self.history.append({"type": "genetic", "generation": gen+1, "architecture": offspring_arch.to_dict(), "performance": offspring_performance})
            
            self.population = sorted(new_population, key=lambda x: x[1], reverse=True)[:self.population_size] # Keep top N
            print(f"[GA] Generation {gen + 1} best performance: {self.population[0][1]:.4f} (Arch: {self.population[0][0]})\n")

        print("\n--- Genetic Algorithm Finished ---")
        best_arch, best_perf = self.population[0]
        print(f"Overall best architecture found: {best_arch}")
        print(f"Overall best performance: {best_perf:.4f}")
        return best_arch, best_perf


def main():
    print("Starting LLM-driven AutoML for E-commerce Recommender System")
    llm_client = LLMClient()

    # Part 1: Architecture Generation (Initial Candidates)
    print("\n--- Phase 1: Initial LLM-driven Architecture Generation ---")
    initial_architectures = []
    for i in range(3):
        arch = llm_client.generate_initial_architecture()
        performance = evaluate_architecture(arch)
        initial_architectures.append({"architecture": arch.to_dict(), "performance": performance})
        print(f"Initial candidate {i+1}: {arch} with performance {performance:.4f}")

    # Part 2: Blackbox Agent for Optimization
    print("\n--- Phase 2: LLM as Blackbox Optimization Agent ---")
    print("Simulating iterative optimization over 3 steps.")
    current_best_arch_dict = initial_architectures[0]["architecture"]
    current_best_performance = initial_architectures[0]["performance"]
    all_trials = list(initial_architectures)

    for i in range(3):
        print(f"  Optimization Step {i+1}: Current best performance: {current_best_performance:.4f}")
        suggested_arch = llm_client.optimize_architecture_blackbox(all_trials)
        suggested_performance = evaluate_architecture(suggested_arch)
        all_trials.append({"architecture": suggested_arch.to_dict(), "performance": suggested_performance})

        if suggested_performance > current_best_performance:
            current_best_performance = suggested_performance
            current_best_arch_dict = suggested_arch.to_dict()
            print(f"  New best architecture found: {suggested_arch} with performance {suggested_performance:.4f}")
        else:
            print(f"  Suggested architecture ({suggested_arch}) did not improve. Performance: {suggested_performance:.4f}")

    print(f"\nBlackbox optimization phase finished. Best architecture found: {RecommenderArchitecture.from_dict(current_best_arch_dict)}")
    print(f"Best performance from blackbox optimization: {current_best_performance:.4f}")

    # Part 3: Genetic Operator Integration (LLM-driven GA)
    print("\n--- Phase 3: LLM-driven Genetic Algorithm for NAS ---")
    ga_optimizer = GeneticAlgorithmOptimizer(llm_client, population_size=5, generations=3, mutation_rate=0.4)
    best_ga_arch, best_ga_perf = ga_optimizer.run()

    print("\n--- Overall Summary ---")
    print(f"Initial LLM-generated architectures (top): {initial_architectures[0]["architecture"]} - {initial_architectures[0]["performance"]:.4f}")
    print(f"Best from LLM Blackbox Optimization: {RecommenderArchitecture.from_dict(current_best_arch_dict)} - {current_best_performance:.4f}")
    print(f"Best from LLM-driven Genetic Algorithm: {best_ga_arch} - {best_ga_perf:.4f}")

if __name__ == "__main__":
    main()
