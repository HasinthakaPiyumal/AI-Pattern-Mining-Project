import random
import numpy as np
from tqdm import tqdm

def generate_architecture_candidates(problem_description):
    architectures = [
        "InputLayer-Embedding(128)-Dense(64)-Output",
        "InputLayer-Embedding(256)-Dense(128)-Dropout(0.2)-Output",
        "InputLayer-Embedding(64)-Dense(32)-Dense(16)-Output",
        "InputLayer-Embedding(128)-GRU(64)-Dense(32)-Output",
        "InputLayer-Embedding(256)-LSTM(128)-Dense(64)-Dropout(0.3)-Output",
    ]
    return random.sample(architectures, k=3) 

def suggest_next_architecture(history):
    if not history:
        return "InputLayer-Embedding(128)-Dense(64)-Output"

    best_architecture_so_far = max(history, key=lambda x: x[1])[0]
    
    # Simulate LLM modifying the best architecture
    parts = best_architecture_so_far.split('-')
    if "Dropout" in best_architecture_so_far and random.random() < 0.5:
        parts = [p for p in parts if not p.startswith("Dropout")]
    elif random.random() < 0.3: 
        insert_idx = random.randint(1, len(parts) - 2) 
        parts.insert(insert_idx, f"Dense({random.choice([16, 32, 64])})")
    
    return "-".join(parts)

def llm_mutate(architecture_string):
    parts = architecture_string.split('-')
    if len(parts) < 3: 
        return architecture_string

    mutation_type = random.choice(["swap", "add_dropout", "change_units"])

    if mutation_type == "swap":
        idx1, idx2 = random.sample(range(1, len(parts) - 1), 2)
        parts[idx1], parts[idx2] = parts[idx2], parts[idx1]
    elif mutation_type == "add_dropout":
        if not any("Dropout" in p for p in parts):
            insert_idx = random.randint(1, len(parts) - 2)
            parts.insert(insert_idx, f"Dropout({round(random.uniform(0.1, 0.4), 1)})")
    elif mutation_type == "change_units":
        for i, part in enumerate(parts):
            if "Dense" in part or "Embedding" in part or "GRU" in part or "LSTM" in part:
                try:
                    current_units = int(part.split('(')[1].split(')')[0])
                    new_units = current_units + random.choice([-32, -16, 16, 32])
                    new_units = max(16, new_units)
                    parts[i] = part.replace(str(current_units), str(new_units))
                    break
                except ValueError:
                    continue
    return "-".join(parts)

def llm_crossover(parent1_architecture, parent2_architecture):
    p1_parts = parent1_architecture.split('-')
    p2_parts = parent2_architecture.split('-')

    crossover_point1 = random.randint(1, len(p1_parts) - 2)
    crossover_point2 = random.randint(1, len(p2_parts) - 2)

    child_parts = p1_parts[:crossover_point1] + p2_parts[crossover_point2:]
    
    unique_child_parts = []
    seen_layers = set()
    for part in child_parts:
        layer_type = part.split('(')[0]
        if layer_type not in seen_layers:
            unique_child_parts.append(part)
            seen_layers.add(layer_type)
        elif "Dropout" in part or "Dense" in part: 
            unique_child_parts.append(part)
    
    if not unique_child_parts or unique_child_parts[0] != "InputLayer" or unique_child_parts[-1] != "Output":
        return random.choice([parent1_architecture, parent2_architecture])

    return "-".join(unique_child_parts)

def train_and_evaluate(architecture_string):
    performance_score = np.random.uniform(0.5, 0.95)
    return performance_score

def auto_ml_platform(generations=10, population_size=5):
    print("Starting AutoML Platform for Recommender System Optimization...")

    problem_description = "Optimize recommender system for e-commerce with diverse product categories."
    current_architectures = generate_architecture_candidates(problem_description)

    history = []
    best_architecture = None
    best_performance = -1.0

    for gen in tqdm(range(generations), desc="Generations"):
        new_architectures = []
        
        # Evaluate current population
        for arch in current_architectures:
            score = train_and_evaluate(arch)
            history.append((arch, score))
            if score > best_performance:
                best_performance = score
                best_architecture = arch
            new_architectures.append((arch, score))
        
        current_architectures_sorted = sorted(new_architectures, key=lambda x: x[1], reverse=True)
        
        next_generation_candidates = []
        
        # Blackbox Optimization: Suggest a new architecture based on history
        suggested_arch = suggest_next_architecture(history)
        next_generation_candidates.append(suggested_arch)

        # Genetic Operations: Mutation and Crossover from top performers
        top_performers = [arch for arch, _ in current_architectures_sorted[:population_size // 2]]
        
        for arch in top_performers:
            mutated_arch = llm_mutate(arch)
            next_generation_candidates.append(mutated_arch)
        
        for _ in range(population_size - len(top_performers) - 1): 
            if len(top_performers) >= 2:
                parent1, parent2 = random.sample(top_performers, 2)
                crossed_arch = llm_crossover(parent1, parent2)
                next_generation_candidates.append(crossed_arch)
            else:
                next_generation_candidates.append(random.choice(top_performers))
        
        # Ensure unique architectures for next generation
        current_architectures = list(set(next_generation_candidates))
        current_architectures = current_architectures[:population_size]
        if not current_architectures: 
            current_architectures = generate_architecture_candidates(problem_description)

    print("\nAutoML Process Completed.")
    print(f"Best Architecture Found: {best_architecture}")
    print(f"Highest Simulated Performance: {best_performance:.4f}")
    return best_architecture, best_performance

if __name__ == "__main__":
    best_arch, best_perf = auto_ml_platform(generations=15, population_size=8)