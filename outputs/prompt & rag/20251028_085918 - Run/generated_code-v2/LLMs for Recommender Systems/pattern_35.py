import tensorflow as tf
import numpy as np
import pandas as pd
from collections import deque
import random


def mock_llm_generate_architecture(business_objectives: str) -> dict:
    if "maximize CTR" in business_objectives:
        return {
            "model_type": "DeepAndCrossNetwork",
            "embedding_dim": 64,
            "dense_layers": [128, 64],
            "activation": "relu",
            "output_activation": "sigmoid",
            "cross_layers": 3
        }
    return {
        "model_type": "DNN",
        "embedding_dim": 32,
        "dense_layers": [64, 32],
        "activation": "relu",
        "output_activation": "sigmoid"
    }


def build_recommender_model(architecture: dict, num_users: int, num_items: int) -> tf.keras.Model:
    user_input = tf.keras.layers.Input(shape=(1,), name="user_id")
    item_input = tf.keras.layers.Input(shape=(1,), name="item_id")

    user_embedding_dim = architecture.get("embedding_dim", 32)
    item_embedding_dim = architecture.get("embedding_dim", 32)
    activation = architecture.get("activation", "relu")
    output_activation = architecture.get("output_activation", "sigmoid")

    user_embedding = tf.keras.layers.Embedding(num_users + 1, user_embedding_dim, name="user_embedding")(user_input)
    user_vec = tf.keras.layers.Flatten()(user_embedding)

    item_embedding = tf.keras.layers.Embedding(num_items + 1, item_embedding_dim, name="item_embedding")(item_input)
    item_vec = tf.keras.layers.Flatten()(item_embedding)

    concat_vec = tf.keras.layers.concatenate([user_vec, item_vec])

    model_type = architecture.get("model_type", "DNN")

    if model_type == "DNN":
        x = concat_vec
        for units in architecture.get("dense_layers", [64, 32]):
            x = tf.keras.layers.Dense(units, activation=activation)(x)
        output = tf.keras.layers.Dense(1, activation=output_activation)(x)
        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)

    elif model_type == "DeepAndCrossNetwork":
        deep_input = concat_vec
        cross_input = concat_vec

        # Deep part
        for units in architecture.get("dense_layers", [128, 64]):
            deep_input = tf.keras.layers.Dense(units, activation=activation)(deep_input)

        # Cross part
        cross_layers = architecture.get("cross_layers", 3)
        x_0 = cross_input
        x_l = x_0
        for _ in range(cross_layers):
            x_l = x_0 * tf.keras.layers.Dense(cross_input.shape[-1], activation=None)(x_l) + x_l

        combined_output = tf.keras.layers.concatenate([deep_input, x_l])
        output = tf.keras.layers.Dense(1, activation=output_activation)(combined_output)
        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
    return model


def train_and_evaluate_model(model: tf.keras.Model, train_data, val_data) -> dict:
    history = model.fit(
        train_data["input"],
        train_data["target"],
        epochs=1,
        batch_size=32,
        validation_data=(val_data["input"], val_data["target"]),
        verbose=0
    )
    val_loss, val_auc = model.evaluate(val_data["input"], val_data["target"], verbose=0)
    return {"loss": val_loss, "auc": val_auc}


def mock_llm_optimize_architecture(trial_history: list[tuple[dict, dict]], current_best_arch: dict) -> dict:
    if not trial_history:
        return current_best_arch

    best_auc = -1
    best_arch = current_best_arch

    for arch, metrics in trial_history:
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_arch = arch

    if best_auc > 0.75:
        new_arch = best_arch.copy()
        new_arch["embedding_dim"] += 8
        print(f"LLM suggests increasing embedding dim to {new_arch['embedding_dim']}")
        return new_arch
    elif best_auc > 0.6:
        new_arch = best_arch.copy()
        if "dense_layers" in new_arch and new_arch["dense_layers"]:
            new_arch["dense_layers"].append(new_arch["dense_layers"][-1] // 2)
        print(f"LLM suggests adding a dense layer: {new_arch['dense_layers']}")
        return new_arch
    else:
        print("LLM suggests a more radical change or trying a different model type.")
        return {
            "model_type": "DeepAndCrossNetwork",
            "embedding_dim": 48,
            "dense_layers": [96, 48],
            "activation": "relu",
            "output_activation": "sigmoid",
            "cross_layers": 2
        }


def mock_llm_mutation(architecture: dict) -> dict:
    mutated_arch = architecture.copy()
    mutation_type = random.choice(["embedding_dim", "dense_layers_units", "add_layer", "activation"])

    if mutation_type == "embedding_dim":
        mutated_arch["embedding_dim"] = max(16, mutated_arch["embedding_dim"] + random.choice([-8, 8]))
    elif mutation_type == "dense_layers_units" and "dense_layers" in mutated_arch and mutated_arch["dense_layers"]:
        idx = random.randint(0, len(mutated_arch["dense_layers"]) - 1)
        mutated_arch["dense_layers"][idx] = max(16, mutated_arch["dense_layers"][idx] + random.choice([-16, 16]))
    elif mutation_type == "add_layer" and "dense_layers" in mutated_arch:
        if len(mutated_arch["dense_layers"]) < 5:
            mutated_arch["dense_layers"].append(mutated_arch["dense_layers"][-1] // 2 or 16)
    elif mutation_type == "activation":
        mutated_arch["activation"] = random.choice(["relu", "tanh", "sigmoid"])

    return mutated_arch


def mock_llm_crossover(arch1: dict, arch2: dict) -> dict:
    child_arch = {}
    for key in arch1.keys():
        child_arch[key] = random.choice([arch1[key], arch2[key]])
    return child_arch


class RecommenderOptimizer:
    def __init__(self, num_users: int, num_items: int, max_history_size: int = 10):
        self.num_users = num_users
        self.num_items = num_items
        self.trial_history = deque(maxlen=max_history_size)
        self.best_architecture = None
        self.best_performance = {"auc": -1}

    def run_optimization_cycle(self, business_objective: str, train_data, val_data):
        print("\n--- Starting Optimization Cycle ---")
        # 1. Architecture Generation
        if not self.best_architecture:
            print("Generating initial architecture...")
            current_architecture = mock_llm_generate_architecture(business_objective)
        else:
            # Use LLM Blackbox Optimizer to suggest next architecture
            print("Optimizing architecture with LLM blackbox agent...")
            current_architecture = mock_llm_optimize_architecture(list(self.trial_history), self.best_architecture)

        print(f"Proposed Architecture: {current_architecture}")

        # 2. Recommender Model Builder
        model = build_recommender_model(current_architecture, self.num_users, self.num_items)
        print("Model built and compiled.")

        # 3. Model Trainer and Evaluator
        print("Training and evaluating model...")
        performance_metrics = train_and_evaluate_model(model, train_data, val_data)
        print(f"Performance: {performance_metrics}")

        # Update history and best architecture
        self.trial_history.append((current_architecture, performance_metrics))

        if performance_metrics["auc"] > self.best_performance["auc"]:
            self.best_performance = performance_metrics
            self.best_architecture = current_architecture
            print("New best architecture found!")

        print(f"Current Best AUC: {self.best_performance['auc']:.4f}")
        return self.best_architecture, self.best_performance

    def run_genetic_optimization_cycle(self, population_size: int, train_data, val_data):
        print("\n--- Starting Genetic Optimization Cycle ---")

        if not self.trial_history:
            population = [mock_llm_generate_architecture("maximize CTR") for _ in range(population_size)]
        else:
            # Select top performers from history for initial population
            sorted_history = sorted(list(self.trial_history), key=lambda x: x[1]["auc"], reverse=True)
            population = [arch for arch, _ in sorted_history[:population_size]]
            while len(population) < population_size:
                 population.append(mock_llm_generate_architecture("general"))

        evaluated_population = []
        for arch in population:
            model = build_recommender_model(arch, self.num_users, self.num_items)
            metrics = train_and_evaluate_model(model, train_data, val_data)
            evaluated_population.append((arch, metrics))
            self.trial_history.append((arch, metrics))

        evaluated_population.sort(key=lambda x: x[1]["auc"], reverse=True)

        # Select parents (e.g., top 50%)
        parents = [arch for arch, _ in evaluated_population[:population_size // 2]]
        next_generation = []

        # Crossover and Mutation using LLM operators
        while len(next_generation) < population_size:
            if len(parents) >= 2:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = mock_llm_crossover(parent1, parent2)
            else:
                child = random.choice(parents).copy()

            if random.random() < 0.3:  # Mutation rate
                child = mock_llm_mutation(child)
            next_generation.append(child)

        # Update best architecture from this generation
        if evaluated_population[0][1]["auc"] > self.best_performance["auc"]:
            self.best_performance = evaluated_population[0][1]
            self.best_architecture = evaluated_population[0][0]
            print("New best architecture found via genetic algorithm!")

        print(f"Genetic Cycle Best AUC: {evaluated_population[0][1]['auc']:.4f}")
        print(f"Overall Best AUC: {self.best_performance['auc']:.4f}")
        return self.best_architecture, self.best_performance


def generate_dummy_data(num_samples=1000, num_users=100, num_items=500):
    user_ids = np.random.randint(1, num_users + 1, num_samples)
    item_ids = np.random.randint(1, num_items + 1, num_samples)
    # Simulate some interaction where higher user_id * item_id has higher chance of 1
    targets = ((user_ids * item_ids) % 100 > 50).astype(int) 
    
    train_samples = int(num_samples * 0.8)
    
    train_data = {
        "input": {"user_id": user_ids[:train_samples], "item_id": item_ids[:train_samples]},
        "target": targets[:train_samples]
    }
    val_data = {
        "input": {"user_id": user_ids[train_samples:], "item_id": item_ids[train_samples:]},
        "target": targets[train_samples:]
    }
    return train_data, val_data, num_users, num_items


if __name__ == "__main__":
    train_data, val_data, num_users, num_items = generate_dummy_data()

    optimizer = RecommenderOptimizer(num_users=num_users, num_items=num_items)

    print("\n=== Running LLM Blackbox Optimization Rounds ===")
    for i in range(3):
        best_arch, best_perf = optimizer.run_optimization_cycle("maximize CTR", train_data, val_data)

    print("\n=== Running LLM Genetic Algorithm Optimization Rounds ===")
    for i in range(2):
        best_arch, best_perf = optimizer.run_genetic_optimization_cycle(population_size=4, train_data=train_data, val_data=val_data)

    print("\n--- Final Optimized Architecture and Performance ---")
    print(f"Architecture: {optimizer.best_architecture}")
    print(f"Performance (AUC): {optimizer.best_performance['auc']:.4f}")
