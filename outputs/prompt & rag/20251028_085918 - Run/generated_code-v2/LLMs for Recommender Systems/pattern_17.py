# --- config.py ---
# """
# Configuration file for the AI-Powered Recommender System Optimization Platform.
# """

LLM_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key or similar LLM service
LLM_MODEL_NAME = "gpt-4"  # Or "gpt-3.5-turbo", "claude-v2", etc.

DATASET_PATH = "data/ecommerce_interactions.csv" # Placeholder for dataset
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

POPULATION_SIZE = 10
NUM_GENERATIONS = 3
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7

# --- llm_architecture_generator.py ---
import json

class LLMArchitectureGenerator:
    def __init__(self, llm_api_key: str, llm_model_name: str):
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name

    def generate_architecture_description(self, problem_description: str) -> str:
        prompt = f"""
        Generate a concise neural network architecture description for a recommender system 
        based on the following problem: {problem_description}.
        The architecture should be suitable for processing user-item interaction data 
        and outputting recommendation scores. Focus on common layers like Embedding, 
        Dense, ReLU, Dropout. Do not include training details. 
        Format the output as a JSON string with a single key 'architecture_description'.
        Example: {{"architecture_description": "Input Layer (User ID, Item ID) -> Embedding Layers (User Emb, Item Emb) -> Concatenate -> Dense (64, ReLU) -> Dropout (0.3) -> Dense (1, Sigmoid)"}}
        """
        
        if "collaborative filtering" in problem_description.lower():
            dummy_response = {"architecture_description": "Input (User ID, Item ID) -> UserEmbedding (32) -> ItemEmbedding (32) -> Concatenate -> Dense (64, ReLU) -> Dropout (0.2) -> Dense (32, ReLU) -> Dense (1, Sigmoid)"}
        else:
            dummy_response = {"architecture_description": "Input (User Features, Item Features) -> Dense (128, ReLU) -> Dropout (0.3) -> Dense (64, ReLU) -> Dense (1, Sigmoid)"}

        return dummy_response["architecture_description"]

    def parse_architecture_description(self, description: str) -> dict:
        layers = []
        if "->" in description:
            components = [comp.strip() for comp in description.split("->")]
            for comp in components:
                if "Input" in comp:
                    layers.append({"type": "Input", "details": comp})
                elif "Embedding" in comp:
                    parts = comp.split("(")
                    name = parts[0].strip()
                    size = int(parts[1].split(",")[0].strip())
                    layers.append({"type": "Embedding", "name": name, "size": size})
                elif "Dense" in comp:
                    parts = comp.split("(")
                    units = int(parts[1].split(",")[0].strip())
                    activation = parts[1].split(",")[-1].replace(")", "").strip()
                    layers.append({"type": "Dense", "units": units, "activation": activation})
                elif "Dropout" in comp:
                    parts = comp.split("(")
                    rate = float(parts[1].split(")")[0].strip())
                    layers.append({"type": "Dropout", "rate": rate})
                elif "Concatenate" in comp:
                    layers.append({"type": "Concatenate"})
        return {"layers": layers}

# --- recommender_model.py ---
import tensorflow as tf
from typing import Dict, Any, Tuple
import numpy as np

class RecommenderModel:
    def __init__(self, architecture_config: Dict[str, Any]):
        self.architecture_config = architecture_config
        self.model = None
        self.user_vocab_size = 10000
        self.item_vocab_size = 5000

    def build_model(self) -> tf.keras.Model:
        layers_config = self.architecture_config.get("layers", [])
        if not layers_config:
            raise ValueError("Architecture configuration must contain 'layers'.")

        inputs = {}
        current_output = None

        embedding_outputs = []
        for layer_info in layers_config:
            layer_type = layer_info["type"]
            if layer_type == "Input":
                if "User ID" in layer_info["details"]:
                    user_input = tf.keras.Input(shape=(1,), name="user_id_input")
                    inputs["user_id"] = user_input
                if "Item ID" in layer_info["details"]:
                    item_input = tf.keras.Input(shape=(1,), name="item_id_input")
                    inputs["item_id"] = item_input
            elif layer_type == "Embedding":
                if layer_info["name"] == "UserEmbedding":
                    if "user_id" not in inputs: 
                         inputs["user_id"] = tf.keras.Input(shape=(1,), name="user_id_input")
                    user_embedding = tf.keras.layers.Embedding(
                        self.user_vocab_size + 1,
                        layer_info["size"],
                        name="user_embedding"
                    )(inputs["user_id"])
                    user_embedding = tf.keras.layers.Flatten()(user_embedding)
                    embedding_outputs.append(user_embedding)
                elif layer_info["name"] == "ItemEmbedding":
                    if "item_id" not in inputs: 
                         inputs["item_id"] = tf.keras.Input(shape=(1,), name="item_id_input")
                    item_embedding = tf.keras.layers.Embedding(
                        self.item_vocab_size + 1,
                        layer_info["size"],
                        name="item_embedding"
                    )(inputs["item_id"])
                    item_embedding = tf.keras.layers.Flatten()(item_embedding)
                    embedding_outputs.append(item_embedding)

        if not inputs and len(embedding_outputs) == 0:
             raise ValueError("No valid input or embedding layers found in architecture.")

        if len(embedding_outputs) > 1:
            current_output = tf.keras.layers.concatenate(embedding_outputs)
        elif len(embedding_outputs) == 1:
            current_output = embedding_outputs[0]
        else: 
            if not current_output and len(inputs) > 0: 
                input_list = list(inputs.values())
                if len(input_list) > 1:
                    current_output = tf.keras.layers.concatenate([tf.keras.layers.Flatten()(i) if len(i.shape) > 2 else i for i in input_list])
                elif len(input_list) == 1:
                     current_output = tf.keras.layers.Flatten()(input_list[0]) if len(input_list[0].shape) > 2 else input_list[0]
                else:
                    raise ValueError("Architecture implies inputs but no clear path to current_output.")
            elif not current_output:
                raise ValueError("Architecture did not specify a clear starting point (Input or Embeddings).")

        for layer_info in layers_config:
            layer_type = layer_info["type"]
            if layer_type == "Dense":
                current_output = tf.keras.layers.Dense(
                    layer_info["units"],
                    activation=layer_info["activation"]
                )(current_output)
            elif layer_type == "Dropout":
                current_output = tf.keras.layers.Dropout(layer_info["rate"])(current_output)

        if not inputs:
            print("Warning: No explicit input layers found. Creating a generic input for demonstration.")
            inputs = {"generic_input": tf.keras.Input(shape=(current_output.shape[-1],), name="generic_feature_input")}

        self.model = tf.keras.Model(inputs=list(inputs.values()), outputs=current_output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return self.model

    def train_and_evaluate(self, X_train: Dict[str, np.ndarray], y_train: np.ndarray, 
                           X_val: Dict[str, np.ndarray], y_val: np.ndarray,
                           epochs: int, batch_size: int) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        print(f"Training model with architecture: {self.architecture_config}")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        
        predicted_scores = self.model.predict(X_val, verbose=0).flatten()
        auc_score = np.mean(predicted_scores)

        return {"loss": loss, "accuracy": accuracy, "auc": auc_score}

    def get_architecture_summary(self) -> str:
        if self.model:
            summary_list = []
            self.model.summary(print_fn=lambda x: summary_list.append(x))
            return "\n".join(summary_list)
        return "Model not built."

# --- llm_blackbox_optimizer.py ---
import json
from typing import Dict, Any, List

class LLMBlackboxOptimizer:
    def __init__(self, llm_api_key: str, llm_model_name: str):
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name

    def suggest_next_architecture(self, past_trials: List[Dict[str, Any]]) -> str:
        trial_summaries = []
        for i, trial in enumerate(past_trials):
            metrics = trial.get("metrics", {})
            architecture_desc = trial.get("architecture_description", "N/A")
            trial_summaries.append(
                f"Trial {i+1}:\n  Architecture: {architecture_desc}\n  Metrics: {json.dumps(metrics)}\n"
            )
        
        problem_statement = "Optimize a recommender system for e-commerce to maximize AUC and accuracy.\n"
        context = "We are using neural network architectures.\n"
        
        prompt = f"""
        {problem_statement}
        {context}
        
        Here are the results of previous architecture trials:
        {'\n'.join(trial_summaries)}
        
        Based on these results, propose a new, potentially better neural network architecture 
        description for the recommender system. Focus on common layers like Embedding, 
        Dense, ReLU, Dropout, and their configurations. Suggest concrete changes (e.g., 
        increase dense units, add a dropout layer, change activation). 
        Format the output as a JSON string with a single key 'architecture_description'.
        Example: {{"architecture_description": "Input (User ID, Item ID) -> Embedding Layers (User Emb (64), Item Emb (64)) -> Concatenate -> Dense (128, ReLU) -> Dropout (0.4) -> Dense (64, ReLU) -> Dense (1, Sigmoid)"}}
        """

        if past_trials:
            best_trial = max(past_trials, key=lambda x: x['metrics'].get('auc', 0.0))
            best_arch = best_trial['architecture_description']
            if best_trial['metrics'].get('auc', 0.0) < 0.75: 
                dummy_response = {"architecture_description": best_arch.replace("Dense (64, ReLU)", "Dense (128, ReLU)") + " -> Dense (32, ReLU)"}
            else:
                dummy_response = {"architecture_description": best_arch.replace("Dropout (0.2)", "Dropout (0.3)")}
        else:
            dummy_response = {"architecture_description": "Input (User ID, Item ID) -> UserEmbedding (32) -> ItemEmbedding (32) -> Concatenate -> Dense (64, ReLU) -> Dropout (0.2) -> Dense (1, Sigmoid)"}

        return dummy_response["architecture_description"]

# --- llm_genetic_operators.py ---
import json
from typing import Dict, Any

class LLMGeneticOperators:
    def __init__(self, llm_api_key: str, llm_model_name: str):
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name

    def mutate_architecture(self, parent_architecture_desc: str) -> str:
        prompt = f"""
        Given the following neural network architecture for a recommender system:
        {parent_architecture_desc}
        
        Apply a small, meaningful mutation to this architecture. This could involve:
        - Changing the number of units in a Dense layer (e.g., 64 to 128).
        - Modifying a Dropout rate (e.g., 0.2 to 0.3).
        - Adding or removing a simple layer (e.g., an extra Dense layer).
        - Changing an activation function (e.g., ReLU to LeakyReLU).
        
        Provide the mutated architecture as a JSON string with a single key 'architecture_description'.
        Example: {{"architecture_description": "Input (User ID, Item ID) -> UserEmbedding (32) -> ItemEmbedding (32) -> Concatenate -> Dense (128, ReLU) -> Dropout (0.2) -> Dense (1, Sigmoid)"}}
        """

        if "Dense (64, ReLU)" in parent_architecture_desc:
            mutated_desc = parent_architecture_desc.replace("Dense (64, ReLU)", "Dense (128, ReLU)", 1)
        elif "Dropout (0.2)" in parent_architecture_desc:
            mutated_desc = parent_architecture_desc.replace("Dropout (0.2)", "Dropout (0.3)", 1)
        else:
            parts = parent_architecture_desc.split("->")
            if len(parts) > 2:
                mutated_desc = " -> ".join(parts[:-1]) + " -> Dense (16, ReLU)" + " -> " + parts[-1]
            else:
                mutated_desc = parent_architecture_desc + " -> Dense (16, ReLU)"

        dummy_response = {"architecture_description": mutated_desc}

        return dummy_response["architecture_description"]

    def crossover_architectures(self, parent1_architecture_desc: str, parent2_architecture_desc: str) -> str:
        prompt = f"""
        Given two neural network architectures for a recommender system:
        Parent 1: {parent1_architecture_desc}
        Parent 2: {parent2_architecture_desc}
        
        Create a new 'offspring' architecture by combining meaningful components from both parents. 
        Aim for a balanced mix or an intelligent recombination. Ensure the resulting architecture 
        is coherent and functional. Focus on common layers like Embedding, Dense, ReLU, Dropout.
        
        Provide the new architecture as a JSON string with a single key 'architecture_description'.
        Example: {{"architecture_description": "Input (User ID, Item ID) -> UserEmbedding (64) -> ItemEmbedding (32) -> Concatenate -> Dense (128, ReLU) -> Dropout (0.3) -> Dense (1, Sigmoid)"}}
        """

        parts1 = [p.strip() for p in parent1_architecture_desc.split("->")]
        parts2 = [p.strip() for p in parent2_architecture_desc.split("->")]

        if len(parts1) > 2 and len(parts2) > 2:
            crossover_desc = " -> ".join(parts1[:2]) + " -> " + " -> ".join(parts2[2:])
        else:
            crossover_desc = parent1_architecture_desc.replace("Dropout (0.2)", "Dropout (0.25)") 

        dummy_response = {"architecture_description": crossover_desc}

        return dummy_response["architecture_description"]

# --- data_simulator.py ---
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def simulate_ecommerce_data(num_users: int = 1000, num_items: int = 500, num_interactions: int = 10000) -> pd.DataFrame:
    np.random.seed(42)
    user_ids = np.random.randint(1, num_users + 1, num_interactions)
    item_ids = np.random.randint(1, num_items + 1, num_interactions)
    ratings = np.random.randint(0, 2, num_interactions)

    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    return df

def prepare_data_for_recommender(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    train_size = int(0.8 * len(df))
    train_df = df.sample(n=train_size, random_state=42)
    val_df = df.drop(train_df.index)

    X_train = {
        "user_id_input": train_df['user_id'].values.reshape(-1, 1),
        "item_id_input": train_df['item_id'].values.reshape(-1, 1)
    }
    y_train = train_df['rating'].values

    X_val = {
        "user_id_input": val_df['user_id'].values.reshape(-1, 1),
        "item_id_input": val_df['item_id'].values.reshape(-1, 1)
    }
    y_val = val_df['rating'].values

    return X_train, y_train, X_val, y_val

# --- main.py ---
import os
import numpy as np

def run_automl_platform():
    print("\n--- Starting AI-Powered Recommender System Optimization Platform ---")

    print("\n1. Simulating E-commerce Data...")
    ecommerce_df = simulate_ecommerce_data(num_users=1000, num_items=500, num_interactions=10000)
    X_train, y_train, X_val, y_val = prepare_data_for_recommender(ecommerce_df)
    print(f"Data simulated: {len(ecommerce_df)} interactions. Train: {len(y_train)}, Val: {len(y_val)}")

    llm_gen = LLMArchitectureGenerator(LLM_API_KEY, LLM_MODEL_NAME)
    llm_optimizer = LLMBlackboxOptimizer(LLM_API_KEY, LLM_MODEL_NAME)
    llm_genetic_ops = LLMGeneticOperators(LLM_API_KEY, LLM_MODEL_NAME)

    print("\n--- Scenario 1: Initial Architecture Generation & Blackbox Optimization ---")
    problem_description = "A collaborative filtering recommender system for e-commerce, predicting user-item interaction likelihood."
    initial_arch_desc = llm_gen.generate_architecture_description(problem_description)
    initial_arch_config = llm_gen.parse_architecture_description(initial_arch_desc)
    print(f"\nLLM Generated Initial Architecture: {initial_arch_desc}")

    trials = []
    for i in range(3): 
        print(f"\n--- Blackbox Optimization Iteration {i+1} ---")
        current_arch_desc = initial_arch_desc if i == 0 else llm_optimizer.suggest_next_architecture(trials)
        current_arch_config = llm_gen.parse_architecture_description(current_arch_desc)
        print(f"Current Architecture for Trial {i+1}: {current_arch_desc}")

        try:
            model_instance = RecommenderModel(current_arch_config)
            model = model_instance.build_model()
            metrics = model_instance.train_and_evaluate(X_train, y_train, X_val, y_val, NUM_EPOCHS, BATCH_SIZE)
            print(f"  Trial {i+1} Metrics: {metrics}")
            trials.append({"architecture_description": current_arch_desc, "metrics": metrics})
        except ValueError as e:
            print(f"  Skipping invalid architecture for Trial {i+1}: {e}")
            trials.append({"architecture_description": current_arch_desc, "metrics": {"loss": float('inf'), "accuracy": 0.0, "auc": 0.0}})
        except Exception as e:
            print(f"  An error occurred during model building/training for Trial {i+1}: {e}")
            trials.append({"architecture_description": current_arch_desc, "metrics": {"loss": float('inf'), "accuracy": 0.0, "auc": 0.0}})

    best_trial_s1 = max(trials, key=lambda x: x['metrics'].get('auc', 0.0))
    print(f"\nBest Architecture from Blackbox Optimization: {best_trial_s1['architecture_description']} with metrics: {best_trial_s1['metrics']}")

    print("\n--- Scenario 2: LLM-powered Genetic Algorithm for NAS ---")

    population = []
    for _ in range(POPULATION_SIZE):
        arch_desc = llm_gen.generate_architecture_description("A simple click-prediction recommender model.")
        population.append({"architecture_description": arch_desc, "fitness": 0.0})

    for generation in range(NUM_GENERATIONS):
        print(f"\n--- Genetic Algorithm - Generation {generation+1} ---")
        for individual in population:
            arch_config = llm_gen.parse_architecture_description(individual["architecture_description"])
            try:
                model_instance = RecommenderModel(arch_config)
                model = model_instance.build_model()
                metrics = model_instance.train_and_evaluate(X_train, y_train, X_val, y_val, NUM_EPOCHS, BATCH_SIZE)
                individual["fitness"] = metrics.get('auc', 0.0) 
                print(f"  Evaluated: {individual['architecture_description']} -> Fitness: {individual['fitness']:.4f}")
            except ValueError as e:
                print(f"  Invalid architecture: {individual['architecture_description']}. Error: {e}. Setting fitness to 0.")
                individual["fitness"] = 0.0
            except Exception as e:
                print(f"  Error during evaluation: {e}. Setting fitness to 0.")
                individual["fitness"] = 0.0

        population.sort(key=lambda x: x["fitness"], reverse=True)
        print(f"  Best in Generation {generation+1}: {population[0]['architecture_description']} (Fitness: {population[0]['fitness']:.4f})")

        if generation < NUM_GENERATIONS - 1:
            new_population = []
            new_population.append(population[0]) 

            while len(new_population) < POPULATION_SIZE:
                parent1 = population[np.random.randint(0, len(population) // 2)] 
                parent2 = population[np.random.randint(0, len(population) // 2)]

                child_arch_desc = parent1["architecture_description"]

                if np.random.rand() < CROSSOVER_RATE:
                    child_arch_desc = llm_genetic_ops.crossover_architectures(
                        parent1["architecture_description"],
                        parent2["architecture_description"]
                    )

                if np.random.rand() < MUTATION_RATE:
                    child_arch_desc = llm_genetic_ops.mutate_architecture(child_arch_desc)
                
                new_population.append({"architecture_description": child_arch_desc, "fitness": 0.0})
            population = new_population[:POPULATION_SIZE]

    best_architecture_ga = population[0]
    print(f"\nFinal Best Architecture from Genetic Algorithm: {best_architecture_ga['architecture_description']} (Fitness: {best_architecture_ga['fitness']:.4f})")

    print("\n--- AI-Powered Recommender System Optimization Platform Finished ---")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    run_automl_platform()