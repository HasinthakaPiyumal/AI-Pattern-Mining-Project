import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Model, optimizers, losses, metrics
import random

class DataSimulator:
    def __init__(self, num_users=1000, num_items=500, num_interactions=50000):
        self.num_users = num_users
        self.num_items = num_items
        self.num_interactions = num_interactions
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def generate_synthetic_data(self):
        user_ids = np.random.randint(0, self.num_users, self.num_interactions)
        item_ids = np.random.randint(0, self.num_items, self.num_interactions)
        ratings = np.random.randint(1, 6, self.num_interactions) # Example: 1-5 star ratings

        df_interactions = pd.DataFrame({
            "user_id": user_ids,
            "item_id": item_ids,
            "rating": ratings
        })

        item_features = pd.DataFrame({
            "item_id": np.arange(self.num_items),
            "category": np.random.choice(["electronics", "books", "clothing", "home", "sports"], self.num_items),
            "price": np.random.rand(self.num_items) * 100 + 10 
        })

        return df_interactions, item_features

    def preprocess_data(self, df_interactions, item_features):
        df_interactions["encoded_user_id"] = self.user_encoder.fit_transform(df_interactions["user_id"])
        df_interactions["encoded_item_id"] = self.item_encoder.fit_transform(df_interactions["item_id"])

        num_unique_users = len(self.user_encoder.classes_)
        num_unique_items = len(self.item_encoder.classes_)

        item_features_encoded = item_features.copy()
        item_features_encoded = item_features_encoded[item_features_encoded["item_id"].isin(self.item_encoder.classes_)]
        item_features_encoded["encoded_item_id"] = self.item_encoder.transform(item_features_encoded["item_id"])
        item_features_encoded = item_features_encoded.set_index("encoded_item_id").sort_index()

        item_features_encoded = pd.get_dummies(item_features_encoded, columns=["category"], prefix="cat")
        item_feature_cols = [col for col in item_features_encoded.columns if col not in ["item_id"]]
        item_feature_matrix = item_features_encoded[item_feature_cols].values
        item_feature_dim = item_feature_matrix.shape[1]

        train_df, val_df = train_test_split(df_interactions, test_size=0.2, random_state=42)

        return (train_df, val_df, num_unique_users, num_unique_items, item_feature_matrix, item_feature_dim)

class LLMAutoMLAgent:
    def __init__(self, num_users, num_items, item_feature_dim):
        self.num_users = num_users
        self.num_items = num_items
        self.item_feature_dim = item_feature_dim

    def _generate_random_architecture_params(self):
        emb_size = random.choice([32, 64, 128])
        num_dense_layers = random.choice([1, 2, 3])
        dense_layer_sizes = [random.choice([64, 128, 256]) for _ in range(num_dense_layers)]
        activation = random.choice(["relu", "tanh"])
        learning_rate = random.choice([0.001, 0.0005, 0.0001])
        return f"emb_size:{emb_size},dense_layers:{dense_layer_sizes},activation:{activation},lr:{learning_rate}"

    def generate_initial_architecture(self, problem_description):
        return self._generate_random_architecture_params()

    def suggest_new_architecture_blackbox(self, history_of_trials):
        if history_of_trials:
            best_trial = max(history_of_trials, key=lambda x: x[1])
            arch_str = best_trial[0]
            parts = arch_str.split(',')
            param_to_change = random.choice(parts)
            if "emb_size" in param_to_change:
                new_size = random.choice([32, 64, 128, 256])
                arch_str = arch_str.replace(param_to_change, f"emb_size:{new_size}")
            elif "dense_layers" in param_to_change:
                current_sizes_str = param_to_change.split(':')[1].strip('[]')
                current_sizes = [int(s) for s in current_sizes_str.split(',') if s]
                if current_sizes:
                    idx = random.randint(0, len(current_sizes) - 1)
                    current_sizes[idx] = random.choice([64, 128, 256, 512])
                    arch_str = arch_str.replace(param_to_change, f"dense_layers:{current_sizes}")
            elif "activation" in param_to_change:
                new_act = random.choice(["relu", "tanh", "sigmoid"])
                arch_str = arch_str.replace(param_to_change, f"activation:{new_act}")
            elif "lr" in param_to_change:
                new_lr = random.choice([0.001, 0.0005, 0.0002, 0.0001])
                arch_str = arch_str.replace(param_to_change, f"lr:{new_lr}")
            return arch_str
        return self._generate_random_architecture_params()

    def genetic_mutation(self, parent_architecture):
        return self.suggest_new_architecture_blackbox([(parent_architecture, 0)])

    def genetic_crossover(self, parent1_architecture, parent2_architecture):
        p1_parts = {p.split(':')[0]: p.split(':')[1] for p in parent1_architecture.split(',')}
        p2_parts = {p.split(':')[0]: p.split(':')[1] for p in parent2_architecture.split(',')}

        child_parts = {}
        for key in p1_parts:
            child_parts[key] = random.choice([p1_parts[key], p2_parts.get(key, p1_parts[key])])

        return ",".join([f"{k}:{v}" for k, v in child_parts.items()])

class RecommenderModel:
    def __init__(self, num_users, num_items, item_feature_matrix, item_feature_dim):
        self.num_users = num_users
        self.num_items = num_items
        self.item_feature_matrix = item_feature_matrix
        self.item_feature_dim = item_feature_dim
        self.model = None

    def _parse_architecture_string(self, arch_str):
        params = {}
        parts = arch_str.split(',')
        for part in parts:
            key, value = part.split(':', 1)
            if key == "emb_size":
                params[key] = int(value)
            elif key == "dense_layers":
                params[key] = eval(value) 
            elif key == "activation":
                params[key] = value
            elif key == "lr":
                params[key] = float(value)
        return params

    def build_model(self, architecture_string):
        params = self._parse_architecture_string(architecture_string)
        emb_size = params.get("emb_size", 64)
        dense_layers_sizes = params.get("dense_layers", [128, 64])
        activation = params.get("activation", "relu")

        user_input = layers.Input(shape=(1,), name="user_id")
        item_input = layers.Input(shape=(1,), name="item_id")

        user_embedding = layers.Embedding(self.num_users, emb_size, name="user_embedding")(user_input)
        user_vec = layers.Flatten(name="user_vec")(user_embedding)

        item_embedding = layers.Embedding(self.num_items, emb_size, name="item_embedding")(item_input)
        item_vec = layers.Flatten(name="item_vec")(item_embedding)

        item_features_input = layers.Input(shape=(self.item_feature_dim,), name="item_features")

        concat_vec = layers.concatenate([user_vec, item_vec, item_features_input])

        x = concat_vec
        for layer_size in dense_layers_sizes:
            x = layers.Dense(layer_size, activation=activation)(x)
            x = layers.Dropout(0.2)(x)

        output = layers.Dense(1)(x)

        self.model = Model(inputs=[user_input, item_input, item_features_input], outputs=output)

        learning_rate = params.get("lr", 0.001)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=losses.MeanSquaredError(), metrics=[metrics.MeanAbsoluteError()])
        return self.model

    def train_model(self, model, train_df, epochs=5, batch_size=32):
        if model is None:
            raise ValueError("Model is not built. Call build_model first.")

        train_item_features = self.item_feature_matrix[train_df["encoded_item_id"].values]

        history = model.fit(
            {"user_id": train_df["encoded_user_id"], "item_id": train_df["encoded_item_id"], "item_features": train_item_features},
            train_df["rating"],
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        return history

    def evaluate_model(self, model, val_df, k=10):
        if model is None:
            raise ValueError("Model is not built. Call build_model first.")

        val_user_ids = val_df["encoded_user_id"].unique()
        all_item_ids = np.arange(self.num_items)

        all_item_features = self.item_feature_matrix[all_item_ids]

        recall_scores = []
        ndcg_scores = []

        for user_id in val_user_ids:
            actual_items = val_df[val_df["encoded_user_id"] == user_id]["encoded_item_id"].values

            if len(actual_items) == 0:
                continue

            user_input_for_pred = np.full((self.num_items, 1), user_id)
            item_input_for_pred = all_item_ids.reshape(-1, 1)

            item_features_for_pred = all_item_features

            predictions = model.predict(
                {"user_id": user_input_for_pred, "item_id": item_input_for_pred, "item_features": item_features_for_pred},
                verbose=0
            ).flatten()

            ranked_item_indices = np.argsort(predictions)[::-1]
            top_k_recommended_items = all_item_ids[ranked_item_indices[:k]]

            hits = len(set(top_k_recommended_items) & set(actual_items))
            recall_at_k = hits / min(len(actual_items), k)

            dcg = 0.0
            idcg = 0.0
            for i, item in enumerate(top_k_recommended_items):
                if item in actual_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            for i in range(min(len(actual_items), k)):
                idcg += 1.0 / np.log2(i + 2)

            ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

            recall_scores.append(recall_at_k)
            ndcg_scores.append(ndcg_at_k)

        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        return {"Recall@K": avg_recall, "NDCG@K": avg_ndcg}

class AutoMLOrchestrator:
    def __init__(self, data_simulator, llm_agent, recommender_model_builder, generations=5, population_size=10):
        self.data_simulator = data_simulator
        self.llm_agent = llm_agent
        self.recommender_model_builder = recommender_model_builder
        self.generations = generations
        self.population_size = population_size
        self.best_architecture = None
        self.best_performance = -1

    def run_automl(self, train_df, val_df, num_users, num_items, item_feature_matrix, item_feature_dim):
        population = []
        history_of_trials = []

        print("Initializing population...")
        for _ in range(self.population_size):
            arch = self.llm_agent.generate_initial_architecture("E-commerce recommender system")
            population.append({"architecture": arch, "performance": -1, "model": None})

        for gen in range(self.generations):
            print(f"\n--- Generation {gen + 1}/{self.generations} ---")
            for i, individual in enumerate(population):
                print(f"  Evaluating individual {i + 1}/{self.population_size} with architecture: {individual['architecture']}")
                model = self.recommender_model_builder.build_model(individual["architecture"])
                self.recommender_model_builder.train_model(model, train_df)
                metrics = self.recommender_model_builder.evaluate_model(model, val_df)
                performance = metrics["Recall@K"]
                print(f"    Performance (Recall@K): {performance:.4f}")

                individual["performance"] = performance
                history_of_trials.append((individual["architecture"], performance))

                if performance > self.best_performance:
                    self.best_performance = performance
                    self.best_architecture = individual["architecture"]
                    print(f"    New best architecture found: {self.best_architecture} with Recall@K: {self.best_performance:.4f}")

            population.sort(key=lambda x: x["performance"], reverse=True)
            new_population = population[:self.population_size // 2]

            while len(new_population) < self.population_size:
                parent1 = random.choice(new_population)["architecture"]
                parent2 = random.choice(new_population)["architecture"]

                if random.random() < 0.7:
                    offspring_arch = self.llm_agent.genetic_crossover(parent1, parent2)
                else:
                    offspring_arch = self.llm_agent.genetic_mutation(parent1)

                new_population.append({"architecture": offspring_arch, "performance": -1, "model": None})
            population = new_population

        print("\n--- AutoML Process Finished ---")
        print(f"Best Architecture Found: {self.best_architecture}")
        print(f"Best Performance (Recall@K): {self.best_performance:.4f}")

        return self.best_architecture, self.best_performance

if __name__ == "__main__":
    data_simulator = DataSimulator(num_users=100, num_items=50, num_interactions=5000)
    df_interactions, item_features = data_simulator.generate_synthetic_data()
    train_df, val_df, num_users, num_items, item_feature_matrix, item_feature_dim = \
        data_simulator.preprocess_data(df_interactions, item_features)

    print(f"Number of unique users: {num_users}")
    print(f"Number of unique items: {num_items}")
    print(f"Item feature dimension: {item_feature_dim}")
    print(f"Train interactions: {len(train_df)}")
    print(f"Validation interactions: {len(val_df)}")

    llm_agent = LLMAutoMLAgent(num_users, num_items, item_feature_dim)
    recommender_model_builder = RecommenderModel(num_users, num_items, item_feature_matrix, item_feature_dim)

    orchestrator = AutoMLOrchestrator(
        data_simulator,
        llm_agent,
        recommender_model_builder,
        generations=3,
        population_size=5
    )

    best_arch, best_perf = orchestrator.run_automl(
        train_df, val_df, num_users, num_items, item_feature_matrix, item_feature_dim
    )

    print("\nFinal Best Architecture:")
    print(best_arch)
    print(f"Final Best Performance (Recall@K): {best_perf:.4f}")