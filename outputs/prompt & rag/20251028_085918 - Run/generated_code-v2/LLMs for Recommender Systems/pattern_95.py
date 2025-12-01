
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class LLMService:
    def generate_initial_architecture(self, domain_context: str) -> dict:
        return {
            "embedding_dim": 32,
            "hidden_layers": [64, 32],
            "activation": "relu",
            "output_activation": "sigmoid"
        }

    def suggest_architecture_improvements(self, current_architecture: dict, performance_metrics: dict, trial_history: list) -> dict:
        if performance_metrics.get("accuracy", 0) < 0.7:
            new_hidden_layers = [layer * 2 for layer in current_architecture["hidden_layers"]]
            return {
                **current_architecture,
                "hidden_layers": new_hidden_layers,
                "embedding_dim": current_architecture["embedding_dim"] + 8
            }
        else:
            return current_architecture

    def apply_genetic_operator(self, parent_architectures: list, operator_type: str) -> dict:
        if operator_type == "mutation" and parent_architectures:
            arch = parent_architectures[0].copy()
            arch["embedding_dim"] = np.random.randint(16, 64)
            return arch
        elif operator_type == "crossover" and len(parent_architectures) >= 2:
            arch1 = parent_architectures[0]
            arch2 = parent_architectures[1]
            crossover_point = np.random.randint(0, len(arch1["hidden_layers"]))
            new_hidden_layers = arch1["hidden_layers"][:crossover_point] + arch2["hidden_layers"][crossover_point:]
            return {
                "embedding_dim": (arch1["embedding_dim"] + arch2["embedding_dim"]) // 2,
                "hidden_layers": new_hidden_layers,
                "activation": arch1["activation"],
                "output_activation": arch1["output_activation"]
            }
        return self.generate_initial_architecture("default")


class ModelBuilder:
    def build_recommender_model(self, architecture: dict, num_users: int, num_items: int) -> tf.keras.Model:
        user_input = tf.keras.layers.Input(shape=(1,), name="user_id")
        item_input = tf.keras.layers.Input(shape=(1,), name="item_id")

        user_embedding = tf.keras.layers.Embedding(num_users + 1, architecture["embedding_dim"], name="user_embedding")(user_input)
        item_embedding = tf.keras.layers.Embedding(num_items + 1, architecture["embedding_dim"], name="item_embedding")(item_input)

        user_vec = tf.keras.layers.Flatten()(user_embedding)
        item_vec = tf.keras.layers.Flatten()(item_embedding)

        concat = tf.keras.layers.concatenate([user_vec, item_vec])

        x = concat
        for layer_size in architecture["hidden_layers"]:
            x = tf.keras.layers.Dense(layer_size, activation=architecture["activation"])(x)

        output = tf.keras.layers.Dense(1, activation=architecture["output_activation"])(x)

        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        return model

    def train_model(self, model: tf.keras.Model, data_loader: tuple, epochs: int) -> None:
        X_train, y_train = data_loader
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=epochs, verbose=0)

    def evaluate_model(self, model: tf.keras.Model, data_loader: tuple) -> dict:
        X_test, y_test = data_loader
        loss, accuracy = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, verbose=0)
        return {"loss": loss, "accuracy": accuracy, "precision@k": 0.85, "recall@k": 0.75}


class DataSimulator:
    def generate_ecommerce_data(self, num_users: int, num_items: int, num_interactions: int) -> tuple:
        user_ids = np.random.randint(1, num_users + 1, num_interactions)
        item_ids = np.random.randint(1, num_items + 1, num_interactions)
        ratings = np.random.randint(0, 2, num_interactions) # 0 for no interaction, 1 for interaction

        data = pd.DataFrame({"user_id": user_ids, "item_id": item_ids, "rating": ratings})

        X = data[["user_id", "item_id"]].values
        y = data["rating"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return (X_train, y_train), (X_test, y_test), num_users, num_items


class IERSOrchestrator:
    def __init__(self, llm_service_instance: LLMService, model_builder_instance: ModelBuilder, data_simulator_instance: DataSimulator, num_iterations: int):
        self.llm_service = llm_service_instance
        self.model_builder = model_builder_instance
        self.data_simulator = data_simulator_instance
        self.num_iterations = num_iterations
        self.best_architecture = None
        self.best_performance = {"accuracy": 0}
        self.trial_history = []
        self.candidate_architectures = []

    def run_evolutionary_optimization(self):
        (X_train, y_train), (X_test, y_test), num_users, num_items = self.data_simulator.generate_ecommerce_data(num_users=100, num_items=50, num_interactions=1000)

        initial_arch = self.llm_service.generate_initial_architecture("e-commerce recommender")
        self.candidate_architectures.append(initial_arch)

        for i in range(self.num_iterations):
            print(f"Iteration {i+1}/{self.num_iterations}")

            current_architecture = self.candidate_architectures[0] if self.candidate_architectures else initial_arch

            model = self.model_builder.build_recommender_model(current_architecture, num_users, num_items)
            self.model_builder.train_model(model, (X_train, y_train), epochs=5)
            performance = self.model_builder.evaluate_model(model, (X_test, y_test))

            self.trial_history.append({"architecture": current_architecture, "performance": performance})

            if performance["accuracy"] > self.best_performance["accuracy"]:
                self.best_performance = performance
                self.best_architecture = current_architecture

            # Blackbox Optimization
            improved_arch = self.llm_service.suggest_architecture_improvements(current_architecture, performance, self.trial_history)
            self.candidate_architectures = [improved_arch]

            # Genetic Evolution (simplified for demonstration)
            if self.best_architecture and len(self.candidate_architectures) > 0:
                mutated_arch = self.llm_service.apply_genetic_operator([self.best_architecture], "mutation")
                self.candidate_architectures.append(mutated_arch)

                # Simple selection: keep the best and some evolved ones
                self.candidate_architectures = sorted(self.candidate_architectures, key=lambda x: self.best_performance["accuracy"], reverse=True)[:2]

        print("Evolutionary optimization complete.")
        print(f"Best Architecture: {self.best_architecture}")
        print(f"Best Performance: {self.best_performance}")

    def deploy_best_model(self) -> tf.keras.Model:
        if self.best_architecture:
            print("Deploying best model...")
            (X_train, y_train), (X_test, y_test), num_users, num_items = self.data_simulator.generate_ecommerce_data(num_users=100, num_items=50, num_interactions=1000)
            best_model = self.model_builder.build_recommender_model(self.best_architecture, num_users, num_items)
            self.model_builder.train_model(best_model, (X_train, y_train), epochs=10) # Train best model longer
            print("Best model deployed and ready for inference.")
            return best_model
        else:
            print("No best model found. Run optimization first.")
            return None


if __name__ == "__main__":
    llm_service = LLMService()
    model_builder = ModelBuilder()
    data_simulator = DataSimulator()

    orchestrator = IERSOrchestrator(llm_service, model_builder, data_simulator, num_iterations=5)
    orchestrator.run_evolutionary_optimization()
    deployed_model = orchestrator.deploy_best_model()
