""" 
AI-powered Customer Support Ticket Classification System with Dynamic Exemplar Ordering.

This system demonstrates how to optimize the order of few-shot exemplars presented to 
a Large Language Model (LLM) for ticket classification. It implements different ordering 
strategies (random, semantic similarity, diversity) and tracks their performance to 
dynamically select the best strategy for incoming support tickets.
"""

import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time # For simulating LLM delay

# --- Mock Components for Demonstration ---

class MockEmbeddingModel:
    """A mock embedding model to simulate text embeddings."""
    def __init__(self, vocab=None):
        # A small vocabulary for demonstration. In a real app, this would be a pre-trained model.
        self.vocab = vocab if vocab else ["issue", "login", "payment", "refund", "delivery", "technical", "billing", "account", "order"]
        self.embedding_dim = 10
        # Assign random vectors to words in the vocabulary
        self.word_embeddings = {word: np.random.rand(self.embedding_dim) for word in self.vocab}
        self.word_embeddings["<UNK>"] = np.random.rand(self.embedding_dim) # Embedding for unknown words

    def encode(self, texts):
        """Generates mock embeddings for a list of texts by averaging word embeddings."""
        embeddings = []
        for text in texts:
            words = text.lower().split()
            sentence_embedding = np.zeros(self.embedding_dim)
            count = 0
            for word in words:
                if word in self.word_embeddings:
                    sentence_embedding += self.word_embeddings[word]
                    count += 1
                else:
                    sentence_embedding += self.word_embeddings["<UNK>"] 
                    count += 1
            if count > 0:
                embeddings.append(sentence_embedding / count)
            else:
                embeddings.append(np.zeros(self.embedding_dim)) # Handle empty text
        return np.array(embeddings)

class MockLLMInterface:
    """A mock LLM for few-shot classification, simulating an actual LLM call."""
    def __init__(self, possible_labels, mock_delay=0.05):
        self.possible_labels = possible_labels # The set of possible output labels
        self.mock_delay = mock_delay # Simulate network latency or processing time

    def _generate_few_shot_prompt(self, query, exemplars):
        """Constructs a simplified few-shot prompt string."""
        prompt_parts = []
        for ex in exemplars:
            prompt_parts.append(f"Ticket: {ex['text']}\nLabel: {ex['label']}")
        prompt_parts.append(f"Ticket: {query}\nLabel:")
        return "\n---\n".join(prompt_parts)

    def classify(self, query_text, exemplars, true_label=None, strategy_name="unknown"):
        """Simulates LLM classification. Accuracy is influenced by the 'strategy_name'."""
        time.sleep(self.mock_delay) # Simulate API call delay

        _ = self._generate_few_shot_prompt(query_text, exemplars) # Prompt is generated but not actually used by mock

        predicted_label = random.choice(self.possible_labels) # Default: random guess

        if true_label:
            # Simulate better performance for 'semantic_similarity' strategy
            # This makes the demonstration more illustrative of the pattern's value.
            is_good_ordering_strategy = (strategy_name == "semantic_similarity")

            if is_good_ordering_strategy and random.random() < 0.8: # 80% chance for correct prediction
                predicted_label = true_label
            elif not is_good_ordering_strategy and random.random() < 0.3: # 30% chance for other strategies
                predicted_label = true_label
        
        return predicted_label

# --- Core Components of the Classification System ---

class ExemplarManager:
    """Manages the collection of labeled few-shot exemplars."""
    def __init__(self, exemplars):
        # exemplars is a list of dictionaries: [{'text': '...', 'label': '...'}, ...]
        self.exemplars = list(exemplars)

    def get_exemplars(self):
        """Returns all stored exemplars."""
        return self.exemplars

    def add_exemplar(self, text, label):
        """Adds a new exemplar to the manager's collection."""
        self.exemplars.append({"text": text, "label": label})
        print(f"Added new exemplar: '{text}' -> '{label}'")

class OrderingStrategies:
    """Provides various methods for ordering few-shot exemplars."""

    @staticmethod
    def random_order(exemplars):
        """Randomly shuffles the given list of exemplars."""
        ordered_exemplars = list(exemplars)
        random.shuffle(ordered_exemplars)
        return ordered_exemplars

    @staticmethod
    def semantic_similarity_order(query_text, exemplars, embedding_model):
        """Orders exemplars by their semantic similarity to the query text (most similar first)."""
        if not exemplars:
            return []

        exemplar_texts = [ex['text'] for ex in exemplars]
        
        query_embedding = embedding_model.encode([query_text])[0]
        exemplar_embeddings = embedding_model.encode(exemplar_texts)

        similarities = cosine_similarity([query_embedding], exemplar_embeddings)[0]
        
        # Pair exemplars with their similarity scores and sort in descending order
        exemplars_with_sim = sorted(
            zip(exemplars, similarities),
            key=lambda x: x[1],
            reverse=True 
        )
        return [ex for ex, sim in exemplars_with_sim]

    @staticmethod
    def diversity_order(exemplars, embedding_model, top_n=5):
        """ 
        Orders exemplars to maximize diversity. 
        Simplified approach: attempts to select exemplars with unique labels first, 
        then fills with random if more are needed, up to `top_n`.
        A more robust implementation might use clustering or Maximum Mean Discrepancy (MMD).
        """
        if not exemplars:
            return []
        
        unique_labels = list(set(ex['label'] for ex in exemplars))
        diverse_set = []
        remaining_exemplars = list(exemplars)

        # Prioritize one exemplar per unique label
        for label in unique_labels:
            candidates = [ex for ex in remaining_exemplars if ex['label'] == label]
            if candidates:
                chosen_exemplar = random.choice(candidates)
                diverse_set.append(chosen_exemplar)
                remaining_exemplars.remove(chosen_exemplar)
                if len(diverse_set) >= top_n:
                    break
        
        # Fill remaining spots with random exemplars if `top_n` not reached
        while len(diverse_set) < top_n and remaining_exemplars:
            chosen_exemplar = random.choice(remaining_exemplars)
            diverse_set.append(chosen_exemplar)
            remaining_exemplars.remove(chosen_exemplar)

        return diverse_set


class TicketClassifier:
    """
    Orchestrates the ticket classification process, dynamically selecting 
    the best exemplar ordering strategy.
    """
    def __init__(self, initial_exemplars, possible_labels, num_few_shot_exemplars=3):
        self.exemplar_manager = ExemplarManager(initial_exemplars)
        # Initialize embedding model with a vocabulary derived from initial exemplars and labels
        all_text_words = " ".join([ex['text'] for ex in initial_exemplars]).lower().split()
        self.embedding_model = MockEmbeddingModel(vocab=list(set(all_text_words + possible_labels)))
        self.llm_interface = MockLLMInterface(possible_labels=possible_labels)
        self.num_few_shot_exemplars = num_few_shot_exemplars

        self.available_strategies = {
            "random": OrderingStrategies.random_order,
            "semantic_similarity": OrderingStrategies.semantic_similarity_order,
            "diversity": OrderingStrategies.diversity_order, 
        }
        self.strategy_performance = {
            name: {"correct": 0, "total": 0} for name in self.available_strategies
        } # Tracks correct/total predictions for each strategy
        self.current_best_strategy = "random" # Default starting strategy

        print(f"Initialized TicketClassifier with {len(initial_exemplars)} exemplars.")
        print(f"Available ordering strategies: {list(self.available_strategies.keys())}")

    def _get_best_exemplars_for_strategy(self, strategy_name, query_text):
        """Selects and orders a subset of exemplars using the specified strategy."""
        all_exemplars = self.exemplar_manager.get_exemplars()
        
        if strategy_name == "random":
            ordered_exemplars = OrderingStrategies.random_order(all_exemplars)
        elif strategy_name == "semantic_similarity":
            ordered_exemplars = OrderingStrategies.semantic_similarity_order(query_text, all_exemplars, self.embedding_model)
        elif strategy_name == "diversity":
            # For diversity, we might want to select from a larger pool to ensure actual diversity
            ordered_exemplars = OrderingStrategies.diversity_order(all_exemplars, self.embedding_model, top_n=self.num_few_shot_exemplars * 2)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Return only the required number of few-shot exemplars
        return ordered_exemplars[:self.num_few_shot_exemplars]


    def _evaluate_strategy_on_ticket(self, strategy_name, query_ticket_text, true_label=None):
        """
        Applies a specific ordering strategy, gets LLM prediction, and 
        optionally updates performance metrics.
        """
        selected_exemplars = self._get_best_exemplars_for_strategy(strategy_name, query_ticket_text)
        
        # Get LLM prediction from the mock interface
        predicted_label = self.llm_interface.classify(
            query_ticket_text,
            selected_exemplars,
            true_label=true_label,
            strategy_name=strategy_name
        )

        # Update performance metrics if a true label is provided for evaluation
        if true_label is not None:
            self.strategy_performance[strategy_name]["total"] += 1
            if predicted_label == true_label:
                self.strategy_performance[strategy_name]["correct"] += 1
        
        return predicted_label

    def _update_best_strategy(self):
        """Selects the best performing strategy based on cumulative accuracy."""
        best_accuracy = -1.0
        new_best_strategy = self.current_best_strategy

        for strategy_name, stats in self.strategy_performance.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    new_best_strategy = strategy_name
            elif stats["total"] == 0 and best_accuracy == -1.0: 
                # If no data yet, default to the first strategy encountered or 'random'
                new_best_strategy = strategy_name
                best_accuracy = 0.0 

        if new_best_strategy != self.current_best_strategy:
            print(f"Strategy update: '{self.current_best_strategy}' -> '{new_best_strategy}' (Accuracy: {best_accuracy:.2f})")
            self.current_best_strategy = new_best_strategy
        
        return self.current_best_strategy

    def classify_ticket(self, query_ticket_text, true_label=None, evaluate_all_strategies=True):
        """
        Classifies an incoming ticket. If `evaluate_all_strategies` is True, it tries 
        all strategies for learning; otherwise, it uses the current best strategy.
        
        Args:
            query_ticket_text (str): The text of the incoming support ticket.
            true_label (str, optional): The actual label, used for performance evaluation.
            evaluate_all_strategies (bool): If True, all strategies are evaluated to update 
                                            performance metrics. If False, only the 
                                            `current_best_strategy` is used.

        Returns:
            tuple: (predicted_label, chosen_strategy_name)
        """
        if evaluate_all_strategies:
            predictions_per_strategy = {}
            for strategy_name in self.available_strategies:
                predicted = self._evaluate_strategy_on_ticket(strategy_name, query_ticket_text, true_label)
                predictions_per_strategy[strategy_name] = predicted
            
            # After all strategies are evaluated, update the best one based on cumulative performance
            chosen_strategy = self._update_best_strategy()
            predicted_label = predictions_per_strategy[chosen_strategy] # The prediction from the chosen strategy
            
            if true_label is not None:
                print(f"\nTicket: '{query_ticket_text}' (True: {true_label})")
                for strategy_name, pred in predictions_per_strategy.items():
                    correct_status = "CORRECT" if pred == true_label else "INCORRECT"
                    print(f"  Strategy '{strategy_name}': Predicted '{pred}' ({correct_status})")
                print(f"  Chosen strategy for this ticket: '{chosen_strategy}' -> Final Predicted: '{predicted_label}'")
            
        else:
            # Use only the current best strategy for classification
            chosen_strategy = self.current_best_strategy
            predicted_label = self._evaluate_strategy_on_ticket(chosen_strategy, query_ticket_text, true_label)
            if true_label is not None:
                correct_status = "CORRECT" if predicted_label == true_label else "INCORRECT"
                print(f"\nTicket: '{query_ticket_text}' (True: {true_label}) -> Chosen strategy '{chosen_strategy}' Predicted: '{predicted_label}' ({correct_status})")

        return predicted_label, chosen_strategy

    def get_strategy_accuracies(self):
        """Returns the current accuracy for each strategy."""
        accuracies = {}
        for strategy_name, stats in self.strategy_performance.items():
            if stats["total"] > 0:
                accuracies[strategy_name] = stats["correct"] / stats["total"]
            else:
                accuracies[strategy_name] = 0.0 # No evaluations yet
        return accuracies

# --- Demonstration Function ---
def run_ticket_classification_demo():
    print("--- Starting AI Customer Support Ticket Classification System Demo ---")

    # 1. Define initial exemplars (few-shot examples)
    initial_exemplars = [
        {"text": "My internet is not working at all, can't connect.", "label": "Technical Issue"},
        {"text": "I need to reset my password for my account.", "label": "Account Management"},
        {"text": "Where is my order? It was supposed to arrive yesterday.", "label": "Order Tracking"},
        {"text": "I want to change my billing address.", "label": "Billing"},
        {"text": "The app crashed after the last update.", "label": "Technical Issue"},
        {"text": "How do I upgrade my subscription plan?", "label": "Account Management"},
        {"text": "I was charged twice for the same item.", "label": "Billing"},
        {"text": "The delivery was incomplete, missing an item.", "label": "Order Tracking"},
        {"text": "I can't log in to my dashboard.", "label": "Account Management"},
        {"text": "My device keeps disconnecting from Wi-Fi.", "label": "Technical Issue"},
    ]
    possible_labels = list(set(ex['label'] for ex in initial_exemplars))

    # 2. Initialize the Ticket Classifier system
    classifier = TicketClassifier(initial_exemplars, possible_labels, num_few_shot_exemplars=3)

    print("\n--- Simulating Incoming Tickets and Dynamic Exemplar Ordering ---")

    # Simulate a series of incoming tickets for classification and evaluation
    incoming_tickets = [
        {"text": "My router is not connecting to the internet.", "true_label": "Technical Issue"},
        {"text": "I forgot my username and password.", "true_label": "Account Management"},
        {"text": "Can you tell me the status of my recent purchase?", "true_label": "Order Tracking"},
        {"text": "There's an error on my latest invoice.", "true_label": "Billing"},
        {"text": "My payment failed, please assist.", "true_label": "Billing"},
        {"text": "The software is crashing constantly.", "true_label": "Technical Issue"},
        {"text": "How do I add a new user to my account?", "true_label": "Account Management"},
        {"text": "I need to know when my package will arrive.", "true_label": "Order Tracking"},
        {"text": "My subscription details are incorrect.", "true_label": "Billing"},
        {"text": "The website is down.", "true_label": "Technical Issue"},
        {"text": "I want to cancel my recurring payment.", "true_label": "Billing"},
    ]

    for i, ticket in enumerate(incoming_tickets):
        print(f"\n--- Processing Ticket {i+1}/{len(incoming_tickets)} ---")
        # For each ticket, evaluate all strategies to update their performance metrics
        predicted_label, chosen_strategy = classifier.classify_ticket(ticket["text"], ticket["true_label"], evaluate_all_strategies=True)
        print(f"Current strategy accuracies: {classifier.get_strategy_accuracies()}")
        print(f"Chosen best strategy overall: {classifier.current_best_strategy}")

    print("\n--- Demo Complete ---")
    print("\nFinal Strategy Accuracies:")
    for strategy, accuracy in classifier.get_strategy_accuracies().items():
        print(f"  '{strategy}': {accuracy:.2f}")
    print(f"\nOverall best performing strategy: {classifier.current_best_strategy}")

# Entry point for the demo
if __name__ == "__main__":
    run_ticket_classification_demo()
