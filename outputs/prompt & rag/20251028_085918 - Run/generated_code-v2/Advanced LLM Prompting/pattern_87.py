import random
import math

class MultilingualEmbedder:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim

    def embed_text(self, text):
        """Mocks embedding generation. In a real scenario, this would use a sentence-transformer model."""
        random.seed(hash(text)) # Deterministic random for a given text for mock purposes
        return [random.uniform(-1, 1) for _ in range(self.embedding_dim)]

class ExampleStore:
    def __init__(self, embedder):
        self.examples = []
        self.embedder = embedder

    def add_example(self, text, language, label):
        embedding = self.embedder.embed_text(text)
        self.examples.append({
            "text": text,
            "language": language,
            "label": label,
            "embedding": embedding
        })

    def get_all_examples(self):
        return self.examples

class XInSTA_PromptGenerator:
    def __init__(self, example_store):
        self.example_store = example_store

    def _dot_product(self, vec1, vec2):
        return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

    def _norm(self, vec):
        return math.sqrt(sum(v * v for v in vec))

    def _cosine_similarity(self, vec1, vec2):
        norm1 = self._norm(vec1)
        norm2 = self._norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return self._dot_product(vec1, vec2) / (norm1 * norm2)

    def generate_prompt(self, query_text, query_embedding, alignment_strategy, k=3, predicted_label=None):
        all_examples = self.example_store.get_all_examples()
        selected_examples = []

        if alignment_strategy == "semantic":
            similarities = []
            for ex in all_examples:
                sim = self._cosine_similarity(query_embedding, ex["embedding"])
                similarities.append((sim, ex))
            similarities.sort(key=lambda x: x[0], reverse=True)
            selected_examples = [ex for sim, ex in similarities[:k]]

        elif alignment_strategy == "task_based":
            if predicted_label is None:
                raise ValueError("predicted_label must be provided for 'task_based' alignment.")
            task_examples = [ex for ex in all_examples if ex["label"] == predicted_label]
            selected_examples = task_examples[:k] # Take top k if many exist

        elif alignment_strategy == "combined":
            if predicted_label is None:
                raise ValueError("predicted_label must be provided for 'combined' alignment.")

            # Semantic filtering first
            similarities = []
            for ex in all_examples:
                sim = self._cosine_similarity(query_embedding, ex["embedding"])
                similarities.append((sim, ex))
            similarities.sort(key=lambda x: x[0], reverse=True)
            semantically_similar = [ex for sim, ex in similarities[:k*2]] # Get more to allow for label filtering

            # Then task-based filtering from semantically similar
            combined_examples = [ex for ex in semantically_similar if ex["label"] == predicted_label]
            selected_examples = combined_examples[:k]

        else:
            raise ValueError(f"Unknown alignment strategy: {alignment_strategy}")

        prompt_parts = []
        for ex in selected_examples:
            prompt_parts.append(f"Example - Query: {ex['text']} (Language: {ex['language']})\nExample - Label: {ex['label']}")
        prompt_parts.append(f"\nNew Query: {query_text}\nClassify the New Query and provide only the label:")

        return "\n".join(prompt_parts)

class LLM_Interface:
    def classify_query(self, prompt):
        """Mocks an LLM classification call."""
        # In a real application, this would call an actual LLM API
        if "billing" in prompt.lower() or "invoice" in prompt.lower():
            return "Billing Issue"
        elif "delivery" in prompt.lower() or "shipping" in prompt.lower():
            return "Delivery Inquiry"
        elif "product" in prompt.lower() or "item" in prompt.lower():
            return "Product Information"
        else:
            return "General Inquiry"

class MultilingualChatbotOrchestrator:
    def __init__(self, embedder, example_store, prompt_generator, llm_interface):
        self.embedder = embedder
        self.example_store = example_store
        self.prompt_generator = prompt_generator
        self.llm_interface = llm_interface

    def handle_query(self, query_text, language, alignment_strategy, predicted_label=None):
        print(f"\n--- Handling new query: '{query_text}' (Language: {language}) using '{alignment_strategy}' alignment ---")

        # 1. Embed the query
        query_embedding = self.embedder.embed_text(query_text)

        # 2. Generate optimized prompt with aligned examples
        try:
            prompt = self.prompt_generator.generate_prompt(
                query_text,
                query_embedding,
                alignment_strategy,
                predicted_label=predicted_label
            )
            print("Generated Prompt:\n---\n" + prompt + "\n---")
        except ValueError as e:
            print(f"Error generating prompt: {e}")
            return f"Error: {e}"

        # 3. Get classification from LLM
        classification = self.llm_interface.classify_query(prompt)
        print(f"LLM Classification: {classification}")
        return classification

# --- Demonstration --- 
if __name__ == "__main__":
    # Initialize components
    embedder = MultilingualEmbedder()
    example_store = ExampleStore(embedder)

    # Add some multilingual examples
    example_store.add_example("Where is my order?", "en", "Delivery Inquiry")
    example_store.add_example("Mi pedido no ha llegado.", "es", "Delivery Inquiry")
    example_store.add_example("I have a question about my last invoice.", "en", "Billing Issue")
    example_store.add_example("Ich habe eine Frage zu meiner letzten Rechnung.", "de", "Billing Issue")
    example_store.add_example("What are the features of product X?", "en", "Product Information")
    example_store.add_example("Características del producto Y.", "es", "Product Information")
    example_store.add_example("How do I reset my password?", "en", "Account Management")
    example_store.add_example("Comment réinitialiser mon mot de passe?", "fr", "Account Management")

    prompt_generator = XInSTA_PromptGenerator(example_store)
    llm_interface = LLM_Interface()

    chatbot = MultilingualChatbotOrchestrator(
        embedder,
        example_store,
        prompt_generator,
        llm_interface
    )

    # Test queries
    chatbot.handle_query("When will my package arrive?", "en", "semantic")
    chatbot.handle_query("Tengo una duda sobre mi factura.", "es", "task_based", predicted_label="Billing Issue")
    chatbot.handle_query("Wo ist meine Sendung?", "de", "combined", predicted_label="Delivery Inquiry")
    chatbot.handle_query("J'ai besoin d'aide avec mon compte.", "fr", "semantic")
    chatbot.handle_query("Tell me about the new phone.", "en", "combined", predicted_label="Product Information")
    chatbot.handle_query("What is your refund policy?", "en", "semantic")

    # Demonstrate a query where task-based needs a predicted label
    print("\n--- Attempting task_based without predicted_label (expected error) ---")
    chatbot.handle_query("How do I pay my bill?", "en", "task_based")