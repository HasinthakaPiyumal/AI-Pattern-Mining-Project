from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np

class EmbeddingModule:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text)

class KNNExemplarSelector:
    def __init__(self, historical_embeddings, historical_data):
        self.historical_embeddings = historical_embeddings
        self.historical_data = historical_data
        self.knn_model = NearestNeighbors(n_neighbors=1, metric='cosine') # n_neighbors will be set dynamically in select_exemplars
        if len(historical_embeddings) > 0:
            self.knn_model.fit(historical_embeddings)

    def select_exemplars(self, query_embedding, k=3):
        if len(self.historical_embeddings) == 0:
            return []
        distances, indices = self.knn_model.kneighbors(query_embedding.reshape(1, -1), n_neighbors=min(k, len(self.historical_embeddings)))
        return [self.historical_data[i] for i in indices[0]]

def build_few_shot_prompt(query, exemplars):
    prompt_parts = []
    for i, ex in enumerate(exemplars):
        prompt_parts.append(f"Customer: {ex['query']}\nChatbot: {ex['response']}")
    
    prompt = """The following are examples of customer interactions and chatbot responses. Use them to answer the new customer query.

""" + "\n\n".join(prompt_parts) + f"\n\nCustomer: {query}\nChatbot:"
    return prompt

# Main Chatbot Logic
if __name__ == "__main__":
    # 1. Dummy Historical Dataset (Dtrain)
    Dtrain = [
        {'query': 'My internet is not working.', 'response': 'Please try restarting your router. If the issue persists, contact technical support.'},
        {'query': 'How can I change my billing address?', 'response': 'You can update your billing address in your account settings under the "Billing Information" section.'},
        {'query': 'What are your operating hours?', 'response': 'Our customer support is available 24/7.'},
        {'query': 'I need to upgrade my plan.', 'response': 'You can upgrade your plan through your account portal or by speaking with a sales representative.'},
        {'query': 'I forgot my password.', 'response': 'You can reset your password by clicking on the "Forgot Password" link on the login page.'},
    ]

    # 2. Initialize Embedding Module
    embedding_module = EmbeddingModule()

    # 3. Generate embeddings for Dtrain
    print("Generating embeddings for historical data...")
    Dtrain_queries = [item['query'] for item in Dtrain]
    Dtrain_embeddings = np.array([embedding_module.get_embedding(query) for query in Dtrain_queries])
    print("Embeddings generated.")

    # 4. Initialize KNN Exemplar Selector
    knn_selector = KNNExemplarSelector(Dtrain_embeddings, Dtrain)
    print("KNN Exemplar Selector initialized.")

    # 5. New Customer Query (Dtest_xi)
    new_customer_query = "I can't log in to my account, I think I forgot my password. Can you help me?"
    print(f"\nNew Customer Query: {new_customer_query}")

    # 6. Generate embedding for Dtest_xi
    query_embedding = embedding_module.get_embedding(new_customer_query)

    # 7. Select relevant exemplars (e.g., k=2)
    k_exemplars = 2
    selected_exemplars = knn_selector.select_exemplars(query_embedding, k=k_exemplars)
    print(f"\nSelected {len(selected_exemplars)} exemplars:")
    for ex in selected_exemplars:
        print(f"  - Query: {ex['query']} -> Response: {ex['response']}")

    # 8. Build Few-Shot Prompt
    few_shot_prompt = build_few_shot_prompt(new_customer_query, selected_exemplars)
    print("\n--- Generated Few-Shot Prompt ---")
    print(few_shot_prompt)
    print("---------------------------------")

    # 9. Simulate LLM Response
    # In a real application, you would send this 'few_shot_prompt' to an LLM API (e.g., OpenAI, Gemini, etc.)
    # For demonstration, we'll just show the prompt.
    print("\n(Simulating LLM response based on the above prompt...) ")
    simulated_llm_response = "It sounds like you forgot your password. You can reset it by visiting the login page and clicking on the 'Forgot Password' link. If you continue to have trouble, please let me know!"
    print(f"Simulated Chatbot Response: {simulated_llm_response}")
