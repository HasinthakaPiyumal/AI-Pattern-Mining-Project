from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

class ExemplarSelectionModule:
    def __init__(self, embedding_service, k_clusters=5, num_exemplars_to_propose=3):
        self.embedding_service = embedding_service
        self.k_clusters = k_clusters
        self.num_exemplars_to_propose = num_exemplars_to_propose

    def propose_exemplars_for_annotation(self, unlabeled_queries):
        if not unlabeled_queries:
            return []

        unlabeled_embeddings = self.embedding_service.get_embeddings(unlabeled_queries)
        
        if len(unlabeled_queries) < self.k_clusters:
            self.k_clusters = len(unlabeled_queries)
            
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10)
        kmeans.fit(unlabeled_embeddings)

        proposed_exemplars = []
        for i in range(self.k_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if cluster_indices.size > 0:
                cluster_points = unlabeled_embeddings[cluster_indices]
                centroid = kmeans.cluster_centers_[i]
                distances = euclidean_distances(cluster_points, centroid.reshape(1, -1))
                closest_index_in_cluster = np.argmin(distances)
                original_index = cluster_indices[closest_index_in_cluster]
                proposed_exemplars.append(unlabeled_queries[original_index])
        
        # If we need more exemplars than clusters, or fewer if the data is small
        # Take the top 'num_exemplars_to_propose' diverse exemplars (e.g., from different clusters)
        # For simplicity, we'll just take from the proposed_exemplars list, ensuring no duplicates
        return list(dict.fromkeys(proposed_exemplars))[:self.num_exemplars_to_propose]

class FewShotPromptingModule:
    def __init__(self, embedding_service, k_neighbors=3):
        self.embedding_service = embedding_service
        self.k_neighbors = k_neighbors

    def construct_few_shot_prompt(self, new_query, labeled_exemplars):
        if not labeled_exemplars:
            return f"User: {new_query}\nAgent:"

        labeled_queries = [ex["query"] for ex in labeled_exemplars]
        labeled_responses = [ex["response"] for ex in labeled_exemplars]

        labeled_embeddings = self.embedding_service.get_embeddings(labeled_queries)
        new_query_embedding = self.embedding_service.get_embeddings([new_query])[0]

        if len(labeled_exemplars) < self.k_neighbors:
            self.k_neighbors = len(labeled_exemplars)

        nn = NearestNeighbors(n_neighbors=self.k_neighbors, metric="cosine")
        nn.fit(labeled_embeddings)
        
        distances, indices = nn.kneighbors(new_query_embedding.reshape(1, -1))
        
        prompt_parts = []
        for i in indices[0]:
            prompt_parts.append(f"User: {labeled_queries[i]}\nAgent: {labeled_responses[i]}")
        
        return "\n\n".join(prompt_parts) + f"\n\nUser: {new_query}\nAgent:"

class LLMService:
    def generate_response(self, prompt):
        # Simulate LLM response for demonstration
        if "reset my password" in prompt.lower():
            return "I can help you reset your password. Please visit our website's 'Forgot Password' link or call our support line for assistance."
        elif "shipping status" in prompt.lower():
            return "To check your shipping status, please provide your order number. You can also track it directly on our website."
        elif "product refund" in prompt.lower():
            return "For a product refund, please review our return policy on the website. You may be eligible for a refund within 30 days of purchase."
        elif "technical issue" in prompt.lower() or " troubleshoot" in prompt.lower():
            return "I understand you're experiencing a technical issue. Could you please describe the problem in more detail? I'll do my best to help you troubleshoot."
        else:
            # A very simple 'default' response based on the last query in the prompt
            last_query_start = prompt.rfind("\n\nUser: ") + len("\n\nUser: ")
            last_query_end = prompt.rfind("\nAgent:")
            if last_query_start != -1 and last_query_end != -1 and last_query_start < last_query_end:
                last_query = prompt[last_query_start:last_query_end].strip()
                return f"Thank you for contacting support regarding '{last_query}'. I'm looking into this for you. Please hold on a moment."
            return "I received your query. How can I assist you further?"

def main():
    embedding_service = EmbeddingService()
    exemplar_selector = ExemplarSelectionModule(embedding_service, k_clusters=5, num_exemplars_to_propose=3)
    few_shot_prompter = FewShotPromptingModule(embedding_service, k_neighbors=3)
    llm_service = LLMService()

    # --- Data Storage (Simulated) ---
    unlabeled_queries = [
        "My internet is not working. What should I do?",
        "How can I change my billing address?",
        "I need help with my account login. I forgot my username.",
        "Where is my order? I ordered last week.",
        "The mobile app keeps crashing after the update.",
        "Can I get a refund for a service I cancelled?",
        "My device is making a strange noise. Is it broken?",
        "How do I upgrade my subscription plan?",
        "I can't access my email. What's wrong?",
        "When will my delivery arrive?",
        "I want to report a bug in your software.",
        "Is it possible to pause my subscription?"
    ]

    labeled_exemplars = [
        {"query": "I forgot my password, how can I reset it?", "response": "You can reset your password by clicking 'Forgot Password' on the login page or through your account settings."},
        {"query": "What is the status of my recent order?", "response": "Please provide your order number, and I can check the status for you."},
        {"query": "My product arrived damaged, what's the return policy?", "response": "We apologize for the inconvenience. Please visit our returns page for instructions on how to return a damaged item."},
        {"query": "How do I contact customer support by phone?", "response": "Our customer support line is available Monday to Friday, 9 AM to 5 PM. The number is 1-800-XXX-XXXX."},
    ]

    print("Welcome to the Automated Customer Support AI Demo!")

    while True:
        print("\n--- Menu ---")
        print("1. Propose new exemplars for human annotation (Stage 1)")
        print("2. Ask a new customer query (Stage 2)")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            print("\nProposing diverse unlabeled queries for annotation...")
            proposed = exemplar_selector.propose_exemplars_for_annotation(unlabeled_queries)
            
            if not proposed:
                print("No unlabeled queries to propose.")
                continue

            print("Proposed queries for human annotation:")
            for i, q in enumerate(proposed):
                print(f"{i+1}. {q}")

            print("\n--- Human Annotation (Simulated) ---")
            for q in proposed:
                response = input(f"Please provide a response for '{q}': ")
                labeled_exemplars.append({"query": q, "response": response})
                # Remove from unlabeled after it's been 'labeled'
                if q in unlabeled_queries:
                    unlabeled_queries.remove(q)
            print("Exemplars added to the labeled pool.")

        elif choice == "2":
            new_query = input("\nEnter a new customer query: ")
            if not new_query.strip():
                print("Query cannot be empty.")
                continue

            prompt = few_shot_prompter.construct_few_shot_prompt(new_query, labeled_exemplars)
            print("\n--- Generated Few-Shot Prompt ---")
            print(prompt)
            print("-----------------------------------")

            print("\n--- LLM Generating Response ---")
            response = llm_service.generate_response(prompt)
            print("\nAgent: ", response)

        elif choice == "3":
            print("Exiting demo. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()