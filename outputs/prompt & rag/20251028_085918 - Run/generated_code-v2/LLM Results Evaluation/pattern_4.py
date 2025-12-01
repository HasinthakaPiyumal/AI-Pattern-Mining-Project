import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import chromadb
import random

class VectorStoreManager:
    def __init__(self, collection_name="exemplars"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_exemplars(self, ids, texts, embeddings, metadatas=None):
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def query_exemplars(self, query_embedding, num_results=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
            include=
            ["documents", "embeddings"]
        )
        return results

    def get_all_embeddings_and_texts(self):
        results = self.collection.get(ids=self.collection.get(include=[])["ids"], include=["documents", "embeddings"])
        return results["documents"], results["embeddings"]

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts).tolist()

class ExemplarProposalService:
    def __init__(self, embedding_generator, vector_store_manager, n_clusters=5, num_proposals=3):
        self.embedding_generator = embedding_generator
        self.vector_store_manager = vector_store_manager
        self.n_clusters = n_clusters
        self.num_proposals = num_proposals

    def propose_exemplars(self, unlabeled_queries):
        if not unlabeled_queries:
            return []

        unlabeled_embeddings = self.embedding_generator.generate_embeddings(unlabeled_queries)
        
        if len(unlabeled_queries) < self.n_clusters:
            kmeans = KMeans(n_clusters=len(unlabeled_queries), random_state=42, n_init=10)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        
        kmeans.fit(unlabeled_embeddings)

        proposals = []
        for i in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) > 0:
                # Select a representative from each cluster (e.g., closest to centroid)
                centroid = kmeans.cluster_centers_[i]
                distances = [np.linalg.norm(np.array(unlabeled_embeddings[idx]) - centroid) for idx in cluster_indices]
                closest_idx_in_cluster = cluster_indices[np.argmin(distances)]
                proposals.append((unlabeled_queries[closest_idx_in_cluster], unlabeled_embeddings[closest_idx_in_cluster]))
        
        # If we need more proposals than clusters, or just want diverse selection
        if len(proposals) < self.num_proposals and len(unlabeled_queries) > len(proposals):
            remaining_queries = [(q, e) for q, e in zip(unlabeled_queries, unlabeled_embeddings) if (q, e) not in proposals]
            random.shuffle(remaining_queries)
            proposals.extend(remaining_queries[:self.num_proposals - len(proposals)])

        return [p[0] for p in proposals[:self.num_proposals]] # Return only the query text

class HumanAnnotationSimulator:
    def __init__(self):
        self.labeled_data = []

    def annotate(self, query):
        print(f"Human annotator: Please provide a smart reply for: '{query}'")
        response = input("Enter smart reply: ")
        self.labeled_data.append({"query": query, "response": response})
        return {"query": query, "response": response}

    def get_labeled_data(self):
        return self.labeled_data

class MockLLM:
    def generate_response(self, prompt):
        if "reset my password" in prompt.lower():
            return "Please visit our website's password reset page or contact support for assistance." 
        elif "shipping status" in prompt.lower() or "delivery" in prompt.lower():
            return "To check your shipping status, please provide your order number. You can find it in your confirmation email." 
        elif "refund" in prompt.lower():
            return "For refund requests, please review our refund policy on our website. You can typically initiate a refund from your order history." 
        else:
            return "I understand you have a question. Could you please provide more details so I can assist you better?"

class FewShotPromptingService:
    def __init__(self, embedding_generator, vector_store_manager, llm, k_similar=10, m_diverse=3):
        self.embedding_generator = embedding_generator
        self.vector_store_manager = vector_store_manager
        self.llm = llm
        self.k_similar = k_similar
        self.m_diverse = m_diverse

    def _select_diverse_exemplars(self, query_embedding, top_k_exemplars):
        if not top_k_exemplars:
            return []
        
        exemplar_embeddings = np.array([e[2] for e in top_k_exemplars]) # Assuming exemplar is (id, text, embedding)
        selected_indices = []
        selected_exemplars = []

        # Start with the most similar exemplar
        max_similarity_idx = np.argmax(cosine_similarity([query_embedding], exemplar_embeddings)[0])
        selected_indices.append(max_similarity_idx)
        selected_exemplars.append(top_k_exemplars[max_similarity_idx])

        for _ in range(1, min(self.m_diverse, len(top_k_exemplars))):
            best_candidate_idx = -1
            max_min_similarity = -1

            for i in range(len(top_k_exemplars)):
                if i in selected_indices:
                    continue
                
                candidate_embedding = exemplar_embeddings[i]
                # Calculate similarity of candidate to already selected exemplars
                similarities_to_selected = cosine_similarity([candidate_embedding], np.array([exemplar_embeddings[idx] for idx in selected_indices]))[0]
                min_similarity_to_selected = np.min(similarities_to_selected)

                # MMR-like score: (similarity to query) - lambda * (max similarity to selected)
                # Simplified: maximize (similarity to query - similarity to already selected)
                # Even simpler: pick the one that is most similar to query AND least similar to already selected (min_similarity_to_selected is actually max_similarity_to_selected if we are looking for diversity)
                
                # Let's use a simpler heuristic: find the one that has the highest similarity to the query but is also most 