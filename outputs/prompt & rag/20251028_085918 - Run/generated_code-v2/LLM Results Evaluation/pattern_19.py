"""Implements the K-Nearest Neighbor (KNN) algorithm for exemplar selection."""

from sklearn.neighbors import NearestNeighbors
import numpy as np

class ExemplarSelector:
    def __init__(self, k_neighbors: int):
        self.k_neighbors = k_neighbors
        self.nn_model = None
        self.exemplar_embeddings = None
        self.exemplar_data = None

    def fit(self, exemplar_embeddings: np.ndarray, exemplar_data: list):
        """Fits the KNN model with the historical data embeddings."
        Args:
            exemplar_embeddings: Embeddings of the historical queries (Dtrain).
            exemplar_data: The actual historical data (e.g., queries and responses).
        """
        if not isinstance(exemplar_embeddings, np.ndarray):
            raise ValueError("exemplar_embeddings must be a numpy array.")
        if exemplar_embeddings.ndim != 2:
            raise ValueError("exemplar_embeddings must be a 2D array.")
        if not isinstance(exemplar_data, list):
            raise ValueError("exemplar_data must be a list.")

        self.exemplar_embeddings = exemplar_embeddings
        self.exemplar_data = exemplar_data
        self.nn_model = NearestNeighbors(n_neighbors=self.k_neighbors, metric='cosine')
        self.nn_model.fit(self.exemplar_embeddings)
        print(f"KNN model fitted with {len(self.exemplar_embeddings)} exemplars.")

    def select_exemplars(self, query_embedding: np.ndarray) -> list:
        """Selects the k nearest exemplars for a given query embedding."
        Args:
            query_embedding: The embedding of the current customer query (Dtest_xi).

        Returns:
            A list of selected exemplars (e.g., historical query-response pairs).
        """
        if self.nn_model is None:
            raise RuntimeError("KNN model has not been fitted yet. Call .fit() first.")
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("query_embedding must be a numpy array.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1) # Reshape for single query

        distances, indices = self.nn_model.kneighbors(query_embedding)
        
        selected_exemplars = []
        for i in indices[0]:
            # Assuming exemplar_data is a list of tuples or objects where we can retrieve info
            # For this example, let's assume exemplar_data is a list of (query, response) tuples
            # and we want to return the full (query, response) for the selected index.
            selected_exemplars.append(self.exemplar_data[i])
        
        print(f"Selected {len(selected_exemplars)} exemplars.")
        return selected_exemplars
