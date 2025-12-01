import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNExemplarSelector:
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.exemplar_embeddings = None
        self.exemplar_data = [] # Stores (text, label) for easy retrieval
        self.nn_model = None

    def fit(self, texts: list[str], embeddings: np.ndarray, labels: list[str]):
        """
        Fits the KNN model with exemplar embeddings and associated data.
        :param texts: List of original exemplar texts.
        :param embeddings: NumPy array of embeddings for the exemplars.
        :param labels: List of labels corresponding to the exemplars.
        """
        if len(texts) != len(embeddings) or len(texts) != len(labels):
            raise ValueError("Texts, embeddings, and labels must have the same length.")

        self.exemplar_embeddings = embeddings
        self.exemplar_data = [(text, label) for text, label in zip(texts, labels)]
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
        self.nn_model.fit(self.exemplar_embeddings)

    def select_exemplars(self, query_embedding: np.ndarray) -> list[dict]:
        """
        Selects k-nearest exemplars for a given query embedding.
        :param query_embedding: NumPy array of the query embedding.
        :return: A list of dictionaries, each containing 'text' and 'label' of an exemplar.
        """
        if self.nn_model is None:
            raise RuntimeError("KNNExemplarSelector has not been fitted yet. Call .fit() first.")

        # Reshape query_embedding for single sample prediction
        distances, indices = self.nn_model.kneighbors(query_embedding.reshape(1, -1))

        selected = []
        for i in indices[0]:
            text, label = self.exemplar_data[i]
            selected.append({"text": text, "label": label})
        return selected