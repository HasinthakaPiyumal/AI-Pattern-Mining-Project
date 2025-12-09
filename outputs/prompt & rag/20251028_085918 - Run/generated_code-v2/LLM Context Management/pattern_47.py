import faiss
import numpy as np
import os
import time

class VectorIndexManager:
    def __init__(self, embedding_dimension):
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatL2(embedding_dimension) # L2 distance for similarity
        self.doc_ids = [] # To map index vectors back to original document IDs
        self.current_index_path = None

    def add_vectors(self, embeddings, doc_ids):
        """
        Adds vectors (embeddings) and their corresponding document IDs to the index.
        embeddings: A 2D numpy array of shape (num_vectors, embedding_dimension).
        doc_ids: A list of unique identifiers for each document.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.cpu().numpy() # Convert torch tensor to numpy if applicable

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dimension}, got {embeddings.shape[1]}")

        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)
        print(f"Added {len(doc_ids)} vectors to the index. Total vectors: {self.index.ntotal}")

    def search(self, query_embedding, k=5):
        """
        Performs a similarity search for the query embedding in the index.
        query_embedding: A 1D numpy array representing the query vector.
        k: The number of nearest neighbors to retrieve.
        Returns: A tuple of (distances, retrieved_doc_ids).
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.cpu().numpy() # Convert torch tensor to numpy if applicable

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1) # Reshape for FAISS search

        if query_embedding.shape[1] != self.embedding_dimension:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.embedding_dimension}, got {query_embedding.shape[1]}")

        distances, indices = self.index.search(query_embedding, k)
        
        retrieved_doc_ids = []
        for i in indices[0]:
            if i != -1: # FAISS returns -1 for empty slots if k is larger than ntotal
                retrieved_doc_ids.append(self.doc_ids[i])
            
        return distances[0].tolist(), retrieved_doc_ids

    def save_index(self, path):
        """
        Saves the FAISS index and its associated document IDs to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        np.save(f"{path}.doc_ids.npy", self.doc_ids)
        self.current_index_path = path
        print(f"Index saved to {path}.faiss and {path}.doc_ids.npy")

    @classmethod
    def load_index(cls, path):
        """
        Loads a FAISS index and its associated document IDs from disk.
        """
        index = faiss.read_index(f"{path}.faiss")
        doc_ids = np.load(f"{path}.doc_ids.npy", allow_pickle=True).tolist()
        
        manager = cls(index.d)
        manager.index = index
        manager.doc_ids = doc_ids
        manager.current_index_path = path
        print(f"Index loaded from {path}.faiss and {path}.doc_ids.npy")
        return manager

    def get_total_vectors(self):
        return self.index.ntotal

    def clear_index(self):
        """
        Clears the current index and associated doc_ids.
        """
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.doc_ids = []
        print("Index cleared.")

if __name__ == "__main__":
    # Example Usage
    embedding_dim = 384  # Example dimension, matches all-MiniLM-L6-v2
    manager = VectorIndexManager(embedding_dim)

    # Simulate some initial data
    initial_embeddings = np.random.rand(100, embedding_dim).astype('float32')
    initial_doc_ids = [f"doc_{i}" for i in range(100)]
    manager.add_vectors(initial_embeddings, initial_doc_ids)

    # Save the initial index
    initial_index_path = "./indices/initial_news_index"
    manager.save_index(initial_index_path)

    # Simulate a query
    query_vector = np.random.rand(embedding_dim).astype('float32')
    distances, retrieved_ids = manager.search(query_vector, k=3)
    print("\nSearch Results (Initial Index):")
    print("Distances:", distances)
    print("Retrieved IDs:", retrieved_ids)

    # --- Simulate Hotswapping --- 
    print("\n--- Simulating Hotswap ---")
    # Create a new index manager
    new_manager = VectorIndexManager(embedding_dim)
    new_embeddings = np.random.rand(120, embedding_dim).astype('float32') # More or updated data
    new_doc_ids = [f"doc_{i}" for i in range(100)] + [f"new_doc_{i}" for i in range(20)]
    new_manager.add_vectors(new_embeddings, new_doc_ids)

    # Save the new index
    new_index_path = "./indices/updated_news_index"
    new_manager.save_index(new_index_path)

    # In a real application, the main service would now switch to loading 'new_index_path'
    # For demonstration, let's load the new index into our original manager variable
    print("\nHotswapping: Loading new index...")
    manager = VectorIndexManager.load_index(new_index_path) # Simulating the swap

    print("Total vectors in hotswapped index:", manager.get_total_vectors())

    # Query the hotswapped index
    distances_new, retrieved_ids_new = manager.search(query_vector, k=3)
    print("\nSearch Results (Hotswapped Index):")
    print("Distances:", distances_new)
    print("Retrieved IDs:", retrieved_ids_new)

    # Clean up dummy index files
    time.sleep(1) # Give a moment for files to be written
    try:
        os.remove(f"{initial_index_path}.faiss")
        os.remove(f"{initial_index_path}.doc_ids.npy")
        os.remove(f"{new_index_path}.faiss")
        os.remove(f"{new_index_path}.doc_ids.npy")
        os.rmdir("./indices")
        print("\nCleaned up dummy index files and directory.")
    except OSError as e:
        print(f"Error cleaning up: {e}")