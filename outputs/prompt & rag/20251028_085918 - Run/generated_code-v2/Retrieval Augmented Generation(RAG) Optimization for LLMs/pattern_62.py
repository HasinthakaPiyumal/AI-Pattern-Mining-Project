from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DPRRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the DPRRetriever with a SentenceTransformer model.

        Args:
            model_name (str): The name of the pre-trained SentenceTransformer model.
        """
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        """
        Encodes a list of documents into dense vector representations.

        Args:
            documents (list[str]): A list of text documents.

        Returns:
            np.ndarray: A 2D numpy array where each row is the embedding of a document.
        """
        print(f"Encoding {len(documents)} documents...")
        document_embeddings = self.encoder.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        print("Document encoding complete.")
        return document_embeddings

    def build_index(self, documents: list[str]):
        """
        Builds a FAISS index from a list of documents.

        Args:
            documents (list[str]): A list of text documents to index.
        """
        self.documents = documents
        document_embeddings = self.encode_documents(documents)

        # Ensure embeddings are float32 for FAISS
        embedding_dim = document_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # IP for Inner Product similarity
        self.index.add(document_embeddings.astype('float32'))
        print(f"FAISS index built with {self.index.ntotal} documents.")

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Retrieves the top-k most relevant documents for a given query.

        Args:
            query (str): The customer query.
            k (int): The number of top documents to retrieve.

        Returns:
            list[tuple[str, float]]: A list of tuples, where each tuple contains
                                      (document_text, similarity_score).
        """
        if self.index is None:
            raise ValueError("FAISS index has not been built. Call build_index() first.")

        print(f"Encoding query: '{query}'")
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')

        print(f"Searching FAISS index for top {k} results...")
        distances, indices = self.index.search(query_embedding, k)

        retrieved_results = []
        for i in range(len(indices[0])):
            doc_index = indices[0][i]
            if doc_index < len(self.documents):
                retrieved_results.append((self.documents[doc_index], distances[0][i]))
        
        print("Retrieval complete.")
        return retrieved_results

# Example of how to use this class (will be put into app.py normally)
if __name__ == '__main__':
    # Dummy data for demonstration
    faq_articles = [
        "How do I return a product? You can return most items within 30 days of purchase.",
        "What is your shipping policy? We offer free standard shipping on all orders over $50.",
        "How can I track my order? You will receive a tracking number via email once your order ships.",
        "Do you offer international shipping? Currently, we only ship within the United States.",
        "How do I contact customer support? You can reach us via email at support@example.com or call us at 1-800-123-4567.",
        "What payment methods do you accept? We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
        "Can I change my shipping address after placing an order? Please contact support immediately to request a change.",
        "What is your refund process? Refunds are typically processed within 5-7 business days after we receive the returned item."
    ]

    # Initialize the retriever
    retriever = DPRRetriever()

    # Build the FAISS index
    retriever.build_index(faq_articles)

    # Simulate a customer query
    customer_query = "I want to send back something I bought."
    
    # Retrieve relevant articles
    results = retriever.retrieve(customer_query, k=3)

    print("\n--- Retrieved Results ---")
    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1} (Score: {score:.4f}): {doc}")

    customer_query_2 = "How can I talk to someone?"
    results_2 = retriever.retrieve(customer_query_2, k=2)
    print("\n--- Retrieved Results for 'How can I talk to someone?' ---")
    for i, (doc, score) in enumerate(results_2):
        print(f"Result {i+1} (Score: {score:.4f}): {doc}")