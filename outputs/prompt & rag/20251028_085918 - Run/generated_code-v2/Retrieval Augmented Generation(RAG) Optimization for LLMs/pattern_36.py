import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

class SmartChatbot:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_path="faiss_index.bin", articles_path="articles.pkl"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.articles = []
        self.index_path = index_path
        self.articles_path = articles_path

    def build_knowledge_base(self, kb_articles):
        self.articles = kb_articles
        print("Encoding knowledge base articles...")
        article_embeddings = self.model.encode(kb_articles, show_progress_bar=True)
        article_embeddings = np.array(article_embeddings).astype('float32')

        dimension = article_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(article_embeddings)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.articles_path, 'wb') as f:
            pickle.dump(self.articles, f)
        print(f"Knowledge base built and saved to {self.index_path} and {self.articles_path}")

    def load_knowledge_base(self):
        if os.path.exists(self.index_path) and os.path.exists(self.articles_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.articles_path, 'rb') as f:
                self.articles = pickle.load(f)
            print("Knowledge base loaded successfully.")
            return True
        else:
            print("Knowledge base files not found. Please build the knowledge base first.")
            return False

    def retrieve_answer(self, query, k=1):
        if self.index is None or not self.articles:
            print("Knowledge base not loaded or built. Cannot retrieve answers.")
            return None

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        distances, indices = self.index.search(query_embedding, k)

        retrieved_info = []
        for i in range(k):
            article_index = indices[0][i]
            if article_index < len(self.articles):
                retrieved_info.append({
                    "article": self.articles[article_index],
                    "distance": distances[0][i]
                })
        return retrieved_info

    def generate_response(self, query):
        retrieved_articles = self.retrieve_answer(query, k=1)
        if retrieved_articles:
            most_relevant_article = retrieved_articles[0]['article']
            response = f"Based on our knowledge base, here's what I found: {most_relevant_article}"
        else:
            response = "I'm sorry, I couldn't find a relevant answer in my knowledge base. Please try rephrasing your question."
        return response

if __name__ == "__main__":
    # Sample Knowledge Base Articles
    sample_articles = [
        "Our return policy allows returns within 30 days of purchase with a valid receipt.",
        "To reset your password, please visit our website and click on 'Forgot Password' link.",
        "Shipping usually takes 3-5 business days for domestic orders.",
        "You can track your order using the tracking number provided in your shipping confirmation email.",
        "We offer 24/7 customer support via live chat and email.",
        "Our products are covered by a 1-year warranty against manufacturing defects.",
        "For technical support, please contact our dedicated technical support team at support@example.com.",
        "Payments can be made via credit card, PayPal, or bank transfer."
    ]

    chatbot = SmartChatbot()

    # Build the knowledge base (run once or when KB changes)
    chatbot.build_knowledge_base(sample_articles)

    # Or load an existing knowledge base
    # if not chatbot.load_knowledge_base():
    #     print("Exiting because knowledge base could not be loaded or built.")
    #     exit()

    # Simulate customer queries
    queries = [
        "How do I return an item?",
        "What is the shipping time?",
        "I need help with my account password.",
        "Where is my package?",
        "How can I contact support?",
        "What payment methods do you accept?",
        "My product is broken, what should I do?"
    ]

    print("\n--- Chatbot Interactions ---")
    for q in queries:
        print(f"Customer: {q}")
        response = chatbot.generate_response(q)
        print(f"Chatbot: {response}\n")

    # Example with a query that might not have a direct answer
    print("Customer: Do you offer international shipping?")
    response = chatbot.generate_response("Do you offer international shipping?")
    print(f"Chatbot: {response}\n")
