import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb

class KnowledgeBaseManager:
    def __init__(self, collection_name="customer_support_kb", model_name="all-MiniLM-L6-v2", db_path=".chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = SentenceTransformer(model_name)
        self.document_ids = []

    def add_documents(self, documents):
        embeddings = self.model.encode(documents).tolist()
        ids = [f"doc_{len(self.document_ids) + i}" for i in range(len(documents))]
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        self.document_ids.extend(ids)

    def retrieve_context(self, query, n_results=1):
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents']
        )
        return results['documents'][0] if results['documents'] else []

class SufficientContextAutorater:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.prompt_template = """
Given a Query (Q) and a Context (C), determine if the Context provides sufficient information for a plausible answer to the Query.

Definition of 'Sufficient Context': An instance (Q, C) has sufficient context if there exists a plausible answer A to Q given C. This definition does not require a pre-existing ground truth answer. A plausible answer is one that a knowledgeable person could reasonably derive solely from the provided context.

Examples:
Query: "What are the return policy for electronics?"
Context: "Our return policy states that most items can be returned within 30 days of purchase for a full refund. Electronics, however, must be returned within 15 days and are subject to a 10% restocking fee."
Classification: Sufficient

Query: "How do I reset my password?"
Context: "To change your password, navigate to the 'Account Settings' page and click on 'Change Password'. Follow the on-screen instructions."
Classification: Sufficient

Query: "What is the capital of France?"
Context: "Paris is a beautiful city known for its art museums and cafes. The Eiffel Tower is a popular landmark."
Classification: Insufficient (The context mentions Paris but doesn't explicitly state it's the capital of France relative to a query asking for the capital)

Query: "How can I track my order?"
Context: "For order tracking, please visit our website and enter your order number in the 'Track Order' section. You will receive an email confirmation with a tracking link once your order ships."
Classification: Sufficient

Query: "What are your shipping costs?"
Context: "We offer various shipping options including standard and express delivery."
Classification: Insufficient

Now, classify the following:
Query: {query}
Context: {context}
Classification: """

    def evaluate_context(self, query, context):
        formatted_context = " ".join(context) if isinstance(context, list) else context
        prompt = self.prompt_template.format(query=query, context=formatted_context)
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error during LLM evaluation: {e}"

if __name__ == "__main__":
    # Set your Google API key from environment variables
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running the script.")

    # --- 1. Initialize RAG System (Simulated) ---
    print("Initializing Knowledge Base...")
    kb_manager = KnowledgeBaseManager()

    # Sample Customer Support Knowledge Base Articles
    knowledge_base_articles = [
        "Our standard return policy allows for returns within 30 days of purchase with a valid receipt. Items must be in their original condition.",
        "Electronics can be returned within 15 days of purchase, but a 10% restocking fee applies. Opened software is not returnable.",
        "To reset your password, go to the login page and click 'Forgot Password'. Follow the instructions sent to your registered email address.",
        "Shipping costs vary based on your location and the selected shipping method (Standard, Express, Overnight). You can see the exact cost at checkout.",
        "Orders typically ship within 1-2 business days. You will receive a shipping confirmation email with a tracking number once your order has been dispatched.",
        "We offer international shipping to select countries. Please check our shipping information page for a list of eligible destinations and associated costs.",
        "Our loyalty program offers exclusive discounts and early access to sales for members. Sign up on our website!",
        "For technical support, please visit our help center or open a support ticket through your account dashboard."
    ]
    kb_manager.add_documents(knowledge_base_articles)
    print(f"Added {len(knowledge_base_articles)} documents to the knowledge base.")

    # --- 2. Initialize Sufficient Context Autorater ---
    print("Initializing Sufficient Context Autorater...")
    autorater = SufficientContextAutorater(api_key=GOOGLE_API_KEY)

    # --- 3. Simulate Customer Queries and Evaluate --- 
    customer_queries = [
        "What is your return policy?",
        "I need to return an opened software. Can I do that?",
        "How quickly do orders ship?",
        "How much will shipping cost for my order?",
        "Can I track my package?",
        "What is your phone number?", # Insufficient context from KB
        "Tell me about your loyalty program benefits.",
        "I want to buy a new laptop. Which models do you recommend?" # Insufficient context from KB
    ]

    print("\n--- Evaluating Queries ---")
    for i, query in enumerate(customer_queries):
        print(f"\nQuery {i+1}: {query}")
        retrieved_context = kb_manager.retrieve_context(query, n_results=1)
        print(f"Retrieved Context: {retrieved_context[0] if retrieved_context else 'No context found.'}")

        if retrieved_context:
            classification = autorater.evaluate_context(query, retrieved_context)
            print(f"Autorater Classification: {classification}")
        else:
            print("Autorater Classification: Insufficient (No context retrieved)")
