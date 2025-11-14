import os
from collections import deque
from typing import List, Dict, Any, Tuple

# --- Dummy LLM and Embedding Models (replace with actual integrations) ---
# In a real scenario, you'd integrate with OpenAI, Google Gemini, Hugging Face models, etc.
# For this example, we'll simulate LLM responses and embeddings.

class DummyLLM:
    """A dummy LLM to simulate responses."""
    def invoke(self, prompt: str) -> str:
        if "product availability" in prompt.lower() and "laptop" in prompt.lower():
            return "The 'QuantumBook Pro' laptop is currently in stock. Estimated delivery is 3-5 business days."
        elif "order status" in prompt.lower() and "12345" in prompt.lower():
            return "Order #12345 has been shipped and is expected to arrive by next Tuesday."
        elif "return policy" in prompt.lower():
            return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please visit our returns page for more details."
        elif "complex technical issue" in prompt.lower():
            return "This seems like a complex technical issue. Based on your description, I recommend checking the device's firmware and connection settings. If the problem persists, please contact our technical support hotline at 1-800-TECH-HELP for further assistance."
        elif "greeting" in prompt.lower() or "hello" in prompt.lower():
             return "Hello! How can I assist you with your e-commerce needs today?"
        else:
            return f"I understand your query. Let me check... (Simulated LLM response for: '{prompt[:100]}...')"

class DummyEmbeddings:
    """A dummy embedding model to simulate vector creation."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Simple hash-based "embedding" for demonstration
        return [[float(hash(text) % 10000) / 10000.0] * 768 for text in texts] # Simulate a 768-dim vector
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Mock Langchain-like components for demonstration
class Document:
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __str__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

class Chroma:
    """A simplified, in-memory ChromaDB simulation for demonstration."""
    def __init__(self, embedding_function: Any):
        self._collection = []
        self._embeddings = embedding_function
        self._id_counter = 0

    def add_documents(self, documents: List[Document]):
        for doc in documents:
            embedding = self._embeddings.embed_query(doc.page_content)
            self._collection.append({
                "id": str(self._id_counter),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embedding
            })
            self._id_counter += 1

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_embedding = self._embeddings.embed_query(query)
        
        # Simple dot product similarity for demonstration
        scores = []
        for item in self._collection:
            # Assuming embedding vectors are 1D lists of floats
            # For real embeddings, you'd compute cosine similarity or L2 distance
            sim_score = sum(q * i for q, i in zip(query_embedding, item["embedding"])) # Dot product
            scores.append((sim_score, item))
        
        scores.sort(key=lambda x: x[0], reverse=True) # Higher score is more similar
        
        results = []
        for _, item in scores[:k]:
            results.append(Document(page_content=item["content"], metadata=item["metadata"]))
        return results

# --- Memory Systems ---

class ShortTermMemory:
    """Manages recent conversational context."""
    def __init__(self, max_interactions: int = 5):
        self.history = deque(maxlen=max_interactions * 2) # Store (user_query, agent_response) pairs

    def add_interaction(self, user_query: str, agent_response: str):
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "agent", "content": agent_response})

    def get_recent_context(self) -> str:
        context_parts = []
        for item in self.history:
            context_parts.append(f"{item['role'].capitalize()}: {item['content']}")
        return "\n".join(context_parts)

    def clear(self):
        self.history.clear()

class LongTermMemory:
    """Manages persistent knowledge using a vector store."""
    def __init__(self, vector_store: Any):
        self.vector_store = vector_store

    def add_knowledge(self, documents: List[Dict[str, str]]):
        """Adds new knowledge to the long-term memory."""
        lc_documents = [Document(page_content=d["content"], metadata=d.get("metadata", {})) for d in documents]
        self.vector_store.add_documents(lc_documents)
        print(f"Added {len(documents)} items to long-term memory.")

    def retrieve_knowledge(self, query: str, k: int = 3) -> List[str]:
        """Retrieves relevant knowledge based on a query."""
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in relevant_docs]

# --- Query Complexity Classifier ---

class QueryClassifier:
    """Classifies the complexity and type of a customer query."""
    def __init__(self, llm: Any):
        self.llm = llm # Can be a small, dedicated LLM or rule-based

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        if "status" in query_lower or "track my order" in query_lower or "where is my package" in query_lower:
            return "order_tracking"
        elif "return policy" in query_lower or "exchange item" in query_lower or "refund" in query_lower:
            return "returns_and_refunds"
        elif "product" in query_lower and ("availability" in query_lower or "stock" in query_lower or "features" in query_lower):
            return "product_info"
        elif "technical issue" in query_lower or "troubleshoot" in query_lower or "device not working" in query_lower:
            return "technical_support"
        elif "hello" in query_lower or "hi" in query_lower or "greeting" in query_lower:
            return "greeting"
        else:
            # For anything not directly matched, use LLM for a more nuanced classification
            # This is a simplified call; in reality, you'd use a more specific prompt or a fine-tuned model
            classification_prompt = (
                f"Classify the following customer query into one of these categories: "
                f"['order_tracking', 'returns_and_refunds', 'product_info', 'technical_support', 'general_inquiry', 'complaint']. "
                f"Query: '{query}'\nCategory:"
            )
            # A dummy LLM might not be able to follow this well, but it illustrates the idea.
            # A real LLM call here would be more robust.
            dummy_classification = self.llm.invoke(classification_prompt)
            if "order_tracking" in dummy_classification.lower(): return "order_tracking"
            if "returns_and_refunds" in dummy_classification.lower(): return "returns_and_refunds"
            if "product_info" in dummy_classification.lower(): return "product_info"
            if "technical_support" in dummy_classification.lower(): return "technical_support"
            if "complaint" in dummy_classification.lower(): return "complaint"
            return "general_inquiry" # Default if LLM fails or is generic


# --- Adaptive Customer Support Agent ---

class AdaptiveCustomerSupportAgent:
    """
    An Intelligent Adaptive Customer Support Agent for e-commerce.

    It leverages:
    - Short-term memory for recent conversation context.
    - Long-term memory (vector store) for factual knowledge.
    - A query classifier to adapt response strategies.
    - A main LLM for generating responses.
    """
    def __init__(self, llm: Any, embeddings: Any, max_short_term_interactions: int = 5):
        self.llm = llm
        self.embeddings = embeddings
        self.short_term_memory = ShortTermMemory(max_interactions=max_short_term_interactions)
        
        # Initialize Long-Term Memory with an in-memory Chroma instance
        self.long_term_vector_store = Chroma(embedding_function=self.embeddings)
        self.long_term_memory = LongTermMemory(vector_store=self.long_term_vector_store)

        self.query_classifier = QueryClassifier(llm=llm) # Can use the main LLM or a lighter one

        # Populate long-term memory with some example e-commerce data
        self._populate_initial_knowledge()

    def _populate_initial_knowledge(self):
        initial_knowledge = [
            {"content": "Our return policy allows returns within 30 days of purchase. Items must be in original packaging and condition. For electronics, a 15% restocking fee may apply if opened.", "metadata": {"source": "FAQ", "topic": "returns"}},
            {"content": "Standard shipping usually takes 3-5 business days. Express shipping delivers in 1-2 business days. International shipping times vary.", "metadata": {"source": "Shipping Info", "topic": "shipping"}},
            {"content": "The 'QuantumBook Pro' laptop features an M3 chip, 16GB RAM, 512GB SSD, and a 14-inch Retina display. It has a 12-hour battery life.", "metadata": {"source": "Product Catalog", "product_id": "QB-1001", "product_name": "QuantumBook Pro"}},
            {"content": "To check your order status, please navigate to 'My Orders' section on our website and enter your order number. You will see real-time updates.", "metadata": {"source": "FAQ", "topic": "order_tracking"}},
            {"content": "Technical support is available 24/7 via phone at 1-800-TECH-HELP or email at support@ecommerce.com. Please have your product model and serial number ready.", "metadata": {"source": "Support Page", "topic": "technical_support"}},
            {"content": "We offer a 1-year warranty on all electronic products, covering manufacturing defects. Extended warranty options are available at checkout.", "metadata": {"source": "Warranty Info", "topic": "warranty"}}
        ]
        self.long_term_memory.add_knowledge(initial_knowledge)
        print("Initial e-commerce knowledge loaded into long-term memory.")

    def process_query(self, user_query: str) -> str:
        # 1. Classify Query Complexity/Type
        query_type = self.query_classifier.classify(user_query)
        print(f"\nUser Query: '{user_query}'")
        print(f"Classified as: {query_type}")

        # 2. Retrieve Relevant Context
        short_term_context = self.short_term_memory.get_recent_context()
        long_term_context_docs = self.long_term_memory.retrieve_knowledge(user_query)
        long_term_context = "\n".join([f"Knowledge: {doc}" for doc in long_term_context_docs])

        # 3. Construct Adaptive Prompt
        system_message = (
            "You are an AI-powered customer support agent for an e-commerce platform. "
            "Your goal is to provide helpful, accurate, and concise responses. "
            "Leverage the provided short-term conversation history and long-term knowledge base. "
            "If you cannot find a definitive answer, politely state that you're looking into it or suggest contacting human support for complex issues." 
            f"The user's query is classified as '{query_type}'. Adapt your response strategy accordingly."
        )

        # Example of adaptive prompting based on query type
        if query_type == "order_tracking":
            system_message += "\nFocus on retrieving order details. Ask for an order number if not provided."
        elif query_type == "technical_support":
            system_message += "\nFor technical queries, provide troubleshooting steps if available, or direct to specialized technical support."
        elif query_type == "returns_and_refunds":
            system_message += "\nProvide information about return policies and procedures."
        elif query_type == "product_info":
            system_message += "\nProvide detailed product specifications or availability."


        full_prompt = (
            f"{system_message}\n\n"
            f"--- Conversation History (Short-Term Memory) ---\n{short_term_context}\n\n"
            f"--- E-commerce Knowledge Base (Long-Term Memory) ---\n{long_term_context}\n\n"
            f"--- User Query ---\nUser: {user_query}\n\n"
            f"Agent:"
        )

        # 4. Get LLM Response
        llm_response = self.llm.invoke(full_prompt)

        # 5. Update Short-Term Memory
        self.short_term_memory.add_interaction(user_query, llm_response)

        return llm_response

# --- Main Execution (Demonstration) ---
if __name__ == "__main__":
    # Initialize dummy LLM and embeddings
    dummy_llm = DummyLLM()
    dummy_embeddings = DummyEmbeddings()

    # Create the Adaptive Customer Support Agent
    agent = AdaptiveCustomerSupportAgent(llm=dummy_llm, embeddings=dummy_embeddings)

    print("\n--- Starting Customer Support Interaction ---")

    # Example interactions
    queries = [
        "Hello, I need some help.",
        "What is your return policy?",
        "Can you tell me about the QuantumBook Pro laptop?",
        "What's the status of my order 12345?",
        "My new smart speaker isn't connecting to Wi-Fi. It's a technical issue.",
        "How long does standard shipping take?",
        "I want to complain about a recent purchase. The item arrived damaged."
    ]

    for i, query in enumerate(queries):
        print(f"\n--- Interaction {i+1} ---")
        response = agent.process_query(query)
        print(f"Agent: {response}")
        print("-" * 50)

    print("\n--- Interaction with conversation history ---")
    response1 = agent.process_query("What is the warranty for electronics?")
    print(f"Agent: {response1}")
    response2 = agent.process_query("And does it cover accidental damage?") # Relying on STM for "it"
    print(f"Agent: {response2}")
