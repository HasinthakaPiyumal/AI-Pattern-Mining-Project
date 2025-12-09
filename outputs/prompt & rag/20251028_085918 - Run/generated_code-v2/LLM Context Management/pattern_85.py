import numpy as np

class SimpleVectorMemory:
    """
    A simple in-memory vector store to simulate memory for the chatbot.
    In a real application, this would be a dedicated vector database (e.g., Chroma, Pinecone).
    """
    def __init__(self):
        self.memory_store = [] # Stores {'text': '...', 'embedding': np.array([...])}
        self.id_counter = 0

    def _generate_embedding(self, text):
        """
        Simulates embedding generation.
        In a real scenario, this would use a sentence-transformer model.
        """
        # For demonstration, a simple hash-based "embedding" or random vector
        # In reality, this would be `model.encode(text)`
        np.random.seed(hash(text) % (2**32 - 1)) # Consistent "embedding" for same text
        return np.random.rand(768) # Simulate a 768-dimension embedding

    def add_interaction(self, text):
        """Adds a new customer interaction to memory."""
        embedding = self._generate_embedding(text)
        self.memory_store.append({'id': self.id_counter, 'text': text, 'embedding': embedding})
        self.id_counter += 1
        print(f"DEBUG: Added to memory: '{text[:50]}...'")

    def retrieve_relevant_interactions(self, query_text, top_k=2):
        """
        Retrieves the most similar past interactions based on the query.
        Uses cosine similarity (dot product for normalized vectors).
        """
        if not self.memory_store:
            return []

        query_embedding = self._generate_embedding(query_text)
        similarities = []

        for item in self.memory_store:
            # Assuming embeddings are normalized for cosine similarity via dot product
            similarity = np.dot(query_embedding, item['embedding']) / (np.linalg.norm(query_embedding) * np.linalg.norm(item['embedding']))
            similarities.append((similarity, item['text']))

        similarities.sort(key=lambda x: x[0], reverse=True)
        relevant_texts = [text for sim, text in similarities[:top_k]]
        print(f"DEBUG: Retrieved relevant interactions for '{query_text[:50]}...': {relevant_texts}")
        return relevant_texts

class ECommerceChatbot:
    """
    An intelligent customer support chatbot augmented with external memory.
    """
    def __init__(self):
        self.memory = SimpleVectorMemory()
        self.context_window_limit = 500 # Simulated context window limit for the LLM

    def _call_llm(self, prompt):
        """
        Simulates an LLM call.
        In a real scenario, this would interact with an actual LLM API (e.g., OpenAI, Gemini).
        """
        print(f"\nDEBUG: LLM receiving prompt (first 200 chars): '{prompt[:200]}...'")
        # Simple rule-based response for demonstration
        if "delivery issue" in prompt.lower() and "order #12345" in prompt:
            return "Thank you for providing your order number. Your delivery for order #12345 is expected by end of day tomorrow. Would you like to track it?"
        elif "return policy" in prompt.lower():
            return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please visit our returns page for more details."
        elif "previous order" in prompt.lower() and "size medium" in prompt and "blue shirt" in prompt:
            return "Ah, I see you previously ordered a blue shirt in size medium. Are you looking for a similar item or perhaps assistance with a new purchase?"
        elif "remember my preference" in prompt.lower():
            return "Yes, I will keep your preference for blue shirts in mind for future recommendations!"
        elif "recommendation" in prompt.lower() and "hiking" in prompt and "boots" in prompt:
            return "Based on your interest in hiking, I can recommend our 'Trailblazer Pro' hiking boots, known for their durability and comfort."
        else:
            return "I'm a smart chatbot designed to help with e-commerce queries. How can I assist you further?"

    def process_query(self, customer_query):
        """
        Processes a customer query, leveraging memory for context.
        """
        # 1. Retrieve relevant past interactions from memory
        relevant_history = self.memory.retrieve_relevant_interactions(customer_query)

        # 2. Construct an augmented prompt for the LLM
        system_message = "You are an intelligent customer support agent for an e-commerce platform. Be helpful, polite, and use past interaction context to provide personalized answers."
        
        memory_context = ""
        if relevant_history:
            memory_context = "\n\nPast interactions (for context):\n"
            for i, item in enumerate(relevant_history):
                memory_context += f"- {item}\n"
            memory_context += "\n"

        # Ensure the prompt doesn't exceed a hypothetical context window limit
        # This is a simplification; real LLMs have token limits.
        
        # Start with a base prompt structure
        current_prompt = f"{system_message}\n{memory_context}Customer Query: {customer_query}"

        # In a real scenario, you'd prune `memory_context` or `customer_query` if too long
        # For this demo, we'll assume it fits or LLM handles truncation.

        # 3. Call the LLM with the augmented prompt
        llm_response = self._call_llm(current_prompt)

        # 4. Store the current interaction (query and response) into memory for future use
        self.memory.add_interaction(f"Customer: {customer_query}")
        self.memory.add_interaction(f"Chatbot: {llm_response}")

        return llm_response

# --- Demonstration ---
if __name__ == "__main__":
    chatbot = ECommerceChatbot()

    print("--- First Interaction (no memory yet) ---")
    response1 = chatbot.process_query("What is your return policy?")
    print(f"Chatbot: {response1}\n")

    print("--- Second Interaction (memory includes previous interaction) ---")
    response2 = chatbot.process_query("I have a delivery issue with order #12345. Can you help?")
    print(f"Chatbot: {response2}\n")

    print("--- Third Interaction (memory includes multiple interactions) ---")
    chatbot.memory.add_interaction("Customer: I bought a blue shirt last month, size medium. I really liked it.")
    chatbot.memory.add_interaction("Chatbot: Glad to hear you liked it! Are you looking for something similar?")
    response3 = chatbot.process_query("Can you recommend another shirt similar to my previous order? Maybe a different color but the same style and fit.")
    print(f"Chatbot: {response3}\n")
    
    print("--- Fourth Interaction (leveraging preference from deeper memory) ---")
    response4 = chatbot.process_query("I am going hiking next month. Do you have any good recommendations for hiking boots?")
    print(f"Chatbot: {response4}\n")

    print("--- Fifth Interaction (testing specific memory retrieval for preferences) ---")
    response5 = chatbot.process_query("Do you remember my preference for blue shirts?")
    print(f"Chatbot: {response5}\n")