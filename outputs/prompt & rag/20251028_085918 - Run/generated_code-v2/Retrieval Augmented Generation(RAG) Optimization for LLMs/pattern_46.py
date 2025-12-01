class RetrievalPredictionModel:
    """
    A simplified model to predict if knowledge retrieval is needed.
    In a real system, this would be a trained classifier (e.g., a small BERT model,
    or a simple logistic regression on features extracted from the conversation).
    """
    def predict(self, conversation_context: str) -> bool:
        """
        Analyzes the conversation context to decide if retrieval is necessary.
        For this example, we'll use a simple keyword-based heuristic.
        """
        context_lower = conversation_context.lower()

        # Keywords indicating a need for specific information (trigger retrieval)
        complex_keywords = ["troubleshoot", "issue", "error code", "return policy",
                            "warranty", "specifications", "compare", "how to", "integrate",
                            "login problem", "shipping status", "contact support"]
        if any(keyword in context_lower for keyword in complex_keywords):
            return True

        # Phrases indicating a simple conversational turn (no retrieval needed)
        simple_phrases = ["hello", "hi", "hey", "good morning", "good afternoon",
                          "thanks", "thank you", "bye", "goodbye", "how are you"]
        if any(phrase in context_lower for phrase in simple_phrases):
            # We check for exact phrases for simplicity for simple greetings
            # A more robust system would check for intent.
            return False

        # If it's not explicitly simple or complex, assume retrieval might be beneficial
        # (conservative approach: better to retrieve than miss info in an ambiguous case)
        return True

class KnowledgeBase:
    """
    Simulates a knowledge base for retrieving relevant documents.
    In a real system, this would be a vector database (e.g., Chroma, Pinecone, FAISS)
    with embeddings of knowledge articles.
    """
    def __init__(self):
        self.documents = {
            "login_troubleshooting": "If you are experiencing login issues, please ensure your internet connection is stable and clear your browser's cache and cookies. If the problem persists, reset your password via the 'Forgot Password' link.",
            "return_policy": "Our return policy allows returns within 30 days of purchase for a full refund, provided the item is in its original condition. Some exclusions apply, please refer to our full terms and conditions.",
            "product_warranty": "All our electronics come with a 1-year manufacturer's warranty covering defects in material and workmanship. Accidental damage is not covered.",
            "shipping_tracking": "To track your order, please visit the 'My Orders' section on our website and click on the 'Track Shipment' link next to your order. You will need your order number.",
            "contact_support": "You can contact our support team via live chat on our website, by emailing support@example.com, or by calling 1-800-123-4567 during business hours.",
            "general_info": "We are an e-commerce platform specializing in consumer electronics and home goods. Our mission is to provide high-quality products and excellent customer service."
        }
        self.document_ids = list(self.documents.keys())

    def retrieve(self, query: str) -> list[str]:
        """
        Retrieves relevant documents based on the query.
        For this example, a simple keyword-based search is used across document content and keys.
        """
        retrieved_docs_content = []
        query_words = set(query.lower().split())

        for doc_id, doc_content in self.documents.items():
            # Check for keyword match in document ID (topic) or content
            if any(word in doc_id.lower() for word in query_words) or \
               any(word in doc_content.lower() for word in query_words):
                retrieved_docs_content.append(doc_content)

        return retrieved_docs_content if retrieved_docs_content else ["No specific knowledge found in the knowledge base for your query."]

class LLM:
    """
    Simulates a Large Language Model for generating responses.
    In a real system, this would be an API call to a model like GPT-3/4, Gemini,
    or a locally hosted model.
    """
    def generate_response(self, prompt: str, context: str = None) -> str:
        """
        Generates a response based on the prompt and optional context.
        """
        if context and context != "No specific knowledge found in the knowledge base for your query.":
            return f"**[LLM Response with Retrieved Knowledge]**\nBased on the information I found: \"{context}\"\n\nRegarding your query about \"{prompt}\", I can tell you that..."
        else:
            # Simple placeholder responses for common scenarios handled by the base LLM
            prompt_lower = prompt.lower()
            if "hello" in prompt_lower or "hi" in prompt_lower:
                return "**[LLM Base Response]** Hello! How can I assist you today?"
            elif "thanks" in prompt_lower or "thank you" in prompt_lower:
                return "**[LLM Base Response]** You're most welcome! Is there anything else I can help with?"
            elif "bye" in prompt_lower or "goodbye" in prompt_lower:
                return "**[LLM Base Response]** Goodbye! Have a great day!"
            else:
                return f"**[LLM Base Response - No Specific Retrieval]** I understand you're asking about '{prompt}'. I'll do my best to help. (A more advanced LLM would generate a detailed response here from its internal knowledge or general reasoning.)"

class CustomerSupportChatbot:
    """
    Orchestrates the chatbot interaction with conditional knowledge retrieval.
    """
    def __init__(self):
        self.rpm = RetrievalPredictionModel()
        self.kb = KnowledgeBase()
        self.llm = LLM()
        self.conversation_history = []

    def chat(self, user_query: str) -> str:
        self.conversation_history.append(f"User: {user_query}")
        # Consider the last few turns or the entire conversation for context
        current_conversation_context = " ".join(self.conversation_history[-5:]) # Last 5 turns for RPM

        needs_retrieval = self.rpm.predict(current_conversation_context)
        retrieved_info_string = ""

        if needs_retrieval:
            print(f"[DEBUG] RPM: Retrieval needed for query: '{user_query}'")
            retrieved_docs = self.kb.retrieve(user_query)
            if retrieved_docs:
                retrieved_info_string = "\n".join(retrieved_docs)
            print(f"[DEBUG] KB Retrieved: {retrieved_info_string[:150]}...") # Show beginning of retrieved info
        else:
            print(f"[DEBUG] RPM: No retrieval needed for query: '{user_query}'")

        if retrieved_info_string:
            response = self.llm.generate_response(user_query, retrieved_info_string)
        else:
            response = self.llm.generate_response(user_query)

        self.conversation_history.append(f"Bot: {response}")
        return response

# Example Usage:
if __name__ == "__main__":
    chatbot = CustomerSupportChatbot()

    print("Welcome to the Dynamic Customer Support Chatbot! Type 'exit' to end the conversation.")
    print("Try asking about 'hello', 'return policy', 'troubleshoot login', 'how are you', 'shipping status'.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Thanks for chatting.")
            break

        bot_response = chatbot.chat(user_input)
        print(f"Bot: {bot_response}")
