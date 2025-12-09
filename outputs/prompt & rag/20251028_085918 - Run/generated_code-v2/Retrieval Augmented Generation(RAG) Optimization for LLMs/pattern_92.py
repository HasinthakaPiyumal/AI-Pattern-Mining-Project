
import random

class MultiTaskLLM:
    """Simulates a multi-task LLM capable of both generating responses and ranking documents.
    In a real application, this would involve a fine-tuned transformer model.
    """
    def __init__(self, model_name="simulated-llm"):
        self.model_name = model_name
        # In a real scenario, the LLM would be loaded here
        # e.g., self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, prompt: str) -> str:
        """Generates a natural language response to a given prompt."""
        print(f"[LLM - Generating Response for: '{prompt[:50]}...']")
        # Simulate LLM's response generation
        if "product availability" in prompt.lower() or "in stock" in prompt.lower():
            return "Thank you for your query. Let me check the real-time stock for you. Could you please provide the product name or ID?"
        elif "order status" in prompt.lower() or "where is my order" in prompt.lower():
            return "I can help with that! Please provide your order number and I'll look up its current status."
        elif "return policy" in prompt.lower():
            return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. For more details, I can fetch the specific policy document."
        else:
            return "I'm designed to assist with a variety of inquiries. Please tell me more about what you need assistance with."

    def rank_documents(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        """Ranks a list of documents based on their relevance to the query.
        This simulates the LLM's understanding of relevance from its instruction-tuning.
        """
        print(f"[LLM - Ranking Documents for query: '{query[:50]}...']")
        # Simulate LLM's ranking capability. In a real scenario, this would involve
        # feeding query and document pairs/triplets to the LLM and getting a relevance score.
        ranked_docs = []
        for doc in documents:
            # Simple heuristic for simulation: higher score for keyword matches
            relevance_score = 0.1 # base score
            if query.lower() in doc.lower():
                relevance_score += 0.8
            if any(word in doc.lower() for word in query.lower().split()):
                relevance_score += 0.3
            
            # Introduce some randomness to make it look less deterministic
            relevance_score = min(1.0, relevance_score + random.uniform(-0.1, 0.2))
            ranked_docs.append((doc, relevance_score))
        
        # Sort by relevance score in descending order
        return sorted(ranked_docs, key=lambda x: x[1], reverse=True)


class SmartCustomerSupportAssistant:
    """E-commerce Smart Customer Support Assistant using a multi-task LLM.
    It handles customer queries by generating responses and ranking relevant knowledge.
    """
    def __init__(self, llm: MultiTaskLLM, knowledge_base_docs: list[str]):
        self.llm = llm
        self.knowledge_base = knowledge_base_docs
        print("Smart Customer Support Assistant initialized.")

    def handle_query(self, customer_query: str) -> dict:
        """Processes a customer query, generates a response, and ranks relevant documents."""
        print(f"\nCustomer Query: '{customer_query}'")
        
        # 1. Generate an initial natural language response using the LLM
        response = self.llm.generate_response(customer_query)
        print(f"Assistant Response: {response}")

        # 2. Rank relevant knowledge base documents using the same LLM
        ranked_articles = self.llm.rank_documents(customer_query, self.knowledge_base)
        
        # Filter out less relevant documents for display
        highly_relevant_articles = [doc for doc, score in ranked_articles if score > 0.5]
        
        output = {
            "generated_response": response,
            "ranked_documents": highly_relevant_articles
        }

        if highly_relevant_articles:
            print("\nMost Relevant Articles (from knowledge base):")
            for doc, score in ranked_articles[:3]: # Show top 3 for brevity
                print(f"- [Score: {score:.2f}] {doc}")
        else:
            print("\nNo highly relevant articles found in the knowledge base.")

        return output

# --- Demonstration --- 
if __name__ == "__main__":
    # Simulate a small knowledge base
    sample_knowledge_base = [
        "Article: Our shipping policy details delivery times, costs, and international options.",
        "FAQ: How to track your order. Use your order ID on the 'Track Order' page.",
        "Product Return Policy: Items can be returned within 30 days if unused and in original packaging.",
        "Troubleshooting: Common issues with product setup and quick fixes.",
        "Contact Support: How to reach our support team via chat, email, or phone.",
        "Product availability: Check product pages for real-time stock updates."
    ]

    # Initialize the simulated multi-task LLM
    llm_model = MultiTaskLLM()

    # Initialize the customer support assistant
    assistant = SmartCustomerSupportAssistant(llm=llm_model, knowledge_base_docs=sample_knowledge_base)

    # Test queries
    assistant.handle_query("Where is my order?")
    assistant.handle_query("What is your return policy for a shirt?")
    assistant.handle_query("Is the new smartphone in stock?")
    assistant.handle_query("I have a problem with my delivery.")
