"""
This script implements an Adaptive Retrieval (Entity Frequency-based) customer support assistant.
It dynamically decides whether to use a retrieval module based on the frequency of entities in a query.
"""

# Mock Entity Frequency Database
ENTITY_FREQUENCIES = {
    "return policy": 1000,
    "track order": 950,
    "shipping": 800,
    "payment methods": 700,
    "contact support": 600,
    "reset password": 550,
    "product details": 400,
    "warranty": 350,
    "refund status": 300,
    "XYZ-123 laptop": 50,
    "order #98765": 10,
    "bluetooth headphones": 200,
    "sizing chart": 250,
    "assembly instructions": 150,
}

# Mock Knowledge Base
KNOWLEDGE_BASE = {
    "return policy": "Our return policy allows returns within 30 days of purchase with original packaging.",
    "track order": "You can track your order using the tracking number provided in your shipping confirmation email.",
    "shipping": "Standard shipping takes 3-5 business days. Expedited options are available at checkout.",
    "payment methods": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
    "XYZ-123 laptop specifications": "The XYZ-123 laptop features an Intel i7 processor, 16GB RAM, and a 512GB SSD.",
    "order #98765 status": "Order #98765 was shipped on 2023-10-26 and is expected to arrive by 2023-10-30.",
    "bluetooth headphones troubleshooting": "If your headphones are not connecting, ensure they are charged and in pairing mode. Refer to the manual for specific instructions.",
}

class QueryProcessor:
    def __init__(self, known_entities):
        self.known_entities = known_entities

    def extract_entities(self, query):
        found_entities = []
        query_lower = query.lower()
        for entity in self.known_entities:
            if entity in query_lower:
                found_entities.append(entity)
        return found_entities

class AdaptiveDecisionModule:
    def __init__(self, entity_frequencies, retrieval_threshold):
        self.entity_frequencies = entity_frequencies
        self.retrieval_threshold = retrieval_threshold

    def decide_retrieval(self, entities):
        if not entities:
            return False
        
        total_frequency = 0
        found_count = 0
        for entity in entities:
            # Use a default low frequency if entity is not explicitly in our mock DB
            freq = self.entity_frequencies.get(entity, 1)
            total_frequency += freq
            found_count += 1
        
        average_frequency = total_frequency / found_count
        return average_frequency < self.retrieval_threshold

class RetrievalModule:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def retrieve_documents(self, entities):
        retrieved_docs = []
        for entity in entities:
            # A simplified retrieval, in a real system this would be more sophisticated
            # We try to match entities to keys in our knowledge base
            doc = self.knowledge_base.get(f"{entity} specifications") or self.knowledge_base.get(f"{entity} status") or self.knowledge_base.get(entity)
            if doc:
                retrieved_docs.append(doc)
        return "\n".join(retrieved_docs) if retrieved_docs else None

class LLMInteractionModule:
    def mock_llm_response(self, query, context=None):
        if context:
            return f"Based on the information, here is what I found regarding your query: {context} If you need further assistance, please let me know."
        else:
            # Simulate LLM's parametric memory for common queries
            query_lower = query.lower()
            if "return policy" in query_lower:
                return "Our standard return policy allows for returns within 30 days of purchase."
            elif "track my order" in query_lower:
                return "Please provide your order number, and I can help you track it."
            elif "payment methods" in query_lower:
                return "We accept major credit cards and PayPal."
            else:
                return f"I understand you're asking about '{query}'. Please provide more details, or for more specific information, our system may need to perform a lookup."

class AdaptiveCustomerSupportAssistant:
    def __init__(self, retrieval_threshold=200):
        self.query_processor = QueryProcessor(list(ENTITY_FREQUENCIES.keys()))
        self.adaptive_decision_module = AdaptiveDecisionModule(ENTITY_FREQUENCIES, retrieval_threshold)
        self.retrieval_module = RetrievalModule(KNOWLEDGE_BASE)
        self.llm_interaction_module = LLMInteractionModule()

    def handle_query(self, query):
        entities = self.query_processor.extract_entities(query)
        should_retrieve = self.adaptive_decision_module.decide_retrieval(entities)

        context = None
        if should_retrieve:
            context = self.retrieval_module.retrieve_documents(entities)
            if context:
                print(f"DEBUG: Retrieval activated. Context: {context[:50]}...")
            else:
                print("DEBUG: Retrieval activated, but no relevant documents found.")
        else:
            print("DEBUG: Relying on LLM parametric memory (no retrieval).")

        response = self.llm_interaction_module.mock_llm_response(query, context)
        return response

if __name__ == "__main__":
    assistant = AdaptiveCustomerSupportAssistant(retrieval_threshold=300)

    print("\n--- Query 1: Common Query (No Retrieval Expected) ---")
    query1 = "What is your return policy?"
    response1 = assistant.handle_query(query1)
    print(f"User: {query1}")
    print(f"Assistant: {response1}")

    print("\n--- Query 2: Common Query (No Retrieval Expected) ---")
    query2 = "How can I track my order?"
    response2 = assistant.handle_query(query2)
    print(f"User: {query2}")
    print(f"Assistant: {response2}")

    print("\n--- Query 3: Specific Product Query (Retrieval Expected) ---")
    query3 = "Tell me about the XYZ-123 laptop specifications."
    response3 = assistant.handle_query(query3)
    print(f"User: {query3}")
    print(f"Assistant: {response3}")

    print("\n--- Query 4: Specific Order Query (Retrieval Expected) ---")
    query4 = "What is the status of my order #98765?"
    response4 = assistant.handle_query(query4)
    print(f"User: {query4}")
    print(f"Assistant: {response4}")
    
    print("\n--- Query 5: Moderately Specific Query (Threshold Dependent) ---")
    query5 = "I need help with my bluetooth headphones."
    response5 = assistant.handle_query(query5)
    print(f"User: {query5}")
    print(f"Assistant: {response5}")
    
    print("\n--- Query 6: Unrecognized Entity (No Retrieval Expected by default logic) ---")
    query6 = "What about the ABC-789 smart speaker?"
    response6 = assistant.handle_query(query6)
    print(f"User: {query6}")
    print(f"Assistant: {response6}")
