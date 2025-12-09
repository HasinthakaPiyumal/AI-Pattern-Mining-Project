
class KnowledgeBase:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query_keywords, top_n=3):
        retrieved_docs = []
        for doc in self.documents:
            if any(keyword.lower() in doc["content"].lower() for keyword in query_keywords):
                retrieved_docs.append(doc)
        # Simple scoring based on keyword overlap (can be more sophisticated)
        scored_docs = []
        for doc in retrieved_docs:
            score = sum(1 for keyword in query_keywords if keyword.lower() in doc["content"].lower())
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_n]]

class SufficientContextAutorater:
    def __init__(self, min_docs_for_sufficiency=2, min_confidence_score=0.6):
        self.min_docs_for_sufficiency = min_docs_for_sufficiency
        self.min_confidence_score = min_confidence_score

    def evaluate(self, retrieved_context, query_keywords):
        num_docs = len(retrieved_context)
        
        if num_docs == 0:
            return {"sufficient": False, "confidence": 0.0, "reason": "No relevant documents found."}

        # Simulate a confidence score based on keyword density in retrieved docs
        total_keyword_matches = 0
        total_content_length = 0
        for doc in retrieved_context:
            total_content_length += len(doc["content"].split())
            for keyword in query_keywords:
                total_keyword_matches += doc["content"].lower().count(keyword.lower())
        
        # Simple confidence calculation: higher density of query keywords means higher confidence
        confidence = (total_keyword_matches / total_content_length) if total_content_length > 0 else 0.0
        
        sufficient = num_docs >= self.min_docs_for_sufficiency and confidence >= self.min_confidence_score
        
        reason = "Context is sufficient and confident." if sufficient else \
                 ("Insufficient number of documents." if num_docs < self.min_docs_for_sufficiency else \
                  "Low confidence in retrieved context.")

        return {"sufficient": sufficient, "confidence": round(confidence, 2), "reason": reason}

class CustomerSupportAgent:
    def __init__(self, knowledge_base_data):
        self.knowledge_base = KnowledgeBase(knowledge_base_data)
        self.autorater = SufficientContextAutorater()
        self.max_retrieval_attempts = 2

    def _extract_keywords(self, query):
        # A simple keyword extraction. In a real system, use NLP libraries.
        return [word for word in query.lower().replace("(", "").replace(")", "").replace("?", "").split() if len(word) > 2]

    def _refine_query(self, original_query, current_keywords, iteration):
        # Simulate query refinement. In a real system, an LLM would rephrase or add context.
        if iteration == 1:
            new_keywords = current_keywords + ["troubleshoot", "solution"]
            print(f"  -> Refining query: Adding general problem-solving keywords: {new_keywords}")
            return new_keywords
        else:
            # For simplicity, no further refinement after one attempt
            return current_keywords

    def _generate_answer(self, query, context):
        # This is a placeholder for an actual LLM call.
        # In a real system, you'd pass query and context to an LLM like OpenAI GPT, Cohere, etc.
        context_str = "\n".join([doc["content"] for doc in context])
        print("\n--- Generating Answer ---")
        print(f"Query: {query}")
        print(f"Context used:\n{context_str}")
        mock_answer = f"Based on your query '{query}' and the available information, here's a synthesized answer: [LLM would generate a detailed answer here using the provided context]. For example, if you're asking about '{query}', and the context mentions '{context[0]["content"][:50]}...', then the answer might be related to that. Please consult the full context for details."
        return mock_answer

    def handle_query(self, customer_query):
        print(f"\n--- Customer Query: {customer_query} ---")
        current_keywords = self._extract_keywords(customer_query)
        retrieved_context = []
        
        for attempt in range(self.max_retrieval_attempts):
            print(f"\nRetrieval Attempt {attempt + 1}/{self.max_retrieval_attempts}")
            print(f"  Keywords for retrieval: {current_keywords}")
            
            new_retrieved_docs = self.knowledge_base.retrieve(current_keywords)
            retrieved_context.extend(new_retrieved_docs)
            
            # Remove duplicates if any (based on 'id')
            unique_context = {doc['id']: doc for doc in retrieved_context}.values()
            retrieved_context = list(unique_context)

            evaluation = self.autorater.evaluate(retrieved_context, current_keywords)
            print(f"  Context Evaluation: Sufficient={evaluation['sufficient']}, Confidence={evaluation['confidence']}, Reason='{evaluation['reason']}'")
            
            if evaluation["sufficient"]:
                print("  Decision: Context sufficient. Generating answer.")
                return self._generate_answer(customer_query, retrieved_context)
            elif attempt < self.max_retrieval_attempts - 1:
                print("  Decision: Context insufficient but retrievable. Refining query for another attempt.")
                current_keywords = self._refine_query(customer_query, current_keywords, attempt + 1)
            else:
                print("  Decision: Context insufficient after multiple attempts or low confidence. Abstaining.")
                print("\n--- Escalating to Human Agent ---")
                escalation_message = f"Could not confidently answer the query '{customer_query}' after {self.max_retrieval_attempts} retrieval attempts. Current context: {[doc['content'] for doc in retrieved_context]}. Please review and assist."
                return escalation_message
        
        # Fallback if loop finishes without returning (should not happen with proper logic)
        return "An unexpected error occurred."


# --- Example Usage ---
if __name__ == "__main__":
    # Simulate an e-commerce knowledge base
    ecommerce_docs = [
        {"id": "prod_001", "content": "Our shipping policy states that standard delivery takes 5-7 business days. Express shipping takes 2-3 business days. International shipping varies by destination."},
        {"id": "prod_002", "content": "Returns are accepted within 30 days of purchase, provided the item is in its original condition with tags attached. Refunds are processed within 7-10 business days after receiving the returned item."},
        {"id": "prod_003", "content": "To track your order, please visit the 'My Orders' section on your account page and click on the 'Track' button next to your order number. You will receive a tracking number via email once your order ships."},
        {"id": "prod_004", "content": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay. Cash on Delivery is not available."},
        {"id": "prod_005", "content": "Our customer support can be reached via live chat from 9 AM to 5 PM EST, or by email at support@ecommerce.com. Phone support is available during business hours at 1-800-123-4567."},
        {"id": "prod_006", "content": "Product warranty covers manufacturing defects for 1 year from the date of purchase. It does not cover accidental damage or wear and tear."},
        {"id": "prod_007", "content": "Promotional code 'SAVE15' gives 15% off your first order. Minimum purchase of $50 required."},
        {"id": "prod_008", "content": "If you want to cancel your order, please do so within 24 hours of placement. After 24 hours, the order may have already been processed for shipping."},
        {"id": "prod_009", "content": "Troubleshooting common login issues: Ensure correct username/password, clear browser cache, or use the 'Forgot Password' link."},
        {"id": "prod_010", "content": "Our loyalty program rewards points for every purchase. Points can be redeemed for discounts on future orders. Check the 'Rewards' section for details."},
    ]

    agent = CustomerSupportAgent(ecommerce_docs)

    # Test Case 1: Sufficient context, direct answer
    print(agent.handle_query("What is your shipping policy?"))

    # Test Case 2: Insufficient initial context, refined query helps
    print(agent.handle_query("How do I track my package?"))

    # Test Case 3: Insufficient context, leads to escalation (not enough specific info)
    print(agent.handle_query("My charger is not working, what should I do?"))
    
    # Test Case 4: Another sufficient context query
    print(agent.handle_query("How can I return an item?"))
    
    # Test Case 5: Query requiring refinement
    print(agent.handle_query("I forgot my password, help me."))

    # Test Case 6: Very vague query, likely to escalate
    print(agent.handle_query("I have a problem."))
