import random

PROMPT_TEMPLATE = """You are an expert context evaluator. Your task is to determine if the provided 'Context' contains sufficient information to plausibly answer the 'Query'.

Definition of 'Sufficient Context': An instance (Q, C) has sufficient context if there exists a plausible answer A to Q given C. This definition does not require a pre-existing ground truth answer. You only need to determine if an answer *could* be formulated from the context.

Instructions:
1.  Carefully read the 'Query' and the 'Context'.
2.  Determine if a plausible answer to the 'Query' can be derived *solely* from the 'Context'.
3.  Respond with either 'Sufficient' or 'Insufficient'. Do not provide any other text.

Query: {query}

Context: {context}

Decision:"""

class SimulatedRAGRetriever:
    def __init__(self):
        self.knowledge_base = {
            "product details": {
                "sufficient": "The 'NovaTech Laptop' features an Intel i7 processor, 16GB RAM, 512GB SSD, and a 15-inch full HD display. It comes with a 1-year warranty. The price is $1200.",
                "insufficient": "Our products are designed for high performance and durability. Many customers love our laptops."
            },
            "shipping policy": {
                "sufficient": "Standard shipping takes 5-7 business days for domestic orders. Express shipping is available for an additional $20 and delivers in 2-3 business days. International shipping varies by destination.",
                "insufficient": "We pride ourselves on fast delivery. Shipping costs are calculated at checkout."
            },
            "return process": {
                "sufficient": "To return an item, please visit our website's 'Returns' section, fill out the return form within 30 days of purchase, and ship the item back in its original packaging. Refunds are processed within 7 business days after we receive the item.",
                "insufficient": "We have a customer-friendly return policy. Your satisfaction is our priority."
            },
            "payment methods": {
                "sufficient": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay. We do not currently support cryptocurrency payments.",
                "insufficient": "We offer various secure payment options for your convenience."
            }
        }

    def retrieve_context(self, query):
        query_lower = query.lower()
        for keyword, contexts in self.knowledge_base.items():
            if keyword in query_lower:
                # Simulate a mix of sufficient and insufficient contexts for demonstration
                if random.random() < 0.7:  # 70% chance of sufficient context
                    return contexts["sufficient"]
                else:
                    return contexts["insufficient"]
        return "No relevant information found in our knowledge base. Please try rephrasing your query."

class SufficientContextAutorater:
    def evaluate(self, query, context):
        full_prompt = PROMPT_TEMPLATE.format(query=query, context=context)

        # Simulate LLM response based on keywords (simple heuristic for demonstration)
        if "features an Intel i7 processor" in context or \
           "Standard shipping takes 5-7 business days" in context or \
           "fill out the return form within 30 days" in context or \
           "We accept Visa, MasterCard, American Express" in context:
            return "Sufficient"
        elif "No relevant information found" in context or \
             "Our products are designed for high performance" in context or \
             "We pride ourselves on fast delivery" in context or \
             "We have a customer-friendly return policy" in context or \
             "We offer various secure payment options" in context:
            return "Insufficient"
        else:
            # Default to insufficient if specific keywords aren't hit (could be improved)
            return "Insufficient"


if __name__ == "__main__":
    retriever = SimulatedRAGRetriever()
    autorater = SufficientContextAutorater()

    customer_queries = [
        "What are the specifications of the NovaTech Laptop?",
        "How long does standard shipping take?",
        "What is your return policy?",
        "What payment methods do you accept?",
        "Can I pay with Bitcoin?",
        "Tell me about your latest discount offers."
    ]

    print("--- Automated Customer Support RAG Evaluator ---")
    print("\n")

    for i, query in enumerate(customer_queries):
        print(f"--- Query {i+1}: {query} ---")
        retrieved_context = retriever.retrieve_context(query)
        print(f"Retrieved Context: {retrieved_context[:100]}...")

        sufficiency_decision = autorater.evaluate(query, retrieved_context)
        print(f"Autorater Decision: {sufficiency_decision}")

        if sufficiency_decision == "Sufficient":
            print("Action: Proceed with LLM answer generation.")
            # In a real system, an LLM would generate an answer here using the context
        else:
            print("Action: Escalate to human agent or flag RAG for improvement.")
        print("\n")

    print("--- Evaluation Complete ---")