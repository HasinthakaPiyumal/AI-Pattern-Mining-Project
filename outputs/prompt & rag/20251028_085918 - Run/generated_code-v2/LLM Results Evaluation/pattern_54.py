import random

class MockLLM:
    def __init__(self):
        self.responses = {
            "order_status_template_1": [
                "Your order #12345 is currently processing and expected to ship within 2 business days.",
                "Order #12345 has been placed and is awaiting fulfillment. You will receive a tracking number soon.",
                "We are checking the status of your order #12345. It should be updated shortly."
            ],
            "order_status_template_2": [
                "Good news! Order #12345 is on its way and should arrive by [Date]. You can track it here: [Tracking Link].",
                "Your package with order #12345 has been shipped! Check your email for tracking details.",
                "Just to confirm, your order #12345 is being prepared for shipment. We'll notify you when it leaves our warehouse."
            ],
            "return_policy_template_1": [
                "Our return policy allows returns within 30 days of purchase for a full refund.",
                "You can return items within 30 days. Please ensure the item is in its original condition.",
                "For returns, please visit our returns page for detailed instructions and to initiate the process."
            ],
            "return_policy_template_2": [
                "We offer hassle-free returns within 30 days. Simply go to your order history to start a return.",
                "To return an item, it must be unworn/unused and have original tags. You have 30 days from delivery.",
                "Our return policy states a 30-day window for returns from the purchase date for store credit or refund."
            ]
        }

    def generate_response(self, prompt: str) -> str:
        for key in self.responses:
            if key in prompt:
                return random.choice(self.responses[key])
        return "I\'m sorry, I don\'t have enough information to answer that based on the provided template."

class PromptTemplateManager:
    def __init__(self):
        self.templates = {
            "order_status": [
                "Here is your order status for customer query: \'{query}\'. Please use a polite and informative tone. Template: order_status_template_1",
                "Kindly provide the current status for the customer\'s query: \'{query}\'. Focus on clarity and estimated delivery. Template: order_status_template_2"
            ],
            "return_policy": [
                "Explain our return policy clearly for the customer\'s question: \'{query}\'. Ensure all key terms are covered. Template: return_policy_template_1",
                "Summarize our easy return process for the customer query: \'{query}\'. Emphasize customer satisfaction. Template: return_policy_template_2"
            ]
        }

    def get_templates(self, query_type: str) -> list[str]:
        return self.templates.get(query_type, [])

class QueryClassifier:
    def __init__(self):
        self.keywords = {
            "order_status": ["order", "status", "tracking", "delivery", "where is my", "package"],
            "return_policy": ["return", "refund", "exchange", "policy", "days to return", "how to return"]
        }

    def classify_query(self, customer_query: str) -> str | None:
        customer_query_lower = customer_query.lower()
        for q_type, kws in self.keywords.items():
            if any(kw in customer_query_lower for kw in kws):
                return q_type
        return None

class MutualInformationProxyEvaluator:
    def __init__(self, llm: MockLLM):
        self.llm = llm
        self.relevance_keywords = {
            "order_status": ["processing", "shipped", "delivered", "tracking number", "on its way", "arrived"],
            "return_policy": ["30 days", "refund", "exchange", "return process", "original condition", "store credit"]
        }

    def _calculate_relevance_score(self, query_type: str, llm_output: str) -> int:
        score = 0
        output_lower = llm_output.lower()
        if query_type in self.relevance_keywords:
            for keyword in self.relevance_keywords[query_type]:
                if keyword in output_lower:
                    score += 1
        return score

    def select_optimal_template(self, customer_query: str, query_type: str, templates: list[str]) -> str:
        best_template = None
        max_score = -1

        for template in templates:
            formatted_prompt = template.format(query=customer_query)
            llm_response = self.llm.generate_response(formatted_prompt)
            score = self._calculate_relevance_score(query_type, llm_response)
            
            if score > max_score:
                max_score = score
                best_template = template
        
        return best_template if best_template else templates[0] if templates else ""

def run_customer_support_optimizer():
    mock_llm = MockLLM()
    template_manager = PromptTemplateManager()
    query_classifier = QueryClassifier()
    evaluator = MutualInformationProxyEvaluator(mock_llm)

    customer_queries = [
        "Where is my order 12345?",
        "What is your return policy for clothes?",
        "I need to track my package.",
        "Can I get a refund for an item?",
        "How long do I have to return an item?"
    ]

    print("--- Automated Customer Support Response Template Optimization ---")

    for query in customer_queries:
        print(f"\nCustomer Query: \"{query}\"")
        
        query_type = query_classifier.classify_query(query)
        
        if query_type:
            print(f"  Classified as: {query_type}")
            
            available_templates = template_manager.get_templates(query_type)
            
            if available_templates:
                optimal_template = evaluator.select_optimal_template(query, query_type, available_templates)
                print(f"  Optimal Template Selected: \"{optimal_template}\"")
                
                final_prompt = optimal_template.format(query=query)
                final_response = mock_llm.generate_response(final_prompt)
                print(f"  LLM Generated Response: \"{final_response}\"")
            else:
                print(f"  No templates found for query type: {query_type}")
        else:
            print("  Could not classify the query.")

if __name__ == "__main__":
    run_customer_support_optimizer()