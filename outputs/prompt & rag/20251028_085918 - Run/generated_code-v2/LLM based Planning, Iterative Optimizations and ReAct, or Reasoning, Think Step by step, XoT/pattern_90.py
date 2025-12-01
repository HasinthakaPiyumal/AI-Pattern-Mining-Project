import re

class ECommerceChatbot:
    def __init__(self):
        self.ambiguous_keywords = ["problem", "issue", "something"] # General ambiguous terms
        self.specific_domain_keywords = {
            "order": ["order number", "order id", "track my order", "status of order"],
            "delivery": ["delivery status", "package delayed", "missing package", "tracking number"],
            "item": ["damaged item", "wrong item", "missing part", "return item", "exchange item"]
        }
        self.clarification_prompts = {
            "order": "Could you please provide your order number or a specific detail about the order?",
            "delivery": "Are you experiencing an issue with the delivery process, such as tracking, delays, or a missing package?",
            "item": "Can you specify which item in your order is causing the issue, and describe the problem (e.g., damaged, wrong item, missing part)?",
            "general_problem": "Can you please describe your problem in more detail so I can assist you better?"
        }
        self.clarified_responses = {
            "order_number_provided": "Thank you for providing your order details. Let me look up the information for you...",
            "delivery_issue": "I understand you have a delivery issue. Let me check the status or connect you with a delivery specialist.",
            "item_issue": "Regarding the item you mentioned, I'll help you with that. Please confirm the exact issue.",
            "default": "Thank you for the clarification. How can I assist you further with this?"
        }

    def detect_ambiguity(self, query: str) -> bool:
        """
        A simple keyword-based ambiguity detection.
        Detects if the query contains general ambiguous terms or
        if domain-specific terms are used without enough specific context.
        """
        query_lower = query.lower()

        # Check for general ambiguous keywords
        if any(kw in query_lower for kw in self.ambiguous_keywords):
            return True

        # Check for domain-specific terms that might be ambiguous without context
        if "order" in query_lower and not any(spec_kw in query_lower for spec_kw in self.specific_domain_keywords["order"]):
            # If "order" is mentioned but no specific order-related keywords like "order number"
            if not re.search(r'\d{4,}', query_lower): # Check for at least 4 consecutive digits (common for order numbers)
                return True
        
        if "delivery" in query_lower and not any(spec_kw in query_lower for spec_kw in self.specific_domain_keywords["delivery"]):
            return True

        if "item" in query_lower and not any(spec_kw in query_lower for spec_kw in self.specific_domain_keywords["item"]):
            return True

        return False

    def get_clarifying_question(self, query: str) -> str:
        """
        Generates a clarifying question based on the query's detected keywords.
        """
        query_lower = query.lower()
        if "order" in query_lower:
            return self.clarification_prompts["order"]
        elif "delivery" in query_lower:
            return self.clarification_prompts["delivery"]
        elif "item" in query_lower:
            return self.clarification_prompts["item"]
        else:
            return self.clarification_prompts["general_problem"]

    def process_query(self, user_query: str, clarification: str = None) -> str:
        """
        Processes a user query, potentially asking for clarification or providing a direct response.
        """
        output = [f"User: {user_query}"]

        if clarification:
            output.append(f"Chatbot (processing clarification): '{clarification}'")
            output.append(self._get_clarified_response(user_query, clarification))
        else:
            if self.detect_ambiguity(user_query):
                clarifying_q = self.get_clarifying_question(user_query)
                output.append(f"Chatbot (clarification needed): {clarifying_q}")
            else:
                output.append(f"Chatbot (direct response): How can I help you with your specific request: '{user_query}'?")
        
        return "\n".join(output)


    def _get_clarified_response(self, initial_query: str, clarification: str) -> str:
        """
        Internal method to generate a response based on initial query and clarification.
        """
        clarification_lower = clarification.lower()

        if ("order number" in clarification_lower or "order id" in clarification_lower or re.search(r'\d{4,}', clarification_lower)):
            return self.clarified_responses["order_number_provided"]
        elif any(kw in clarification_lower for kw in ["delivery", "tracking", "delayed", "missing", "package"]):
            return self.clarified_responses["delivery_issue"]
        elif any(kw in clarification_lower for kw in ["item", "product", "damaged", "wrong", "part"]):
            return self.clarified_responses["item_issue"]
        else:
            return self.clarified_responses["default"]
