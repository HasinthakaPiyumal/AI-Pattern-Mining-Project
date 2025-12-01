class DialogState:
    def __init__(self, user_query):
        self.user_query = user_query
        self.history = [user_query]
        self.evidence = None
        self.candidate_response = None
        self.feedback = None

    def update_state(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_state(self):
        return {
            "user_query": self.user_query,
            "history": self.history,
            "evidence": self.evidence,
            "candidate_response": self.candidate_response,
            "feedback": self.feedback,
        }

class KnowledgeBase:
    def __init__(self):
        self.knowledge = {
            "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
            "returns": "You can return most items within 30 days of purchase. Please visit our returns page for more details.",
            "account": "To update your account information, log in and go to 'My Profile'.",
            "product details": "Please specify the product name or ID for more details.",
            "payment methods": "We accept Visa, MasterCard, American Express, PayPal, and Google Pay."
        }

    def retrieve_knowledge(self, query):
        query_lower = query.lower()
        for keyword, info in self.knowledge.items():
            if keyword in query_lower:
                return info
        return "I couldn't find specific information on that topic in our knowledge base. Would you like me to connect you with a specialist?"

class PromptEngine:
    def __init__(self):
        pass

    def query_llm(self, prompt):
        if "shipping" in prompt.lower():
            return "It seems you're asking about shipping. Standard shipping typically takes 3-5 business days."
        elif "return" in prompt.lower() or "refund" in prompt.lower():
            return "For returns or refunds, please ensure the item is within the 30-day return window. Do you have an order ID?"
        elif "order status" in prompt.lower():
            return "To check your order status, I'll need your order ID. Can you provide it?"
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I assist you with your e-commerce needs today?"
        elif "product" in prompt.lower():
            return "Could you please tell me which product you are interested in?"
        return "I'm a bit unsure how to respond to that. Can you rephrase or ask something else?"

class OrderManagementTool:
    def get_order_details(self, order_id):
        if order_id == "12345":
            return {"order_id": "12345", "status": "Shipped", "delivery_date": "2023-11-15"}
        elif order_id == "67890":
            return {"order_id": "67890", "status": "Processing", "delivery_date": "N/A"}
        return {"error": "Order not found"}

class ReturnProcessingTool:
    def process_return(self, order_id, item_id=None):
        if order_id == "12345" and item_id == "A1":
            return {"status": "Return initiated", "return_id": "RMA789"}
        elif order_id == "98765":
            return {"status": "Return processed", "refund_amount": "$50.00"}
        return {"error": "Could not process return"}

class Policy:
    def __init__(self):
        pass

    def decide_action(self, dialog_state):
        user_query_lower = dialog_state.user_query.lower()
        history_lower = [h.lower() for h in dialog_state.history]

        if "order status" in user_query_lower or "track my order" in user_query_lower:
            if any(item.isdigit() and len(item) == 5 for item in user_query_lower.split()):
                order_id = next((item for item in user_query_lower.split() if item.isdigit() and len(item) == 5), None)
                return "call_order_tool", {"action": "get_order_details", "order_id": order_id}
            return "query_llm", {"prompt": "Ask for order ID for status check."}

        if "return" in user_query_lower or "refund" in user_query_lower:
            if any(item.isdigit() and len(item) == 5 for item in user_query_lower.split()):
                order_id = next((item for item in user_query_lower.split() if item.isdigit() and len(item) == 5), None)
                return "call_return_tool", {"action": "process_return", "order_id": order_id}
            return "query_llm", {"prompt": "Ask for order ID for return process."}

        if "shipping" in user_query_lower or "delivery" in user_query_lower:
            return "retrieve_knowledge", {"query": "shipping"}

        if "account" in user_query_lower or "profile" in user_query_lower:
            return "retrieve_knowledge", {"query": "account"}

        if "product details" in user_query_lower or "information about" in user_query_lower:
            return "retrieve_knowledge", {"query": "product details"}

        if dialog_state.candidate_response and len(dialog_state.candidate_response) > 20 and "rephrase" not in dialog_state.candidate_response.lower():
            return "send_response", {}

        if len(dialog_state.history) > 3 and not dialog_state.evidence and not dialog_state.candidate_response:
            return "escalate_human", {}

        return "query_llm", {"prompt": dialog_state.user_query}

class CustomerSupportAgent:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.prompt_engine = PromptEngine()
        self.order_tool = OrderManagementTool()
        self.return_tool = ReturnProcessingTool()
        self.policy = Policy()
        self.dialog_state = None

    def handle_query(self, user_query):
        self.dialog_state = DialogState(user_query)
        final_response = ""
        max_turns = 5

        for _ in range(max_turns):
            action, args = self.policy.decide_action(self.dialog_state)

            if action == "retrieve_knowledge":
                evidence = self.knowledge_base.retrieve_knowledge(args["query"])
                self.dialog_state.update_state(evidence=evidence, history=self.dialog_state.history + [f"System: Retrieved knowledge: {evidence}"])
                self.dialog_state.update_state(candidate_response=evidence)
            elif action == "query_llm":
                prompt = args["prompt"]
                if self.dialog_state.evidence:
                    prompt = f"{self.dialog_state.evidence}. Based on this, {prompt}"
                if self.dialog_state.candidate_response:
                    prompt = f"{self.dialog_state.candidate_response}. {prompt}"

                llm_response = self.prompt_engine.query_llm(prompt)
                self.dialog_state.update_state(candidate_response=llm_response, history=self.dialog_state.history + [f"System: LLM says: {llm_response}"])
            elif action == "call_order_tool":
                order_id = args.get("order_id")
                if order_id:
                    order_details = self.order_tool.get_order_details(order_id)
                    self.dialog_state.update_state(evidence=order_details, history=self.dialog_state.history + [f"System: Order details: {order_details}"])
                    self.dialog_state.update_state(candidate_response=f"Here are the details for order {order_id}: Status: {order_details.get('status', 'N/A')}, Delivery Date: {order_details.get('delivery_date', 'N/A')}")
                else:
                    self.dialog_state.update_state(candidate_response="I need an order ID to get details. Can you provide it?")
            elif action == "call_return_tool":
                order_id = args.get("order_id")
                item_id = args.get("item_id")
                if order_id:
                    return_status = self.return_tool.process_return(order_id, item_id)
                    self.dialog_state.update_state(evidence=return_status, history=self.dialog_state.history + [f"System: Return status: {return_status}"])
                    self.dialog_state.update_state(candidate_response=f"Return status for order {order_id}: {return_status.get('status', 'N/A')}")
                else:
                    self.dialog_state.update_state(candidate_response="I need an order ID to process a return. Can you provide it?")
            elif action == "escalate_human":
                final_response = "I'm sorry, I couldn't fully resolve your query. I'm escalating this to a human agent for further assistance."
                break
            elif action == "send_response":
                final_response = self.dialog_state.candidate_response if self.dialog_state.candidate_response else "I'm ready to provide a response, but it seems there's no candidate response yet."
                break
            
            if final_response: # If a final response was set in an action, break the loop
                break
            if self.dialog_state.candidate_response and action != "query_llm":
                 # If a candidate response is ready and the policy didn't explicitly ask for an LLM query, try to send it.
                if self.policy.decide_action(self.dialog_state)[0] == "send_response":
                    final_response = self.dialog_state.candidate_response
                    break

        if not final_response and self.dialog_state.candidate_response:
            final_response = self.dialog_state.candidate_response
        elif not final_response:
            final_response = "I'm sorry, I couldn't process your request fully. Please try again or rephrase."

        return final_response

if __name__ == "__main__":
    agent = CustomerSupportAgent()

    queries = [
        "Hello, I need help.",
        "What is the status of my order 12345?",
        "I want to return item A1 from order 12345.",
        "How long does shipping take?",
        "How do I update my account?",
        "What payment methods do you accept?",
        "Can you tell me more about product XYZ?",
        "I have a very complex issue that needs a specialist.",
        "What about my order 67890?",
        "I want to return something from order 98765"
    ]

    for i, query in enumerate(queries):
        print(f"\n--- User Query {i+1}: {query} ---")
        response = agent.handle_query(query)
        print(f"Agent Response: {response}")