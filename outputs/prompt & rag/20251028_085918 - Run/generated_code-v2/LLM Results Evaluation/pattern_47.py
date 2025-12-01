class WorkingMemory:
    def __init__(self):
        self.task_instructions = "You are a helpful e-commerce customer support assistant. Provide concise and accurate information based on the provided context."
        self.dialog_history = []
        self.current_query = ""
        self.evidence = ""
        self.feedback = ""

    def update_query(self, query):
        self.current_query = query

    def add_to_history(self, role, message):
        self.dialog_history.append(f"{role}: {message}")
        if len(self.dialog_history) > 5: # Keep last 5 turns for brevity
            self.dialog_history = self.dialog_history[-5:]

    def set_evidence(self, evidence):
        self.evidence = evidence

    def set_feedback(self, feedback):
        self.feedback = feedback

class KnowledgeConsolidator:
    def __init__(self):
        self.product_db = {
            "laptop": "Product A: High-performance laptop, 16GB RAM, 512GB SSD. Price: $1200.",
            "headphones": "Product B: Noise-cancelling headphones, 20-hour battery life. Price: $150.",
            "keyboard": "Product C: Mechanical keyboard, RGB lighting. Price: $80."
        }
        self.order_db = {
            "ORD123": "Order ORD123: Laptop (1 unit), shipped on 2023-10-26, estimated delivery 2023-10-30. Status: In Transit.",
            "ORD456": "Order ORD456: Headphones (2 units), processed on 2023-10-25, awaiting shipment. Status: Pending."
        }
        self.policy_db = {
            "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days and costs extra.",
            "returns": "Returns are accepted within 30 days of purchase for unused items with original packaging."
        }

    def get_evidence(self, query):
        evidence_list = []
        query_lower = query.lower()

        if "order" in query_lower or "status" in query_lower:
            order_id = next((word for word in query_lower.split() if word.startswith("ord") and len(word) == 6), None)
            if order_id and order_id.upper() in self.order_db:
                evidence_list.append(self.order_db[order_id.upper()])

        for product_keyword, info in self.product_db.items():
            if product_keyword in query_lower:
                evidence_list.append(info)

        if "shipping" in query_lower:
            evidence_list.append(self.policy_db["shipping"])
        if "return" in query_lower:
            evidence_list.append(self.policy_db["returns"])

        return " ".join(evidence_list) if evidence_list else "No specific external evidence found."

class UtilityModule:
    def generate_feedback(self, llm_response, original_query):
        feedback_score = 0
        feedback_message = "Neutral feedback."

        if "laptop" in original_query.lower() and "$1200" in llm_response:
            feedback_score += 1
        if "order" in original_query.lower() and "in transit" in llm_response.lower():
            feedback_score += 1
        if "return" in original_query.lower() and "30 days" in llm_response.lower():
            feedback_score += 1

        if "I can't help with that" in llm_response:
            feedback_score -= 1

        if feedback_score > 0:
            feedback_message = "Positive: LLM response was relevant and accurate."
        elif feedback_score < 0:
            feedback_message = "Negative: LLM response lacked relevance or accuracy."
        else:
            feedback_message = "Neutral: LLM response provided general information."

        return feedback_message

class PromptEngine:
    def construct_prompt(self, task_instructions, current_query, dialog_history, evidence, feedback):
        prompt_parts = [
            f"Instructions: {task_instructions}",
            "---"
        ]
        if dialog_history:
            prompt_parts.append("Dialog History:")
            prompt_parts.extend(dialog_history)
            prompt_parts.append("---")
        
        prompt_parts.append(f"User Query: {current_query}")
        prompt_parts.append("---")

        if evidence and evidence != "No specific external evidence found.":
            prompt_parts.append(f"External Evidence: {evidence}")
            prompt_parts.append("---")
        
        if feedback and "Positive" not in feedback:
            prompt_parts.append(f"Previous Feedback: {feedback} - Try to improve.")
            prompt_parts.append("---")

        prompt_parts.append("Assistant's Response:")

        return "\n".join(prompt_parts)

class LLMInteraction:
    def get_llm_response(self, prompt):
        # Mock LLM for demonstration
        if "laptop" in prompt and "$1200" in prompt:
            return "The high-performance laptop, Product A, with 16GB RAM and 512GB SSD, costs $1200."
        elif "order ORD123" in prompt:
            return "Your order ORD123 for a laptop was shipped on October 26th and is estimated to arrive by October 30th. It is currently in transit."
        elif "shipping" in prompt and "3-5 business days" in prompt:
            return "Standard shipping takes 3-5 business days. Express shipping is also available."
        elif "return" in prompt and "30 days" in prompt:
            return "You can return unused items within 30 days of purchase with original packaging."
        elif "headphones" in prompt and "$150" in prompt:
            return "Product B, the noise-cancelling headphones, cost $150 and have a 20-hour battery life."
        else:
            return "I am an e-commerce support assistant. How can I help you today? Please provide more details or ask about products, orders, or policies."

class CustomerSupportAssistant:
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.knowledge_consolidator = KnowledgeConsolidator()
        self.utility_module = UtilityModule()
        self.prompt_engine = PromptEngine()
        self.llm_interaction = LLMInteraction()

    def process_query(self, user_query):
        self.working_memory.update_query(user_query)
        self.working_memory.add_to_history("User", user_query)

        evidence = self.knowledge_consolidator.get_evidence(user_query)
        self.working_memory.set_evidence(evidence)

        full_prompt = self.prompt_engine.construct_prompt(
            self.working_memory.task_instructions,
            self.working_memory.current_query,
            self.working_memory.dialog_history,
            self.working_memory.evidence,
            self.working_memory.feedback
        )

        llm_response = self.llm_interaction.get_llm_response(full_prompt)

        feedback = self.utility_module.generate_feedback(llm_response, user_query)
        self.working_memory.set_feedback(feedback)
        self.working_memory.add_to_history("Assistant", llm_response)

        return llm_response

if __name__ == "__main__":
    assistant = CustomerSupportAssistant()

    print("\n--- Scenario 1: Product Inquiry ---")
    response1 = assistant.process_query("What can you tell me about the laptop?")
    print(f"User: What can you tell me about the laptop?")
    print(f"Assistant: {response1}")
    print(f"Feedback: {assistant.working_memory.feedback}")
    print("\n")

    print("\n--- Scenario 2: Order Status Inquiry ---")
    response2 = assistant.process_query("What is the status of my order ORD123?")
    print(f"User: What is the status of my order ORD123?")
    print(f"Assistant: {response2}")
    print(f"Feedback: {assistant.working_memory.feedback}")
    print("\n")

    print("\n--- Scenario 3: Shipping Policy ---")
    response3 = assistant.process_query("How long does shipping take?")
    print(f"User: How long does shipping take?")
    print(f"Assistant: {response3}")
    print(f"Feedback: {assistant.working_memory.feedback}")
    print("\n")

    print("\n--- Scenario 4: General Query ---")
    response4 = assistant.process_query("Tell me something interesting.")
    print(f"User: Tell me something interesting.")
    print(f"Assistant: {response4}")
    print(f"Feedback: {assistant.working_memory.feedback}")
    print("\n")

    print("\n--- Scenario 5: Another Product Inquiry with History ---")
    response5 = assistant.process_query("What about the headphones?")
    print(f"User: What about the headphones?")
    print(f"Assistant: {response5}")
    print(f"Feedback: {assistant.working_memory.feedback}")
    print("\n")