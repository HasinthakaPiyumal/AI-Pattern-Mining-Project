class LLMSimulator:
    def decompose_complaint(self, complaint: str) -> list:
        if "product refund" in complaint.lower():
            return ["Identify core issue: product refund request", "Retrieve customer details", "Check order history for product", "Check refund policy", "Propose refund solution", "Draft customer response for refund"]
        elif "delivery delay" in complaint.lower():
            return ["Identify core issue: delivery delay", "Retrieve customer details", "Check order tracking information", "Contact logistics for update", "Propose compensation/solution for delay", "Draft customer response for delay"]
        else:
            return ["Identify core issue", "Retrieve customer details", "Gather relevant information", "Propose general solution", "Draft general customer response"]

    def resolve_subtask(self, subtask: str, context: dict) -> str:
        if "Identify core issue" in subtask:
            return f"Core issue identified: {subtask.split(':', 1)[1].strip() if ':' in subtask else 'unknown'}"
        elif "Retrieve customer details" in subtask:
            customer_id = context.get("customer_id", "CUST12345")
            customer_info = ExternalSystemSimulator().query_crm(customer_id)
            return f"Customer details retrieved: {customer_info}"
        elif "Check order history" in subtask:
            customer_id = context.get("customer_id", "CUST12345")
            order_history = ExternalSystemSimulator().get_order_history(customer_id)
            return f"Order history checked: {order_history}"
        elif "Check refund policy" in subtask:
            policy = ExternalSystemSimulator().search_knowledge_base("refund policy")
            return f"Refund policy retrieved: {policy}"
        elif "Propose refund solution" in subtask:
            return "Proposed solution: Full refund issued, apologies for inconvenience."
        elif "Propose compensation/solution for delay" in subtask:
            return "Proposed solution: Expedited re-delivery and a 10% discount on next purchase."
        elif "Contact logistics for update" in subtask:
            return "Logistics updated: Package expected within 2 business days."
        elif "Draft customer response" in subtask:
            return f"Drafted response: Dear Customer, Your request has been processed. {context.get('final_solution', 'We are looking into your issue.')} We apologize for the inconvenience."
        elif "Gather relevant information" in subtask:
            return "Relevant information gathered: Specifics of the complaint logged."
        elif "Propose general solution" in subtask:
            return "Proposed solution: A generic solution addressing the complaint."
        else:
            return f"Subtask '{subtask}' resolved with generic information."


class ExternalSystemSimulator:
    def query_crm(self, customer_id: str) -> dict:
        return {"customer_id": customer_id, "name": "John Doe", "email": "john.doe@example.com"}

    def search_knowledge_base(self, query: str) -> str:
        if "refund policy" in query.lower():
            return "Our refund policy states that eligible products can be returned within 30 days for a full refund."
        return "No specific information found for your query."

    def get_order_history(self, customer_id: str) -> list:
        return [
            {"order_id": "ORD001", "product": "Laptop", "status": "Delivered", "date": "2023-01-15"},
            {"order_id": "ORD002", "product": "Mouse", "status": "Pending", "date": "2023-02-01"}
        ]


class ComplaintResolutionSystem:
    def __init__(self):
        self.llm_simulator = LLMSimulator()
        self.external_systems = ExternalSystemSimulator()

    def resolve_complaint(self, complaint: str, customer_id: str = "CUST12345") -> str:
        print(f"\n--- Resolving Complaint: {complaint} ---")
        subtasks = self.llm_simulator.decompose_complaint(complaint)
        print(f"Decomposed into subtasks: {subtasks}")

        resolution_context = {"original_complaint": complaint, "customer_id": customer_id, "resolved_steps": []}

        for i, subtask in enumerate(subtasks):
            print(f"\nExecuting Subtask {i+1}/{len(subtasks)}: {subtask}")
            resolved_output = self.llm_simulator.resolve_subtask(subtask, resolution_context)
            resolution_context["resolved_steps"].append({subtask: resolved_output})
            print(f"Subtask Result: {resolved_output}")
            if "Proposed solution" in resolved_output:
                resolution_context["final_solution"] = resolved_output

        final_resolution_parts = []
        for step in resolution_context["resolved_steps"]:
            for subtask_desc, result in step.items():
                final_resolution_parts.append(result)

        aggregated_resolution = " ".join(final_resolution_parts)
        final_response = f"Final Resolution for customer {customer_id}: {aggregated_resolution}"
        return final_response


def main():
    system = ComplaintResolutionSystem()

    complaint1 = "I want a refund for the faulty product I received last week."
    resolution1 = system.resolve_complaint(complaint1, customer_id="CUST001")
    print(f"\n--- Final Output ---:\n{resolution1}")

    complaint2 = "My order placed two days ago has not been delivered yet. Order ID: ORD002."
    resolution2 = system.resolve_complaint(complaint2, customer_id="CUST002")
    print(f"\n--- Final Output ---:\n{resolution2}")

    complaint3 = "I have a general inquiry about my account."
    resolution3 = system.resolve_complaint(complaint3, customer_id="CUST003")
    print(f"\n--- Final Output ---:\n{resolution3}")

if __name__ == "__main__":
    main()