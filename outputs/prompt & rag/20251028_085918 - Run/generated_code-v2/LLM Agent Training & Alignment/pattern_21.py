import json
import time
from datetime import datetime

class CustomerSupportAgentSimulator:
    """Simulates a human customer support agent's interaction with internal systems."""
    def __init__(self):
        self.knowledge_base = {
            "refund_policy": "Our refund policy allows full refunds within 30 days of purchase, provided the item is unused.",
            "shipping_times": "Standard shipping takes 5-7 business days. Expedited shipping takes 2-3 business days.",
            "account_recovery": "To recover your account, please visit our 'Forgot Password' page and follow the instructions."
        }
        self.order_system = {
            "ORD12345": {"status": "Shipped", "items": ["Laptop X"], "customer_id": "CUST001"},
            "ORD67890": {"status": "Processing", "items": ["Mouse Y", "Keyboard Z"], "customer_id": "CUST002"}
        }

    def search_knowledge_base(self, query):
        print(f"[AGENT ACTION] Searching knowledge base for: '{query}'")
        time.sleep(0.5) # Simulate delay
        for key, value in self.knowledge_base.items():
            if query in key or query in value:
                return {"observation": f"Found info on '{key}': {value}"}
        return {"observation": f"No direct information found for '{query}'."}

    def check_order_status(self, order_id):
        print(f"[AGENT ACTION] Checking order status for ID: '{order_id}'")
        time.sleep(0.7)
        order_info = self.order_system.get(order_id)
        if order_info:
            return {"observation": f"Order {order_id}: Status is '{order_info['status']}'. Items: {', '.join(order_info['items'])}."}
        return {"observation": f"Order ID '{order_id}' not found."}

    def compose_response(self, customer_query, information_gathered):
        print(f"[AGENT ACTION] Composing response for query: '{customer_query}' based on information: {information_gathered}")
        time.sleep(1.0)
        # A very basic example of composing a response
        if "refund" in customer_query.lower() and "refund_policy" in information_gathered:
            return {"action": f"The refund policy allows full refunds within 30 days of purchase, provided the item is unused. Is there anything else I can help with रिपेल?"}
        elif "shipping" in customer_query.lower() and "shipping_times" in information_gathered:
            return {"action": f"Standard shipping takes 5-7 business days. Expedited shipping takes 2-3 business days. Would you like to upgrade your shipping?"}
        elif "order" in customer_query.lower() and "ORD12345" in customer_query and "ORD12345" in information_gathered:
             return {"action": f"Your order ORD12345 is currently Shipped and includes Laptop X. Is there anything else I can help with?"}
        else:
            return {"action": f"I have gathered some information: {information_gathered}. How can I assist you further?"}

def record_demonstration(agent_simulator, customer_query, scenario_id):
    """Records a full demonstration of an agent resolving a customer query."""
    demonstration_log = []
    print(f"\n--- Starting Demonstration for Scenario ID: {scenario_id} ---")
    print(f"Customer Query: {customer_query}")

    # Initial observation (the query itself)
    demonstration_log.append({
        "timestamp": str(datetime.now()),
        "type": "customer_query",
        "content": customer_query
    })

    # Agent's actions and system's observations
    if "refund" in customer_query.lower():
        kb_result = agent_simulator.search_knowledge_base("refund_policy")
        demonstration_log.append({"timestamp": str(datetime.now()), "type": "observation", "content": kb_result["observation"]})
        composed_response = agent_simulator.compose_response(customer_query, kb_result["observation"])
        demonstration_log.append({"timestamp": str(datetime.now()), "type": "agent_response", "content": composed_response["action"]})

    elif "order status" in customer_query.lower() and "ORD12345" in customer_query:
        order_result = agent_simulator.check_order_status("ORD12345")
        demonstration_log.append({"timestamp": str(datetime.now()), "type": "observation", "content": order_result["observation"]})
        composed_response = agent_simulator.compose_response(customer_query, order_result["observation"])
        demonstration_log.append({"timestamp": str(datetime.now()), "type": "agent_response", "content": composed_response["action"]})

    else:
        # Default generic response for other queries
        kb_result = agent_simulator.search_knowledge_base("general assistance") # Simulate a general search
        demonstration_log.append({"timestamp": str(datetime.now()), "type": "observation", "content": kb_result["observation"]})
        composed_response = agent_simulator.compose_response(customer_query, kb_result["observation"])
        demonstration_log.append({"timestamp": str(datetime.now()), "type": "agent_response", "content": composed_response["action"]})

    print(f"--- End Demonstration for Scenario ID: {scenario_id} ---")
    return demonstration_log

if __name__ == "__main__":
    simulator = CustomerSupportAgentSimulator()

    demonstrations = []

    # Example 1: Refund query
    demo1 = record_demonstration(
        simulator,
        "I would like to know about your refund policy for a recent purchase.",
        "SCENARIO_001"
    )
    demonstrations.append(demo1)

    # Example 2: Order status query
    demo2 = record_demonstration(
        simulator,
        "What is the status of my order ORD12345?",
        "SCENARIO_002"
    )
    demonstrations.append(demo2)

    # Example 3: General query
    demo3 = record_demonstration(
        simulator,
        "I have a general question about account settings.",
        "SCENARIO_003"
    )
    demonstrations.append(demo3)

    # Save demonstrations to a JSON file
    output_filename = "demonstration_data.json"
    with open(output_filename, "w") as f:
        json.dump(demonstrations, f, indent=4)
    print(f"\nAll demonstrations saved to {output_filename}")

    # Print a sample demonstration for review
    if demonstrations:
        print("\n--- Sample Demonstration Log (SCENARIO_001) ---")
        print(json.dumps(demonstrations[0], indent=4))
