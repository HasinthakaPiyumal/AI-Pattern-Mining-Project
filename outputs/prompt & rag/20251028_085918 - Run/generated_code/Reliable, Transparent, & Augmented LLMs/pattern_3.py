from tools import ToolManager, get_order_status, search_knowledge_base, create_support_ticket
from explanation import ExplanationGenerator
import re

class AgenticCustomerSupportAI:
    """
    An Agentic and Trustworthy Customer Support AI system.
    Orchestrates LLM decisions, tool execution, and explanation generation.
    """
    def __init__(self, llm_model_name: str = "simulated_llm"):
        self.tool_manager = ToolManager()
        self.explanation_generator = ExplanationGenerator()
        self._initialize_tools()
        self.llm_model_name = llm_model_name
        print(f"AgenticCustomerSupportAI initialized with LLM: {self.llm_model_name}")

    def _initialize_tools(self):
        """Registers all necessary tools with the ToolManager."""
        self.tool_manager.register_tool(
            "get_order_status",
            get_order_status,
            "Retrieves the status of a customer order given an order ID. Args: order_id (str)"
        )
        self.tool_manager.register_tool(
            "search_knowledge_base",
            search_knowledge_base,
            "Searches the customer support knowledge base for relevant information. Args: query (str)"
        )
        self.tool_manager.register_tool(
            "create_support_ticket",
            create_support_ticket,
            "Creates a new support ticket for a customer issue. Args: issue_description (str), customer_id (str)"
        )
        print("External tools registered.")

    def _simulate_llm_decision(self, query: str) -> dict:
        """
        Simulates the LLM's decision-making process.
        In a real system, this would involve prompting an actual LLM.
        """
        query_lower = query.lower()
        if "order status" in query_lower and "order id" in query_lower:
            match = re.search(r"order id (\w+)", query_lower)
            order_id = match.group(1).upper() if match else "UNKNOWN"
            return {
                "action": "tool_call",
                "tool_name": "get_order_status",
                "tool_args": {"order_id": order_id},
                "reasoning": f"User is asking for order status and provided an order ID. Using 'get_order_status' tool."
            }
        elif "return policy" in query_lower or "shipping cost" in query_lower or "product information" in query_lower:
            return {
                "action": "tool_call",
                "tool_name": "search_knowledge_base",
                "tool_args": {"query": query},
                "reasoning": f"User is asking for general information. Using 'search_knowledge_base' tool."
            }
        elif "create a ticket" in query_lower or "escalate" in query_lower or "human help" in query_lower:
            return {
                "action": "tool_call",
                "tool_name": "create_support_ticket",
                "tool_args": {"issue_description": query, "customer_id": "sim_customer_123"},
                "reasoning": f"User explicitly asked to create a ticket or escalate. Using 'create_support_ticket' tool."
            }
        elif "hello" in query_lower or "hi" in query_lower:
            return {
                "action": "direct_response",
                "response": "Hello! How can I assist you today?",
                "reasoning": "A simple greeting, can be handled directly."
            }
        elif "how are you" in query_lower:
            return {
                "action": "direct_response",
                "response": "I am an AI assistant, designed to help you with your queries.",
                "reasoning": "A common conversational query, can be handled directly."
            }
        elif "i don't understand" in query_lower or "can you explain more" in query_lower:
            return {
                "action": "abstain",
                "reasoning": "The query indicates a lack of understanding or requires more nuanced explanation than current capabilities. Abstaining for human review."
            }
        else:
            return {
                "action": "abstain",
                "reasoning": "The query is complex or outside the scope of current direct capabilities, abstaining for human review to ensure accuracy."
            }

    def process_query(self, query: str) -> dict:
        """
        Processes a customer query through the AI agent.
        """
        print(f"Processing query: \"{query}\"")

        # 1. LLM Decision Making (Simulated)
        llm_decision = self._simulate_llm_decision(query)
        print(f"LLM Decision: {llm_decision}")

        action = llm_decision.get("action")
        tool_output = None
        final_response = None

        if action == "tool_call":
            tool_name = llm_decision.get("tool_name")
            tool_args = llm_decision.get("tool_args")
            tool_output = self.tool_manager.execute_tool(tool_name, **tool_args)
            print(f"Tool '{tool_name}' output: {tool_output}")
            final_response = tool_output 

        elif action == "direct_response":
            final_response = llm_decision.get("response")

        elif action == "abstain":
            final_response = "I apologize, but I need a human agent to assist you with this. I'm connecting you now."
        
        # 2. Explanation and Confidence Generation
        explanation, confidence = self.explanation_generator.generate_explanation(
            agent_decision=llm_decision,
            tool_output=tool_output,
            final_response=final_response
        )

        return {
            "query": query,
            "agent_response": final_response,
            "explanation": explanation,
            "confidence": confidence,
            "llm_decision_raw": llm_decision 
        }

if __name__ == "__main__":
    agent = AgenticCustomerSupportAI()

    print("\n--- Test Case 1: Order Status ---")
    result1 = agent.process_query("What is the status of my order with order id ORDER123?")
    print(f"Agent Response: {result1['agent_response']}")
    print(f"Explanation: {result1['explanation']}")
    print(f"Confidence: {result1['confidence']}")

    print("\n--- Test Case 2: Knowledge Base Query ---")
    result2 = agent.process_query("What is your return policy?")
    print(f"Agent Response: {result2['agent_response']}")
    print(f"Explanation: {result2['explanation']}")
    print(f"Confidence: {result2['confidence']}")

    print("\n--- Test Case 3: Create Support Ticket ---")
    result3 = agent.process_query("I need to talk to someone, please create a ticket about my faulty product.")
    print(f"Agent Response: {result3['agent_response']}")
    print(f"Explanation: {result3['explanation']}")
    print(f"Confidence: {result3['confidence']}")

    print("\n--- Test Case 4: Direct Response ---")
    result4 = agent.process_query("Hello there!")
    print(f"Agent Response: {result4['agent_response']}")
    print(f"Explanation: {result4['explanation']}")
    print(f"Confidence: {result4['confidence']}")

    print("\n--- Test Case 5: Abstention ---")
    result5 = agent.process_query("Can you explain the quantum mechanics principles behind your operations?")
    print(f"Agent Response: {result5['agent_response']}")
    print(f"Explanation: {result5['explanation']}")
    print(f"Confidence: {result5['confidence']}")

    print("\n--- Test Case 6: Unknown Order ---")
    result6 = agent.process_query("What's the status of order ID XYZ789?")
    print(f"Agent Response: {result6['agent_response']}")
    print(f"Explanation: {result6['explanation']}")
    print(f"Confidence: {result6['confidence']}")

    print("\n--- Test Case 7: Ambiguous Query (simulated abstention) ---")
    result7 = agent.process_query("I have a problem.")
    print(f"Agent Response: {result7['agent_response']}")
    print(f"Explanation: {result7['explanation']}")
    print(f"Confidence: {result7['confidence']}")
