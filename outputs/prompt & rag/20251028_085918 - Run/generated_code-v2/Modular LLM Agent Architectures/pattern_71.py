from typing import Dict, Any
from llm_interface import LLMInterface, MockLLM
from modules import ContextManager, PolicyModule, ToolExecutor, SentimentAnalyzer

class CustomerSupportAssistant:
    """
    An AI-powered Customer Support Assistant using a Plug-and-Play LLM Augmentation Framework.
    Orchestrates interaction between the LLM and various modules.
    """
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.context_manager = ContextManager()
        self.policy_module = PolicyModule()
        self.tool_executor = ToolExecutor()
        self.sentiment_analyzer = SentimentAnalyzer()

    def register_tool(self, name: str, func: Any):
        """Registers a tool with the internal ToolExecutor."""
        self.tool_executor.register_tool(name, func)

    def process_query(self, user_query: str) -> str:
        """
        Processes a user query by leveraging LLM and augmentation modules.
        """
        self.context_manager.add_message("user", user_query)
        print(f"\n--- User: {user_query} ---")

        # 1. Analyze Sentiment (Utility Module)
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_query)
        self.context_manager.update_session_context("current_sentiment", sentiment_result)
        print(f"  [Sentiment]: {sentiment_result["sentiment"]} (Score: {sentiment_result["score"]})")

        # 2. Policy Decision (Policy Module)
        # Prepare context for policy module
        full_context_for_policy = {
            "conversation_history": self.context_manager.get_history(),
            "user_profile": self.context_manager.get_profile(),
            "session_context": self.context_manager.get_session_context(),
            "sentiment": sentiment_result
        }
        policy_decision = self.policy_module.decide_action(user_query, full_context_for_policy)
        print(f"  [Policy Decision]: {policy_decision["action_type"]}")

        llm_input_context = {}
        llm_prompt_prefix = ""
        tool_output_for_llm = None

        # 3. Execute Tool if policy decides (Action Executor Module)
        if policy_decision["action_type"] == "tool_use":
            tool_name = policy_decision["tool_name"]
            tool_args = policy_decision.get("tool_args", {})
            
            try:
                # In a real scenario, tool_args might be refined by the LLM or another module
                # For this example, we pass what the policy suggests.
                tool_result = self.tool_executor.execute_tool(tool_name, tool_args)
                tool_output_for_llm = tool_result # Store for LLM context
                self.context_manager.update_session_context(f"{tool_name}_result", tool_result)
                llm_prompt_prefix = f"Based on the tool output for {tool_name}: {tool_result}, please respond to the user: "
            except ValueError as e:
                tool_output_for_llm = {"error": str(e)}
                self.context_manager.update_session_context(f"{tool_name}_error", tool_output_for_llm)
                llm_prompt_prefix = f"An error occurred while using the tool {tool_name}: {e}. Please inform the user: "
            
            print(f"  [Tool Result]: {tool_output_for_llm}")

        # 4. Generate LLM Response (LLM Interface)
        # Prepare comprehensive context for LLM
        combined_llm_context = {
            "conversation_history": self.context_manager.get_history(),
            "user_profile": self.context_manager.get_profile(),
            "session_context": self.context_manager.get_session_context(),
            "current_sentiment": sentiment_result,
        }
        if tool_output_for_llm:
            combined_llm_context[f"{tool_name}_output"] = tool_output_for_llm
        
        final_llm_prompt = f"{llm_prompt_prefix}{user_query}"
        llm_response = self.llm.generate_response(prompt=final_llm_prompt, context=combined_llm_context)
        
        self.context_manager.add_message("assistant", llm_response)
        print(f"--- Assistant: {llm_response} ---")
        print("-" * 50)
        return llm_response

# --- Mock Tool Implementations ---

def mock_order_tracker(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock tool to simulate checking an order status.
    In a real application, this would query a database or external API.
    """
    order_id = args.get("order_id")
    if not order_id:
        return {"status": "error", "message": "Order ID is required to track an order."}
    
    # Simulate different order statuses
    if order_id == "ORD123":
        return {"order_id": order_id, "status": "shipped", "delivery_date": "2023-11-25", "items": ["Laptop"]}
    elif order_id == "ORD456":
        return {"order_id": order_id, "status": "processing", "delivery_date": "2023-12-01", "items": ["Monitor"]}
    else:
        return {"order_id": order_id, "status": "not found", "message": "Order ID not found."}

def mock_knowledge_base_lookup(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock tool to simulate looking up information in a knowledge base.
    """
    query = args.get("query", "").lower()
    if "return policy" in query:
        return {"topic": "Return Policy", "content": "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be in original condition."}
    elif "shipping options" in query:
        return {"topic": "Shipping Options", "content": "We offer standard, expedited, and express shipping. Standard shipping is free for orders over $50."}
    elif "contact support" in query:
        return {"topic": "Contact Support", "content": "You can reach our support team via live chat on our website, email at support@example.com, or call us at 1-800-555-0123."}
    else:
        return {"topic": query, "content": "I couldn't find specific information on that topic in our knowledge base."}

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize the Mock LLM
    mock_llm = MockLLM()

    # Initialize the Customer Support Assistant
    assistant = CustomerSupportAssistant(llm=mock_llm)

    # Register the mock tools
    assistant.register_tool("OrderTracker", mock_order_tracker)
    assistant.register_tool("KnowledgeBaseLookup", mock_knowledge_base_lookup)

    print("Customer Support Assistant is ready. Type 'exit' to quit.")

    # Simulate a conversation
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting conversation. Goodbye!")
            break
        
        # Update user profile if relevant info is provided (demonstration)
        if "my order id is" in user_input.lower():
            try:
                order_id = user_input.split("my order id is")[1].strip().split(" ")[0].replace(".", "")
                assistant.context_manager.update_profile("order_id", order_id)
                print(f"[System]: Updated user profile with order ID: {order_id}")
            except Exception:
                pass # Simple error handling for demo

        assistant_response = assistant.process_query(user_input)
        # The assistant_response is already printed inside process_query
