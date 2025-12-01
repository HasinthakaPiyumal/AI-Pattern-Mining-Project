import re
from typing import Dict, Any, Optional

class PolicyModule:
    """Determines the next action based on user input and current context."""
    def __init__(self):
        self.tool_patterns = {
            "get_account_balance": ["balance", "account statement", "funds"],
            "check_order_status": ["order status", "where is my order", "shipment"],
            "reset_password": ["reset password", "forgot password"],
            "get_product_recommendations": ["product recommendations", "suggest products", "what should I buy"]
        }

    def decide_action(self, user_query: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes the user query and context to decide if a tool should be used or LLM should respond.

        Returns a dictionary with 'action_type' and optional 'tool_name' and 'parameters'.
        """
        user_query_lower = user_query.lower()

        # First, check for explicit tool triggers
        for tool_name, patterns in self.tool_patterns.items():
            for pattern in patterns:
                if pattern in user_query_lower:
                    # Extract potential parameters for the tool
                    params = self._extract_tool_parameters(user_query, tool_name)
                    return {"action_type": "tool_use", "tool_name": tool_name, "parameters": params}

        # If no explicit tool, check context for follow-up actions
        if current_context.get("awaiting_tool_response"):
            # This is a placeholder; real logic would involve processing tool output
            return {"action_type": "llm_response", "reason": "Awaiting LLM to process tool output"}

        # Default: route to LLM for general response
        return {"action_type": "llm_response", "reason": "No specific tool trigger found"}

    def _extract_tool_parameters(self, user_query: str, tool_name: str) -> Dict[str, Any]:
        """Extracts parameters for specific tools from the user query."""
        params = {}
        user_query_lower = user_query.lower()

        if tool_name == "get_account_balance":
            if "checking" in user_query_lower:
                params["account_type"] = "checking"
            elif "savings" in user_query_lower:
                params["account_type"] = "savings"
            else:
                params["account_type"] = "default" # Or ask for clarification
        elif tool_name == "check_order_status":
            match = re.search(r"order #?(\d+)", user_query_lower)
            if match:
                params["order_id"] = match.group(1)
            else:
                params["order_id"] = None # Will require LLM to ask for it, or direct query
        # Add more parameter extraction logic for other tools as needed
        return params
