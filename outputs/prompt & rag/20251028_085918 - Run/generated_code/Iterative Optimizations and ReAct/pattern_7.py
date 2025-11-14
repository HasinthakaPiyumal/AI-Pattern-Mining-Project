import json
import time
from typing import Dict, Any, List

# A simple decorator to mark functions as tools for the agent.
def tool(func):
    """A simple decorator to mark functions as tools."""
    func.is_tool = True
    return func


class MockLLM:
    """A mock LLM for demonstration purposes, simulating tool calling and conversational responses.
    
    In a real application, this would be an actual LLM (e.g., from OpenAI, Google, etc.)
    integrated via a framework like LangChain or LlamaIndex.
    """
    def __init__(self, responses: Dict[str, Any] = None):
        self.responses = responses if responses is not None else {}

    def invoke(self, prompt: str) -> str:
        """Simulates the LLM's reasoning and action decision based on the prompt."""
        # Simulate tool calling based on recognized patterns in the prompt
        if "tool_name": # A simplified way to check for tool call intent
            if "search_orders" in prompt.lower() and "order id" in prompt.lower():
                order_id = self._extract_param(prompt, "order_id")
                if order_id: return json.dumps({"action": "tool_call", "tool_name": "search_orders", "args": {"order_id": order_id}})
            elif "get_product_details" in prompt.lower() and "product id" in prompt.lower():
                product_id = self._extract_param(prompt, "product_id")
                if product_id: return json.dumps({"action": "tool_call", "tool_name": "get_product_details", "args": {"product_id": product_id}})
            elif "track_shipping" in prompt.lower() and "tracking number" in prompt.lower():
                tracking_number = self._extract_param(prompt, "tracking_number")
                if tracking_number: return json.dumps({"action": "tool_call", "tool_name": "track_shipping", "args": {"tracking_number": tracking_number}})
            elif "escalate_issue" in prompt.lower() and "issue_summary" in prompt.lower():
                issue_summary = self._extract_param(prompt, "issue_summary")
                if issue_summary: return json.dumps({"action": "tool_call", "tool_name": "escalate_issue", "args": {"issue_summary": issue_summary}})
            elif "resolve_issue" in prompt.lower() and "resolution_notes" in prompt.lower():
                issue_summary = self._extract_param(prompt, "issue_summary")
                resolution_notes = self._extract_param(prompt, "resolution_notes")
                if issue_summary and resolution_notes: return json.dumps({"action": "tool_call", "tool_name": "resolve_issue", "args": {"issue_summary": issue_summary, "resolution_notes": resolution_notes}})
            elif "update_crm_ticket" in prompt.lower() and "ticket_id" in prompt.lower():
                ticket_id = self._extract_param(prompt, "ticket_id")
                status = self._extract_param(prompt, "status")
                notes = self._extract_param(prompt, "notes")
                if ticket_id and status and notes: return json.dumps({"action": "tool_call", "tool_name": "update_crm_ticket", "args": {"ticket_id": ticket_id, "status": status, "notes": notes}})

        # Default conversational response based on keywords
        for keyword, response in self.responses.items():
            if keyword in prompt.lower():
                return json.dumps({"action": "respond", "response": response})
        
        # Fallback if no specific tool call or keyword match
        return json.dumps({"action": "respond", "response": "I\'m thinking step-by-step about how to assist you."})

    def _extract_param(self, text: str, param_name: str) -> Any:
        """Very simple regex-like extraction for demonstration."""
        import re
        # This regex tries to find param_name: 