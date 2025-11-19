import json
from typing import Dict, Any, Callable, List

class AdaptiveAgent:
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.world_model: Dict[str, Any] = {}
        self.tools: Dict[str, Dict[str, Any]] = self._register_tools()
        self.MAX_ITERATIONS = 5

    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        """Registers available tools for the agent."""
        return {
            "get_order_status": {
                "func": self._get_order_status,
                "description": "Retrieves the current status of an order given an order_id."
                               "Parameters: order_id (str)"
            },
            "lookup_faq": {
                "func": self._lookup_faq,
                "description": "Searches the FAQ for answers related to a specific topic or keyword."
                               "Parameters: topic (str)"
            },
            "escalate_to_human": {
                "func": self._escalate_to_human,
                "description": "Escalates the current customer query to a human agent."
                               "Parameters: reason (str)"
            }
        }

    def _get_order_status(self, order_id: str) -> str:
        """Simulates fetching order status."""
        if order_id == "ORDER123":
            return json.dumps({"status": "Shipped", "estimated_delivery": "2023-10-27"})
        return json.dumps({"status": "Not Found", "message": f"Order {order_id} not found."})

    def _lookup_faq(self, topic: str) -> str:
        """Simulates looking up an FAQ entry."""
        if "shipping" in topic.lower():
            return json.dumps({"answer": "Shipping usually takes 3-5 business days. You can track your order using the link in your confirmation email."})
        elif "return" in topic.lower():
            return json.dumps({"answer": "Items can be returned within 30 days of purchase with the original receipt."})
        return json.dumps({"answer": f"No direct FAQ found for '{topic}'."})

    def _escalate_to_human(self, reason: str) -> str:
        """Simulates escalating to a human agent."""
        return json.dumps({"action": "escalated", "message": f"Query escalated to human agent. Reason: {reason}"})

    def _call_llm(self, prompt: str) -> str:
        """Simulates an LLM call. In a real scenario, this would use an actual LLM API.
        For demonstration, it processes the prompt to simulate tool selection, response generation, or reflection.
        """
        # Simulate tool selection
        if "TOOL_SELECTION_REQUEST:" in prompt:
            if "order status" in prompt.lower() and "order_id" in prompt.lower():
                # Extract order_id from prompt for simulation
                import re
                match = re.search(r"order_id=(.+?)(,|$|\n)", prompt)
                order_id = match.group(1).strip() if match else ""
                return json.dumps({"tool_name": "get_order_status", "parameters": {"order_id": order_id if order_id else "UNKNOWN_ORDER"}})
            elif "shipping question" in prompt.lower() or "return policy" in prompt.lower():
                topic = "shipping" if "shipping" in prompt.lower() else "return"
                return json.dumps({"tool_name": "lookup_faq", "parameters": {"topic": topic}})
            elif "cannot resolve" in prompt.lower() or "complex issue" in prompt.lower():
                return json.dumps({"tool_name": "escalate_to_human", "parameters": {"reason": "User query is complex and requires human intervention."}})
            return json.dumps({"tool_name": "NONE", "parameters": {}})

        # Simulate self-reflection
        if "SELF_REFLECTION_REQUEST:" in prompt:
            if "unresolved" in prompt.lower() and "try another tool" in prompt.lower():
                return json.dumps({"reflection": "The previous approach was unsuccessful. I should try to gather more information or escalate.", "action": "REPLAN"})
            elif "resolved" in prompt.lower():
                return json.dumps({"reflection": "The query seems to be resolved successfully.", "action": "TERMINATE"})
            return json.dumps({"reflection": "Current state seems fine. Proceeding.", "action": "CONTINUE"})
        
        # Simulate termination evaluation
        if "TERMINATION_EVALUATION_REQUEST:" in prompt:
            if "resolved successfully" in prompt.lower() or "escalated to human" in prompt.lower():
                return json.dumps({"terminate": True, "reason": "Query resolved or escalated."})
            return json.dumps({"terminate": False, "reason": "Still working on it."})

        # Simulate response generation
        if "RESPONSE_GENERATION_REQUEST:" in prompt:
            if "ORDER123" in self.world_model.get("last_order_id", "") and "Shipped" in self.world_model.get("order_status", ""):
                return "Your order ORDER123 has been shipped and is estimated to arrive on 2023-10-27."
            elif "shipping" in self.world_model.get("last_faq_topic", "") and "3-5 business days" in self.world_model.get("faq_answer", ""):
                return "Shipping usually takes 3-5 business days. You can track your order using the link in your confirmation email."
            elif "escalated" in self.world_model.get("last_action", ""):
                return "I have escalated your query to a human agent. They will contact you shortly."
            return f"I'm still processing your request based on: {prompt.split('Current Context:')[-1].strip()}"

        return "I am an AI assistant and currently cannot process this request outside of a defined flow."

    def _update_world_model(self, key: str, value: Any):
        self.world_model[key] = value

    def _get_llm_tool_prompt(self) -> str:
        tool_descriptions = "\n".join([f"- {name}: {info['description']}" for name, info in self.tools.items()])
        return (
            f"TOOL_SELECTION_REQUEST: Given the conversation history and current world model, "
            f"which tool should be called to resolve the user's request, and with what parameters? "
            f"If no tool is suitable, respond with {{'tool_name': 'NONE', 'parameters': {{}}}}. "
            f"Provide your response as a JSON string."
            f"\nConversation History: {json.dumps(self.conversation_history)}"
            f"\nWorld Model: {json.dumps(self.world_model)}"
            f"\nAvailable Tools:\n{tool_descriptions}"
        )

    def _get_llm_response_prompt(self, tool_output: str = "") -> str:
        return (
            f"RESPONSE_GENERATION_REQUEST: Generate a concise and helpful response to the user "
            f"based on the conversation history, current world model, and any tool output. "
            f"\nConversation History: {json.dumps(self.conversation_history)}"
            f"\nWorld Model: {json.dumps(self.world_model)}"
            f"\nTool Output: {tool_output}"
            f"\nUser Feedback: {self.conversation_history[-1]['feedback'] if self.conversation_history and 'feedback' in self.conversation_history[-1] else 'None'}"
        )

    def _get_llm_reflection_prompt(self, last_action_success: bool, user_feedback: str) -> str:
        return (
            f"SELF_REFLECTION_REQUEST: Reflect on the last action's success ({last_action_success}) and user feedback ('{user_feedback}'). "
            f"Evaluate the current state and world model. Propose an action: CONTINUE, REPLAN, or TERMINATE. "
            f"If REPLAN, suggest how to improve the strategy or what new information is needed. "
            f"Provide your response as a JSON string with 'reflection' and 'action' keys."
            f"\nConversation History: {json.dumps(self.conversation_history)}"
            f"\nWorld Model: {json.dumps(self.world_model)}"
        )

    def _get_llm_termination_prompt(self) -> str:
        return (
            f"TERMINATION_EVALUATION_REQUEST: Based on the current conversation history and world model, "
            f"is the user's query fully resolved or has it been escalated? Respond with "
            f"{{'terminate': True/False, 'reason': '...'}} as a JSON string."
            f"\nConversation History: {json.dumps(self.conversation_history)}"
            f"\nWorld Model: {json.dumps(self.world_model)}"
        )

    def interact(self, user_query: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_query})
        self.world_model["current_query"] = user_query

        for i in range(self.MAX_ITERATIONS):
            print(f"\n--- Iteration {i+1} ---")
            print(f"Current World Model: {self.world_model}")

            # 1. Understand & Plan (simulated by tool selection for now)
            tool_selection_prompt = self._get_llm_tool_prompt()
            llm_tool_response = json.loads(self._call_llm(tool_selection_prompt))
            tool_name = llm_tool_response.get("tool_name")
            tool_params = llm_tool_response.get("parameters", {})
            
            tool_output = ""
            if tool_name and tool_name != "NONE" and tool_name in self.tools:
                print(f"Agent selected tool: {tool_name} with params {tool_params}")
                try:
                    tool_func = self.tools[tool_name]["func"]
                    tool_output = tool_func(**tool_params)
                    self.world_model["last_tool_executed"] = tool_name
                    self.world_model["last_tool_output"] = tool_output
                    self._update_world_model("last_action", f"Tool executed: {tool_name}")
                    # Update specific world model items based on tool output
                    if tool_name == "get_order_status":
                        parsed_output = json.loads(tool_output)
                        self._update_world_model("order_status", parsed_output.get("status"))
                        self._update_world_model("estimated_delivery", parsed_output.get("estimated_delivery"))
                        self._update_world_model("last_order_id", tool_params.get("order_id"))
                    elif tool_name == "lookup_faq":
                        parsed_output = json.loads(tool_output)
                        self._update_world_model("faq_answer", parsed_output.get("answer"))
                        self._update_world_model("last_faq_topic", tool_params.get("topic"))
                    elif tool_name == "escalate_to_human":
                        self._update_world_model("escalated", True)


                except Exception as e:
                    tool_output = json.dumps({"error": str(e), "message": f"Error executing tool {tool_name}"})
                    self._update_world_model("last_action", f"Tool execution failed: {tool_name}")
                    print(f"Tool execution failed: {e}")
            else:
                print("Agent decided not to use a tool or no suitable tool found.")

            # 2. Generate Response
            response_prompt = self._get_llm_response_prompt(tool_output)
            agent_response = self._call_llm(response_prompt)
            self.conversation_history.append({"role": "assistant", "content": agent_response})
            self._update_world_model("last_agent_response", agent_response)
            print(f"Agent Response: {agent_response}")

            # 3. User Feedback (Simulated)
            user_feedback = input("Your feedback (e.g., 'good', 'bad', 'correct', 'incorrect', 'resolved'): ")
            self.conversation_history[-1]["feedback"] = user_feedback # Add feedback to last assistant turn
            self._update_world_model("last_user_feedback", user_feedback)
            
            # 4. Self-Reflect
            reflection_prompt = self._get_llm_reflection_prompt(last_action_success=("error" not in tool_output), user_feedback=user_feedback)
            llm_reflection_response = json.loads(self._call_llm(reflection_prompt))
            reflection_text = llm_reflection_response.get("reflection", "")
            reflection_action = llm_reflection_response.get("action", "CONTINUE")
            print(f"Self-Reflection: {reflection_text} (Action: {reflection_action})")
            self._update_world_model("last_reflection", reflection_text)

            # 5. Check Termination Conditions
            termination_prompt = self._get_llm_termination_prompt()
            llm_termination_response = json.loads(self._call_llm(termination_prompt))
            should_terminate = llm_termination_response.get("terminate", False)
            termination_reason = llm_termination_response.get("reason", "")

            if should_terminate or reflection_action == "TERMINATE" or "resolved" in user_feedback.lower() or self.world_model.get("escalated", False):
                print(f"\n--- Terminating Interaction ---\nReason: {termination_reason or reflection_text}")
                return agent_response
            
            if reflection_action == "REPLAN":
                print("Agent is replanning based on self-reflection.")
                # In a real system, this might involve more sophisticated plan adjustments.
                # For now, it just continues to the next iteration to re-evaluate.

        return "Maximum iterations reached. Please try again or contact support directly."

if __name__ == "__main__":
    agent = AdaptiveAgent()
    print("Adaptive Customer Support AI. Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        
        final_response = agent.interact(user_input)
        print(f"AI Final Response: {final_response}")

