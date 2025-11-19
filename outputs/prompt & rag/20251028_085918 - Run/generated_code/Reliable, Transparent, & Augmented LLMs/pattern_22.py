import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from guardrails.hub import Competency
from guardrails import Guard
from fastapi import FastAPI
import uvicorn

class ToolCall(BaseModel):
    name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(description="Arguments for the tool")

class AgentResponse(BaseModel):
    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of tools to call, if any.")
    reasoning_path: str = Field(description="The step-by-step reasoning process to arrive at the answer.")
    confidence_score: float = Field(description="A self-rated confidence score (0.0-1.0) for the generated response.")
    answer: str = Field(description="The final answer or instructions for the user.")
    human_handoff_needed: bool = Field(default=False, description="True if the complexity or low confidence warrants human intervention.")

class ToolManager:
    def knowledge_base_lookup(self, query: str) -> str:
        if "shipping" in query.lower():
            return "Standard shipping takes 3-5 business days. Expedited shipping takes 1-2 business days. Tracking details are available in your order history."
        elif "return policy" in query.lower():
            return "Our return policy allows returns within 30 days of purchase for a full refund, provided the item is in its original condition. Some exclusions apply, please check our website for details."
        return "Could not find relevant information in the knowledge base for your query."

    def order_status(self, order_id: str) -> str:
        if order_id == "12345":
            return "Order 12345: Shipped on 2023-10-26, estimated delivery 2023-10-29. Tracking #: TRK12345."
        elif order_id == "67890":
            return "Order 67890: Processing. Expected to ship within 2 business days."
        return f"Order {order_id} not found."

    def troubleshooting_guide(self, issue: str) -> str:
        if "login issue" in issue.lower():
            return "For login issues, first try resetting your password. If that doesn't work, ensure your internet connection is stable. If problems persist, contact support with your username."
        elif "payment failed" in issue.lower():
            return "If your payment failed, please verify your card details, ensure sufficient funds, or try a different payment method. If the issue continues, contact your bank."
        return "No specific troubleshooting guide found for this issue."

    def call_tool(self, tool_call: ToolCall) -> str:
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        try:
            if hasattr(self, tool_name):
                tool_func = getattr(self, tool_name)
                return tool_func(**tool_args)
            else:
                return f"Error: Tool '{tool_name}' not found."
        except TypeError as e:
            return f"Error calling tool '{tool_name}': Invalid arguments. Details: {e}"
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

class MockLLM:
    def __init__(self):
        self.tool_manager = ToolManager()
        self.guard = Guard.from_pydantic(AgentResponse)

    def _generate_mock_response(self, query: str, tool_results: Dict[str, str] = None) -> AgentResponse:
        query_lower = query.lower()
        reasoning = f"User asked: '{query}'. "
        confidence = 0.8
        answer = "I'm sorry, I don't have enough information to answer that question."
        tool_calls = []
        human_handoff = False

        if "shipping" in query_lower or "delivery" in query_lower:
            reasoning += "Recognized shipping/delivery query. Calling knowledge_base_lookup tool."
            tool_calls.append(ToolCall(name="knowledge_base_lookup", arguments={"query": "shipping policy"}))
            confidence = 0.9
        elif "return" in query_lower:
            reasoning += "Recognized return policy query. Calling knowledge_base_lookup tool."
            tool_calls.append(ToolCall(name="knowledge_base_lookup", arguments={"query": "return policy"}))
            confidence = 0.9
        elif "order status" in query_lower:
            order_id = next((s for s in query.split() if s.isdigit() and len(s) == 5), None)
            if order_id:
                reasoning += f"Identified order ID {order_id}. Calling order_status tool."
                tool_calls.append(ToolCall(name="order_status", arguments={"order_id": order_id}))
                confidence = 0.95
            else:
                reasoning += "Could not find order ID. Asking user for more info."
                answer = "To check your order status, please provide your order ID."
                confidence = 0.7
        elif "login issue" in query_lower or "cant log in" in query_lower:
            reasoning += "Recognized login issue. Calling troubleshooting_guide tool."
            tool_calls.append(ToolCall(name="troubleshooting_guide", arguments={"issue": "login issue"}))
            confidence = 0.85
        elif "payment failed" in query_lower:
            reasoning += "Recognized payment failure issue. Calling troubleshooting_guide tool."
            tool_calls.append(ToolCall(name="troubleshooting_guide", arguments={"issue": "payment failed"}))
            confidence = 0.85
        elif "complex" in query_lower:
            reasoning += "Query identified as complex. Suggesting human handoff."
            answer = "This seems like a complex issue that requires a human expert. I'm escalating you to a human agent."
            confidence = 0.6
            human_handoff = True
        else:
            reasoning += "No specific tools identified. Attempting to answer generally or indicating inability."
            confidence = 0.7
            if "hello" in query_lower or "hi" in query_lower:
                answer = "Hello! How can I assist you today?"
            elif "thanks" in query_lower:
                answer = "You're welcome! Is there anything else I can help with?"
            else:
                answer = "I am an AI customer support assistant. How can I help you with your order, shipping, or technical issues today?"
                confidence = 0.7

        if tool_results:
            reasoning += "\nProcessing tool results."
            final_answer_parts = []
            for tool_name, result in tool_results.items():
                reasoning += f"\nTool '{tool_name}' returned: {result}"
                final_answer_parts.append(f"Result from {tool_name.replace('_', ' ')}: {result}")
            if final_answer_parts:
                answer = "\n".join(final_answer_parts)
                confidence = min(confidence + 0.1, 1.0)

        if confidence < 0.7 and not human_handoff:
            reasoning += "\nConfidence is low, considering human handoff."
            human_handoff = True
            answer = answer + "\n\nI'm not entirely confident in this answer. Would you like me to connect you with a human agent?"

        return AgentResponse(
            tool_calls=tool_calls,
            reasoning_path=reasoning,
            confidence_score=confidence,
            answer=answer,
            human_handoff_needed=human_handoff
        )

    def generate_response(self, query: str, tool_results: Optional[Dict[str, str]] = None) -> AgentResponse:
        raw_response = self._generate_mock_response(query, tool_results)

        try:
            validated_response = self.guard.validate(raw_response.model_dump_json())
            return AgentResponse.model_validate(json.loads(validated_response.rail.output))
        except Exception as e:
            print(f"Guardrails validation failed: {e}")
            return raw_response

class CustomerSupportAgent:
    def __init__(self):
        self.llm = MockLLM()
        self.tool_manager = ToolManager()
        self.conversation_history = []
        self.guard = Guard().use(Competency(
            competency_name="answer_quality",
            description="Assess if the answer is helpful, relevant, and not hallucinated.",
            statements=["The answer directly addresses the user's query.", "The answer is factually correct.", "The answer is helpful and actionable."],
            score_range=(1, 5)
        ))

    def _format_response(self, agent_response: AgentResponse) -> str:
        formatted_output = f"**AI Assistant Response**\n\n{agent_response.answer}\n\n"
        formatted_output += f"**Confidence Score:** {agent_response.confidence_score:.2f}/1.00\n"
        formatted_output += f"**Reasoning Path:**\n{agent_response.reasoning_path}\n"

        if agent_response.human_handoff_needed:
            formatted_output += "\n**Note:** This interaction has been flagged for human review/escalation due to complexity or low confidence."
            formatted_output += f"\n\n**Summary for Human Agent:**\nUser query: {self.conversation_history[-1]['user_query'] if self.conversation_history else 'N/A'}\nAI Reasoning: {agent_response.reasoning_path}"

        return formatted_output

    def process_query(self, query: str) -> str:
        self.conversation_history.append({"user_query": query})

        initial_llm_response = self.llm.generate_response(query)

        tool_results = {}
        if initial_llm_response.tool_calls:
            print(f"Detected tool calls: {initial_llm_response.tool_calls}")
            for tool_call in initial_llm_response.tool_calls:
                result = self.tool_manager.call_tool(tool_call)
                tool_results[tool_call.name] = result
                print(f"Tool {tool_call.name} result: {result}")

        if tool_results:
            final_llm_response = self.llm.generate_response(query, tool_results)
        else:
            final_llm_response = initial_llm_response

        self.evaluate_response(final_llm_response)

        formatted_output = self._format_response(final_llm_response)
        self.conversation_history.append({"ai_response": formatted_output, "raw_agent_response": final_llm_response.model_dump()})
        return formatted_output

    def evaluate_response(self, agent_response: AgentResponse):
        try:
            if agent_response.confidence_score < 0.5:
                print(f"Evaluation Warning: Low confidence response ({agent_response.confidence_score:.2f}). Consider review.")
            else:
                print(f"Evaluation: Response quality appears acceptable (Confidence: {agent_response.confidence_score:.2f}).")

            validation_result = self.guard.validate(llm_output={"answer_quality": agent_response.answer})

        except Exception as e:
            print(f"Error during evaluation: {e}")

app = FastAPI(
    title="Agentic & Trustworthy AI Customer Support Assistant",
    description="An AI assistant that provides transparent, reliable, and agentic customer support.",
    version="1.0.0",
)

agent = CustomerSupportAgent()

@app.post("/chat")
async def chat_with_assistant(query: Dict[str, str]):
    user_query = query.get("query")
    if not user_query:
        return {"error": "Query parameter is required."}

    response = agent.process_query(user_query)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
