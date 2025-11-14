from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="The name of the tool called.")
    tool_input: Dict[str, Any] = Field(..., description="The input parameters for the tool call.")
    tool_output: Optional[str] = Field(None, description="The output received from the tool.")

class ReasoningStep(BaseModel):
    step_type: str = Field(..., description="Type of reasoning step (e.g., 'thought', 'tool_call', 'observation', 'final_answer').")
    content: Any = Field(..., description="Content of the reasoning step.")

class ChatRequest(BaseModel):
    query: str = Field(..., description="The customer's natural language query.")

class AgentResponse(BaseModel):
    response: str = Field(..., description="The agent's synthesized response to the customer.")
    reasoning_path: List[ReasoningStep] = Field(..., description="A detailed path of the agent's reasoning and tool calls.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="A confidence score for the agent's response (0.0 to 1.0).")
    escalate_to_human: bool = Field(False, description="True if the agent recommends escalating to a human agent.")
