from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CustomerQuery(BaseModel):
    text_input: Optional[str] = Field(None, description="Original text input from the customer.")
    audio_file_path: Optional[str] = Field(None, description="Path to an audio file (e.g., voice note).")
    image_file_path: Optional[str] = Field(None, description="Path to an image file (e.g., product damage).")
    language: str = Field("en", description="Detected or specified language of the query (ISO 639-1 code).")

class Intent(BaseModel):
    name: str = Field(..., description="Name of the recognized intent (e.g., \'check_order_status\', \'report_product_issue\').")
    confidence: float = Field(..., description="Confidence score of the intent recognition.")
    entities: Dict[str, Any] = Field({}, description="Extracted entities and their values.")

class AgentResponse(BaseModel):
    response_text: str = Field(..., description="The generated textual response to the customer.")
    intent_recognized: Optional[Intent] = Field(None, description="The primary intent recognized.")
    requires_escalation: bool = Field(False, description="True if the query needs to be escalated to a human agent.")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation, if applicable.")
    action_taken: Optional[str] = Field(None, description="Description of any backend action taken (e.g., \'order_status_retrieved\').")
