from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Insight(BaseModel):
    type: Literal["bug_report", "feature_request", "general_feedback", "other"] = Field(
        ..., description="The type of insight extracted from the review."
    )
    description: str = Field(..., description="Detailed description of the insight.")
    severity: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Severity of the issue, if applicable."
    )

class ProductReviewOutput(BaseModel):
    review_id: str = Field(..., description="Unique identifier for the review.")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        ..., description="Overall sentiment of the review."
    )
    features_mentioned: List[str] = Field(
        default_factory=list, description="List of product features mentioned in the review."
    )
    actionable_insights: List[Insight] = Field(
        default_factory=list, description="List of actionable insights derived from the review."
    )
    raw_llm_output: Optional[str] = Field(
        None, description="The raw JSON string returned by the LLM (for debugging)."
    )
    validation_error: Optional[str] = Field(
        None, description="Error message if validation failed during processing."
    )