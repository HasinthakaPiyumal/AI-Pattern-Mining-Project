from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DialogTurn(BaseModel):
    role: str  # "user" or "agent"
    content: str
    timestamp: str = Field(default_factory=lambda: "") # Placeholder for actual timestamp

class LLMResponse(BaseModel):
    response_text: str
    confidence_score: float = Field(..., ge=0, le=1)
    reasoning: Optional[str] = None

class WorkingMemory(BaseModel):
    user_query: str = ""
    external_evidence: Dict[str, Any] = Field(default_factory=dict)
    llm_candidate_responses: List[LLMResponse] = Field(default_factory=list)
    utility_scores: Dict[str, float] = Field(default_factory=dict)
    verbalized_feedback: str = ""
    dialog_history: List[DialogTurn] = Field(default_factory=list)

    def add_dialog_turn(self, role: str, content: str):
        self.dialog_history.append(DialogTurn(role=role, content=content))

    def update_user_query(self, query: str):
        self.user_query = query
        self.add_dialog_turn(role="user", content=query)

    def add_external_evidence(self, source: str, data: Any):
        self.external_evidence[source] = data

    def add_llm_response(self, response_text: str, confidence_score: float, reasoning: Optional[str] = None):
        self.llm_candidate_responses.append(LLMResponse(response_text=response_text, confidence_score=confidence_score, reasoning=reasoning))

    def set_utility_score(self, key: str, score: float):
        self.utility_scores[key] = score

    def update_verbalized_feedback(self, feedback: str):
        self.verbalized_feedback = feedback

    def reset_for_new_interaction(self):
        self.user_query = ""
        self.external_evidence = {}
        self.llm_candidate_responses = []
        self.utility_scores = {}
        self.verbalized_feedback = ""
        # Keep dialog history for context, but a new 'interaction' might imply clearing it too, 
        # depending on the desired scope of 'working memory'. For this pattern, we'll keep it.
        # self.dialog_history = []

    def get_current_state_summary(self) -> Dict[str, Any]:
        return {
            "user_query": self.user_query,
            "external_evidence_keys": list(self.external_evidence.keys()),
            "num_llm_responses": len(self.llm_candidate_responses),
            "utility_scores": self.utility_scores,
            "verbalized_feedback": self.verbalized_feedback,
            "dialog_history_length": len(self.dialog_history)
        }

    def __str__(self):
        return self.json(indent=2)
