class WorkingMemory:
    def __init__(self):
        self.current_query = None
        self.external_evidence = []
        self.llm_responses = []
        self.utility_scores = []
        self.verbalized_feedback = None
        self.dialog_history = []

    def update_query(self, query: str):
        self.current_query = query

    def add_external_evidence(self, evidence: str):
        self.external_evidence.append(evidence)

    def add_llm_response(self, response: str, score: float = None):
        self.llm_responses.append(response)
        if score is not None:
            self.utility_scores.append(score)

    def add_feedback(self, feedback: str):
        self.verbalized_feedback = feedback

    def add_dialog_turn(self, user_message: str, agent_response: str):
        self.dialog_history.append((user_message, agent_response))

    def get_current_state(self):
        return {
            "current_query": self.current_query,
            "external_evidence": self.external_evidence,
            "llm_responses": self.llm_responses,
            "utility_scores": self.utility_scores,
            "verbalized_feedback": self.verbalized_feedback,
            "dialog_history": self.dialog_history,
        }

    def reset_turn_specific_memory(self):
        self.llm_responses = []
        self.utility_scores = []
        self.verbalized_feedback = None