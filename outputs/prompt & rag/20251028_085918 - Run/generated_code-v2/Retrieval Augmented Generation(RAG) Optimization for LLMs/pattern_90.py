class UnifiedInstructionFormatter:
    """Manages the standardization of diverse inputs into a unified instruction format
    for multi-task LLM training and inference.
    The format is generally: 'instruction_template [CONTEXT] context_payload [TARGET_OUTPUT_PLACEHOLDER]'
    """

    def format_qa_instruction(self, question: str, context: str = None) -> str:
        """Formats an instruction for Context-rich QA.
        Example: 'Answer the following question from context [Passage]...'
        """
        instruction = f"Answer the following question. Use the provided context if available."
        if context:
            return f"[INST] {instruction}\n[CONTEXT] {context}\n[QUESTION] {question}\n[ANSWER]" # Target output is expected after [ANSWER]
        else:
            return f"[INST] {instruction}\n[QUESTION] {question}\n[ANSWER]"

    def format_ranking_instruction(self, question: str, passage: str) -> str:
        """Formats an instruction for Context Ranking.
        Example: 'For the question [question] assess whether the passage [Passage] is relevant...'
        """
        instruction = f"Assess whether the provided passage is relevant to the question. Return 'True' if relevant, otherwise 'False'."
        return f"[INST] {instruction}\n[QUESTION] {question}\n[PASSAGE] {passage}\n[RELEVANCE]"

    def format_retrieval_augmented_ranking_instruction(self, question: str, passages: list[str]) -> str:
        """Formats an instruction for Retrieval-Augmented Ranking (finding relevant passages).
        Example: 'For the question [question] find all passages from [Passage 1]... that are relevant...'
        """
        instruction = f"From the following passages, identify and return the IDs of all passages that are relevant to the question. Respond with a comma-separated list of relevant passage IDs. If no passages are relevant, respond 'None'."
        passage_blocks = ""
        for i, p in enumerate(passages):
            passage_blocks += f"[PASSAGE_ID_{i+1}] {p}\n"
        return f"[INST] {instruction}\n[QUESTION] {question}\n{passage_blocks}[RELEVANT_PASSAGE_IDS]"

    def format_treatment_plan_instruction(self, patient_data: str, medical_guidelines: str) -> str:
        """Formats an instruction for Treatment Plan Generation.
        """
        instruction = f"Based on the patient data and medical guidelines, suggest potential treatment options and consider drug interactions."
        return f"[INST] {instruction}\n[PATIENT_DATA] {patient_data}\n[MEDICAL_GUIDELINES] {medical_guidelines}\n[TREATMENT_PLAN]"

    def format_patient_education_instruction(self, medical_concept: str, target_audience: str = "patient") -> str:
        """Formats an instruction for Patient Education Content Generation.
        """
        instruction = f"Explain the following medical concept in simple, easy-to-understand language suitable for a {target_audience}."
        return f"[INST] {instruction}\n[MEDICAL_CONCEPT] {medical_concept}\n[EXPLANATION]"




