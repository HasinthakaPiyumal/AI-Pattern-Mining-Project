
import collections

class WorkingMemory:
    """Manages the dynamic context for prompt engineering."""
    def __init__(self):
        self.task_instructions = "You are a medical diagnostic assistant. Provide differential diagnoses, potential causes, and suggest initial steps based on the patient's symptoms and history. Always ask clarifying questions if information is insufficient. Focus on evidence-based reasoning." # Default instruction
        self.user_query = ""
        self.dialog_history = collections.deque(maxlen=10) # Stores recent interactions
        self.external_evidence = ""
        self.feedback = ""

    def update_task_instructions(self, instructions: str):
        self.task_instructions = instructions

    def update_user_query(self, query: str):
        self.user_query = query

    def add_to_dialog_history(self, speaker: str, text: str):
        self.dialog_history.append((speaker, text))

    def get_dialog_history_str(self):
        return "\n".join([f"{s}: {t}" for s, t in self.dialog_history])

    def update_external_evidence(self, evidence: str):
        self.external_evidence = evidence

    def update_feedback(self, feedback: str):
        self.feedback = feedback

    def reset_context(self):
        self.user_query = ""
        self.dialog_history.clear()
        self.external_evidence = ""
        self.feedback = ""


class KnowledgeConsolidator:
    """Simulates fetching and consolidating external medical evidence."""
    def __init__(self, knowledge_base_data=None):
        # In a real system, this would connect to databases, patient records, etc.
        self.knowledge_base_data = knowledge_base_data or {
            "fever headache": "Common causes include viral infections (e.g., flu, common cold), tension headaches, or sometimes more serious conditions like meningitis. Consider recent travel or exposure. Relevant lab: CBC, CRP.",
            "chest pain shortness breath": "Possible myocardial infarction, pneumonia, pleurisy, or anxiety. Immediate ECG and cardiac markers are often necessary. History of smoking, hypertension, diabetes are risk factors.",
            "abdominal pain nausea": "Could be gastroenteritis, appendicitis, gallstones, or food poisoning. Location and character of pain are crucial. Relevant labs: CBC, LFTs, amylase/lipase, urinalysis."
        }

    def get_evidence(self, query: str) -> str:
        """Retrieves relevant evidence based on the query."""
        # Simple keyword matching for demonstration
        for keyword, evidence in self.knowledge_base_data.items():
            if all(k in query.lower() for k in keyword.split()):
                return f"\n--- External Medical Evidence ---\n{evidence}"
        return "\n--- External Medical Evidence ---\nNo specific evidence found for this query in the immediate knowledge base."


class UtilityModule:
    """Simulates generating automated feedback for iterative refinement."""
    def __init__(self):
        pass

    def generate_feedback(self, previous_response: str, expected_outcome: str = None) -> str:
        """Generates mock feedback based on a previous response.
           In a real system, this could involve evaluating response quality,
           factual accuracy, or adherence to guidelines.
        """
        if "insufficient information" in previous_response.lower():
            return "\n--- Automated Feedback ---\nPrevious response indicated insufficient info. Prompt for more details related to onset, duration, and severity."
        elif "anxiety" in previous_response.lower() and "chest pain" in previous_response.lower() and expected_outcome and "cardiac" in expected_outcome.lower():
             return "\n--- Automated Feedback ---\nConsidered anxiety for chest pain, but primary concern was cardiac. Ensure prompt emphasizes rule-out serious conditions first."
        else:
            return "\n--- Automated Feedback ---\nNo specific feedback generated for this interaction. Continue with standard procedure."


class PromptEngine:
    """Dynamically constructs prompts for the LLM based on various contextual elements."""
    def __init__(self, working_memory: WorkingMemory):
        self.working_memory = working_memory

    def construct_prompt(self) -> str:
        """Builds a comprehensive prompt string from working memory components."""
        prompt_parts = [
            f"[TASK INSTRUCTIONS]\n{self.working_memory.task_instructions}"
        ]

        if self.working_memory.get_dialog_history_str():
            prompt_parts.append(f"\n[DIALOG HISTORY]\n{self.working_memory.get_dialog_history_str()}")

        if self.working_memory.external_evidence:
            prompt_parts.append(f"\n[EXTERNAL EVIDENCE]\n{self.working_memory.external_evidence}")

        if self.working_memory.feedback:
            prompt_parts.append(f"\n[PREVIOUS FEEDBACK]\n{self.working_memory.feedback}")

        prompt_parts.append(f"\n[CURRENT USER QUERY]\nDoctor: {self.working_memory.user_query}\nAssistant:")

        return "\n\n".join(prompt_parts)


class LLM_Simulator:
    """A mock LLM to simulate diagnostic responses."""
    def __init__(self):
        pass

    def generate_response(self, prompt: str) -> str:
        """Generates a mock diagnostic response based on the prompt."""
        print(f"\n--- LLM Input Prompt ---\n{prompt}\n---\n")

        if "fever" in prompt.lower() and "headache" in prompt.lower() and "viral infections" in prompt.lower():
            return "Based on symptoms (fever, headache) and external evidence, a viral infection (e.g., flu, common cold) is highly probable. Consider ruling out meningitis if neck stiffness or photophobia are present. Suggest a CBC and CRP. Ask about recent travel and vaccination status."
        elif "chest pain" in prompt.lower() and "shortness of breath" in prompt.lower() and "myocardial infarction" in prompt.lower():
            return "Given chest pain and shortness of breath, myocardial infarction is a critical concern requiring immediate investigation (ECG, cardiac markers). Pneumonia or pleurisy are also possibilities. Inquire about pain character, radiation, and any history of cardiac issues or risk factors."
        elif "abdominal pain" in prompt.lower() and "nausea" in prompt.lower():
            return "Differential diagnoses for abdominal pain and nausea include gastroenteritis, appendicitis (if localized RLQ), gallstones (if RUQ), or food poisoning. Clarify pain location, onset, severity, and any associated symptoms like vomiting or diarrhea. Suggest CBC, LFTs, amylase/lipase, and urinalysis."
        elif "insufficient information" in prompt.lower() and "onset, duration, and severity" in prompt.lower():
            return "Please provide more details on the onset (when did symptoms start?), duration (how long have they lasted?), and severity (on a scale of 1-10) of the patient's symptoms. This will help refine the diagnosis."
        else:
            return "I need more information to provide an accurate diagnosis. Could you please specify the patient's age, gender, exact symptoms, and medical history?"


class MedicalDiagnosticAssistant:
    """Orchestrates the diagnostic process using contextual prompt engineering."""
    def __init__(self, llm_simulator: LLM_Simulator):
        self.working_memory = WorkingMemory()
        self.knowledge_consolidator = KnowledgeConsolidator()
        self.utility_module = UtilityModule()
        self.prompt_engine = PromptEngine(self.working_memory)
        self.llm_simulator = llm_simulator
        self.last_llm_response = ""

    def process_query(self, doctor_query: str, patient_info: dict = None, previous_feedback_needed: bool = False) -> str:
        """Processes a doctor's query to provide diagnostic support."""
        print(f"\n--- Doctor's Query ---\n{doctor_query}")

        self.working_memory.update_user_query(doctor_query)

        # 1. Get external evidence
        external_evidence = self.knowledge_consolidator.get_evidence(doctor_query)
        self.working_memory.update_external_evidence(external_evidence)

        # 2. Apply automated feedback if needed (simulating iterative refinement)
        if previous_feedback_needed and self.last_llm_response:
            feedback = self.utility_module.generate_feedback(self.last_llm_response, expected_outcome=doctor_query) # Simplistic expected_outcome
            self.working_memory.update_feedback(feedback)
        else:
            self.working_memory.update_feedback("") # Clear feedback if not needed

        # 3. Construct the prompt
        full_prompt = self.prompt_engine.construct_prompt()

        # 4. Get LLM response
        llm_response = self.llm_simulator.generate_response(full_prompt)
        self.last_llm_response = llm_response

        # 5. Update dialog history
        self.working_memory.add_to_dialog_history("Doctor", doctor_query)
        self.working_memory.add_to_dialog_history("Assistant", llm_response)

        print(f"\n--- Diagnostic Assistant Response ---\n{llm_response}")
        return llm_response


if __name__ == "__main__":
    llm_mock = LLM_Simulator()
    assistant = MedicalDiagnosticAssistant(llm_mock)

    print("\n--- Scenario 1: Initial Query ---")
    response1 = assistant.process_query("Patient presents with fever and a persistent headache. No other symptoms reported yet.")

    print("\n--- Scenario 2: Refining with more details based on previous suggestion ---")
    # Simulate the assistant asking for more info and the doctor providing it
    assistant.working_memory.update_user_query("The fever started 2 days ago, headache has been constant at 6/10 intensity since then. No neck stiffness or photophobia. Patient recently returned from a trip to Southeast Asia. Vaccinations up to date.")
    assistant.working_memory.add_to_dialog_history("Assistant", response1) # Add previous response to history for proper context
    response2 = assistant.process_query("What is the differential diagnosis now, considering recent travel and no meningeal signs?", previous_feedback_needed=True)

    print("\n--- Scenario 3: New patient, different symptoms ---")
    assistant.working_memory.reset_context()
    response3 = assistant.process_query("A 55-year-old male with sudden onset chest pain radiating to the left arm and shortness of breath. History of hypertension.")

    print("\n--- Scenario 4: Iterative refinement with feedback ---")
    response4_1 = assistant.process_query("Initial assessment: Could this be anxiety given the patient's stress levels?", patient_info={"stress_level": "high"})

    # Simulate feedback for a potentially missed severe condition
    print("\n--- Scenario 4.2: Assistant gets feedback and re-evaluates ---")
    response4_2 = assistant.process_query("Re-evaluate chest pain, ensuring cardiac causes are prioritized, even with stress factors.", previous_feedback_needed=True)
