import os
from typing import List, Dict, Any
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1. Working Memory Module ---

class PatientCase(BaseModel):
    initial_symptoms: str = ""
    current_questions: List[str] = []
    differential_diagnoses: List[str] = []
    final_diagnosis: str = ""

class DialogueEntry(BaseModel):
    role: str
    content: str

class EvidenceEntry(BaseModel):
    source: str
    content: str

class WorkingMemory:
    def __init__(self):
        self.patient_case = PatientCase()
        self.dialogue_history: List[DialogueEntry] = []
        self.evidence_base: List[EvidenceEntry] = []
        self.llm_intermediate_outputs: List[str] = []
        logger.info("Working Memory initialized.")

    def add_symptoms(self, symptoms: str):
        self.patient_case.initial_symptoms = symptoms
        self.add_dialogue("user", f"Patient presents with: {symptoms}")
        logger.info(f"Symptoms added to working memory: {symptoms}")

    def add_dialogue(self, role: str, content: str):
        self.dialogue_history.append(DialogueEntry(role=role, content=content))
        logger.debug(f"Dialogue added: [{role}] {content}")

    def add_evidence(self, source: str, content: str):
        self.evidence_base.append(EvidenceEntry(source=source, content=content))
        logger.info(f"Evidence added from {source}: {content[:50]}...")

    def update_llm_output(self, output: str):
        self.llm_intermediate_outputs.append(output)
        self.add_dialogue("llm", output)
        logger.debug(f"LLM output updated: {output[:50]}...")

    def update_differential_diagnoses(self, diagnoses: List[str]):
        self.patient_case.differential_diagnoses = list(set(self.patient_case.differential_diagnoses + diagnoses))
        logger.info(f"Differential diagnoses updated: {diagnoses}")

    def set_final_diagnosis(self, diagnosis: str):
        self.patient_case.final_diagnosis = diagnosis
        logger.info(f"Final diagnosis set: {diagnosis}")

    def get_current_context(self) -> str:
        context_parts = []
        context_parts.append(f"Patient Symptoms: {self.patient_case.initial_symptoms}")
        if self.patient_case.differential_diagnoses:
            context_parts.append(f"Potential Diagnoses: {', '.join(self.patient_case.differential_diagnoses)}")
        
        dialogue_context = "\n".join([f"{d.role.capitalize()}: {d.content}" for d in self.dialogue_history])
        context_parts.append(f"Dialogue History:\n{dialogue_context}")
        
        if self.evidence_base:
            evidence_context = "\n".join([f"Evidence from {e.source}: {e.content}" for e in self.evidence_base])
            context_parts.append(f"Retrieved Evidence:\n{evidence_context}")

        return "\n\n".join(context_parts)

    def reset_memory(self):
        self.patient_case = PatientCase()
        self.dialogue_history = []
        self.evidence_base = []
        self.llm_intermediate_outputs = []
        logger.info("Working Memory reset.")


# --- 2. Medical Knowledge Base & Retrieval Module ---

class MedicalKnowledgeBase:
    def __init__(self):
        # Placeholder for vector store and embedding model
        # In a real application, you would initialize Chroma/FAISS and a sentence-transformer model here.
        # For demonstration, we use a simple dict lookup.
        self.medical_data = {
            "fever and cough": [
                {"source": "CDC Guidelines", "content": "Fever and cough are common symptoms of respiratory infections, including influenza and common cold. Consider viral panel testing."},
                {"source": "WHO Report 2023", "content": "Persistent cough accompanied by high fever may indicate pneumonia, especially in elderly patients. Chest X-ray recommended."},
            ],
            "chest pain and shortness of breath": [
                {"source": "AHA Guidelines", "content": "Acute chest pain with shortness of breath requires immediate evaluation for cardiac events (e.g., myocardial infarction, angina). ECG and cardiac markers are crucial."},
                {"source": "Pulmonology Journal 2022", "content": "Pleuritic chest pain and dyspnea can also suggest pulmonary embolism or pleurisy. D-dimer and CT angiography may be indicated."},
            ],
            "abdominal pain": [
                {"source": "Gastroenterology Handbook", "content": "Abdominal pain has numerous causes, including appendicitis, gastritis, IBS, and kidney stones. Localization and character of pain are key for diagnosis."}
            ]
        }
        logger.info("Medical Knowledge Base initialized with mock data.")

    def retrieve_information(self, query: str, top_k: int = 2) -> List[EvidenceEntry]:
        logger.info(f"Retrieving information for query: '{query}'")
        # In a real system, embed the query and perform a vector search.
        # For this demo, we'll do a simple keyword match.
        retrieved = []
        query_lower = query.lower()
        for key, evidences in self.medical_data.items():
            if key in query_lower or any(word in query_lower for word in key.split()):
                for evidence in evidences:
                    retrieved.append(EvidenceEntry(**evidence))
                    if len(retrieved) >= top_k:
                        return retrieved
        
        # Fallback if no specific match
        if not retrieved and "fever" in query_lower:
             return [EvidenceEntry(source="General Symptoms", content="Fever often indicates an infection or inflammatory process. Further investigation is needed.")]
        if not retrieved and "pain" in query_lower:
            return [EvidenceEntry(source="General Symptoms", content="Pain is a common symptom with various etiologies. Location and characteristics are vital.")]

        return retrieved


# --- 3. Blackbox LLM Wrapper ---

class LLMInterface:
    def __init__(self, api_key: str = "dummy_api_key"):
        self.api_key = api_key # In a real app, this would be used by openai.OpenAI(api_key=api_key)
        # self.client = openai.OpenAI(api_key=api_key)
        logger.info("LLM Interface initialized with a dummy API key.")

    def generate_response(self, prompt: str) -> str:
        logger.info(f"Sending prompt to LLM: {prompt[:100]}...")
        # This is a mock LLM response. In a real application, you'd call the LLM API.
        # Example using a simplified logic based on keywords in the prompt.
        
        if "initial differential diagnoses" in prompt.lower():
            if "fever and cough" in prompt.lower():
                return "Based on fever and cough, consider common cold, influenza, bronchitis, or early pneumonia. What other symptoms are present?"
            elif "chest pain and shortness of breath" in prompt.lower():
                return "Acute chest pain and dyspnea are concerning. Rule out myocardial infarction, pulmonary embolism, and pericarditis first. Any radiating pain or risk factors?"
            elif "abdominal pain" in prompt.lower():
                return "Abdominal pain is broad. Possible diagnoses include gastritis, appendicitis, IBS, or urinary tract infection. Where exactly is the pain located and what is its character?"
            else:
                return "Please provide more specific symptoms for a differential diagnosis."
        elif "verify facts" in prompt.lower() or "gather more information" in prompt.lower():
            return "I need to consult the medical knowledge base for evidence related to the current patient context and differential diagnoses."
        elif "synthesize information" in prompt.lower():
            return "Synthesizing information from patient data and retrieved medical guidelines to propose a diagnosis."
        else:
            return "Acknowledged. Please provide further instructions or patient details."


# --- 4. Diagnosis Policy Module & Explainability ---

class DiagnosisPolicy:
    def __init__(self, working_memory: WorkingMemory, knowledge_base: MedicalKnowledgeBase, llm_interface: LLMInterface):
        self.working_memory = working_memory
        self.knowledge_base = knowledge_base
        self.llm_interface = llm_interface
        logger.info("Diagnosis Policy initialized.")

    def generate_explanation(self, final_diagnosis: str, evidence: List[EvidenceEntry]) -> str:
        explanation_parts = [f"## Final Diagnosis: {final_diagnosis}", "\n### Supporting Evidence:"]
        for i, entry in enumerate(evidence):
            explanation_parts.append(f"{i+1}. {entry.content} (Source: {entry.source})")
        logger.info("Explanation generated.")
        return "\n".join(explanation_parts)

    def run_diagnosis(self, patient_symptoms: str, max_turns: int = 5) -> Dict[str, Any]:
        self.working_memory.reset_memory()
        self.working_memory.add_symptoms(patient_symptoms)
        logger.info(f"Starting diagnosis for patient with symptoms: {patient_symptoms}")

        for turn in range(max_turns):
            logger.info(f"--- Diagnosis Turn {turn + 1} ---")
            current_context = self.working_memory.get_current_context()
            logger.debug(f"Current context for LLM:\n{current_context[:500]}...")

            # Step 1: Query LLM for initial differential diagnoses or clarifying questions
            llm_prompt = f"Given the following patient context, provide initial differential diagnoses or suggest clarifying questions. Format as a comma-separated list of diagnoses or a question.\n\n{current_context}\n\nSuggest: "
            llm_response = self.llm_interface.generate_response(llm_prompt)
            self.working_memory.update_llm_output(llm_response)

            if "suggest" in llm_response.lower() or "what other symptoms" in llm_response.lower() or "any radiating pain" in llm_response.lower() or "where exactly is the pain" in llm_response.lower():
                # LLM is asking for clarifying questions
                logger.info(f"LLM requested clarification: {llm_response}")
                self.working_memory.add_dialogue("system", f"LLM asked: {llm_response}. Simulating user response (adding more symptoms for demonstration).")
                # For demo, simulate adding more symptoms or answers
                if "fever and cough" in patient_symptoms.lower():
                    self.working_memory.add_dialogue("user", "Patient also reports headache and muscle aches, no rash.")
                    self.working_memory.add_symptoms("fever, cough, headache, muscle aches") # Update initial symptoms to reflect new info
                elif "chest pain" in patient_symptoms.lower():
                    self.working_memory.add_dialogue("user", "The pain radiates to the left arm and jaw. Patient has a history of hypertension.")
                    self.working_memory.add_symptoms("chest pain, shortness of breath, left arm/jaw pain, history of hypertension") # Update initial symptoms
                elif "abdominal pain" in patient_symptoms.lower():
                    self.working_memory.add_dialogue("user", "Pain is sharp, localized to the right lower quadrant, started 6 hours ago.")
                    self.working_memory.add_symptoms("abdominal pain (sharp, R-lower quadrant, 6hrs), no fever") # Update initial symptoms

                continue # Continue to next turn with updated info

            # Parse LLM's differential diagnoses
            diagnoses = [d.strip() for d in llm_response.replace("Based on", "").split(",") if d.strip() and "consider" not in d.lower() and "What other symptoms" not in d and "Rule out" not in d and "Any radiating pain" not in d and "Where exactly is the pain" not in d]
            diagnoses = [d.replace("or early pneumonia", "early pneumonia").replace("or urinary tract infection", "urinary tract infection").strip() for d in diagnoses]
            # Filter out clarifying questions or instructions from LLM response
            diagnoses = [d for d in diagnoses if not d.lower().startswith("what") and not d.lower().startswith("any") and not d.lower().startswith("where")]

            if diagnoses:
                self.working_memory.update_differential_diagnoses(diagnoses)
                logger.info(f"Identified differential diagnoses: {diagnoses}")

                # Step 2: Use Medical Knowledge Base to retrieve evidence for current diagnoses
                for diag in self.working_memory.patient_case.differential_diagnoses:
                    retrieved_evidence = self.knowledge_base.retrieve_information(diag)
                    for ev in retrieved_evidence:
                        self.working_memory.add_evidence(ev.source, ev.content)

                # Step 3: Query LLM again to synthesize and refine diagnosis with evidence
                synthesis_prompt = f"Given the patient's symptoms, differential diagnoses ({', '.join(self.working_memory.patient_case.differential_diagnoses)}), and the following medical evidence, synthesize a refined diagnosis and justification. If unsure, state what further information is needed.\n\n{self.working_memory.get_current_context()}\n\nRefined Diagnosis: "
                llm_refined_response = self.llm_interface.generate_response(synthesis_prompt)
                self.working_memory.update_llm_output(llm_refined_response)
                logger.info(f"LLM refined response: {llm_refined_response}")

                # Simple heuristic to determine if a final diagnosis is reached
                if "diagnosis is" in llm_refined_response.lower() or "likely to be" in llm_refined_response.lower():
                    final_diag_match = llm_refined_response.split("Diagnosis is ")[-1].split(".")[0].strip()
                    if final_diag_match:
                        self.working_memory.set_final_diagnosis(final_diag_match)
                        logger.info(f"Final diagnosis proposed by LLM: {final_diag_match}")
                        break # Exit loop if a final diagnosis is reached

            else:
                logger.warning("LLM did not provide clear differential diagnoses. Asking for more patient details.")
                self.working_memory.add_dialogue("system", "The LLM needs more details. What else can you tell me about the patient's condition?")
                # For demo, break if LLM is repeatedly asking for more info without progress
                if turn == max_turns - 1: 
                    self.working_memory.set_final_diagnosis("Undetermined due to insufficient information.")

        final_diagnosis = self.working_memory.patient_case.final_diagnosis
        supporting_evidence = self.working_memory.evidence_base
        explanation = self.generate_explanation(final_diagnosis, supporting_evidence)

        return {
            "final_diagnosis": final_diagnosis,
            "explanation": explanation,
            "dialogue_history": [d.dict() for d in self.working_memory.dialogue_history]
        }


# --- 5. Application Entry Point ---

if __name__ == "__main__":
    logger.add("medical_diagnosis_app.log", rotation="1 week", level="INFO")
    logger.info("Application started.")

    # Initialize modules
    working_memory = WorkingMemory()
    knowledge_base = MedicalKnowledgeBase()
    llm_interface = LLMInterface(api_key=os.getenv("OPENAI_API_KEY", "sk-dummy")) # Using dummy key for demo
    diagnosis_policy = DiagnosisPolicy(working_memory, knowledge_base, llm_interface)

    # Simulate a diagnostic session 1
    print("\n======== Simulating Diagnostic Session 1: Fever and Cough ========")
    patient_symptoms_1 = "Patient presents with a persistent cough for 3 days, accompanied by a fever of 101.5°F (38.6°C)."
    result_1 = diagnosis_policy.run_diagnosis(patient_symptoms_1)
    print("\n--- Final Result Session 1 ---")
    print(result_1["explanation"])
    print("\nDialogue History:")
    for entry in result_1["dialogue_history"]:
        print(f"[{entry['role'].capitalize()}]: {entry['content']}")

    # Simulate a diagnostic session 2
    print("\n======== Simulating Diagnostic Session 2: Chest Pain ========")
    patient_symptoms_2 = "Patient reports sudden, severe chest pain radiating to the left arm, along with shortness of breath and sweating."
    result_2 = diagnosis_policy.run_diagnosis(patient_symptoms_2)
    print("\n--- Final Result Session 2 ---")
    print(result_2["explanation"])
    print("\nDialogue History:")
    for entry in result_2["dialogue_history"]:
        print(f"[{entry['role'].capitalize()}]: {entry['content']}")

    # Simulate a diagnostic session 3
    print("\n======== Simulating Diagnostic Session 3: Abdominal Pain ========")
    patient_symptoms_3 = "Patient complains of generalized abdominal pain, mild nausea, no fever or vomiting."
    result_3 = diagnosis_policy.run_diagnosis(patient_symptoms_3)
    print("\n--- Final Result Session 3 ---")
    print(result_3["explanation"])
    print("\nDialogue History:")
    for entry in result_3["dialogue_history"]:
        print(f"[{entry['role'].capitalize()}]: {entry['content']}")

    logger.info("Application finished.")
