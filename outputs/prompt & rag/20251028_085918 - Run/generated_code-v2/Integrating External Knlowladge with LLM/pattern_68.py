
import networkx as nx

class MedicalKnowledgeGraph:
    """
    A simplified Medical Knowledge Graph for demonstration.
    Represents relationships between symptoms, diseases, and treatments.
    """
    def __init__(self):
        self.kg = nx.DiGraph()
        self._build_mock_kg()

    def _build_mock_kg(self):
        # Nodes: Symptoms, Diseases, Treatments
        self.kg.add_node("Fever", type="symptom")
        self.kg.add_node("Cough", type="symptom")
        self.kg.add_node("Fatigue", type="symptom")
        self.kg.add_node("Headache", type="symptom")
        self.kg.add_node("Sore Throat", type="symptom")
        self.kg.add_node("Nausea", type="symptom")
        self.kg.add_node("Rash", type="symptom")
        self.kg.add_node("Chest Pain", type="symptom")

        self.kg.add_node("Common Cold", type="disease")
        self.kg.add_node("Influenza", type="disease")
        self.kg.add_node("Streptococcal Pharyngitis", type="disease") # Strep Throat
        self.kg.add_node("Migraine", type="disease")
        self.kg.add_node("COVID-19", type="disease")
        self.kg.add_node("Bronchitis", type="disease")

        self.kg.add_node("Rest", type="treatment")
        self.kg.add_node("Fluids", type="treatment")
        self.kg.add_node("Antibiotics", type="treatment")
        self.kg.add_node("Pain Relievers", type="treatment")
        self.kg.add_node("Antivirals", type="treatment")
        self.kg.add_node("Steroids", type="treatment")

        # Edges: Relationships (has_symptom, causes, treats, indicates)
        self.kg.add_edge("Common Cold", "Fever", relation="has_symptom")
        self.kg.add_edge("Common Cold", "Cough", relation="has_symptom")
        self.kg.add_edge("Common Cold", "Sore Throat", relation="has_symptom")
        self.kg.add_edge("Common Cold", "Fatigue", relation="has_symptom")
        self.kg.add_edge("Common Cold", "Rest", relation="treatable_by")
        self.kg.add_edge("Common Cold", "Fluids", relation="treatable_by")

        self.kg.add_edge("Influenza", "Fever", relation="has_symptom")
        self.kg.add_edge("Influenza", "Cough", relation="has_symptom")
        self.kg.add_edge("Influenza", "Fatigue", relation="has_symptom")
        self.kg.add_edge("Influenza", "Headache", relation="has_symptom")
        self.kg.add_edge("Influenza", "Sore Throat", relation="has_symptom")
        self.kg.add_edge("Influenza", "Rest", relation="treatable_by")
        self.kg.add_edge("Influenza", "Fluids", relation="treatable_by")
        self.kg.add_edge("Influenza", "Antivirals", relation="treatable_by")

        self.kg.add_edge("Streptococcal Pharyngitis", "Sore Throat", relation="has_symptom")
        self.kg.add_edge("Streptococcal Pharyngitis", "Fever", relation="has_symptom")
        self.kg.add_edge("Streptococcal Pharyngitis", "Nausea", relation="has_symptom")
        self.kg.add_edge("Streptococcal Pharyngitis", "Rash", relation="has_symptom") # Sometimes
        self.kg.add_edge("Streptococcal Pharyngitis", "Antibiotics", relation="treatable_by")

        self.kg.add_edge("Migraine", "Headache", relation="has_symptom")
        self.kg.add_edge("Migraine", "Nausea", relation="has_symptom")
        self.kg.add_edge("Migraine", "Pain Relievers", relation="treatable_by")

        self.kg.add_edge("COVID-19", "Fever", relation="has_symptom")
        self.kg.add_edge("COVID-19", "Cough", relation="has_symptom")
        self.kg.add_edge("COVID-19", "Fatigue", relation="has_symptom")
        self.kg.add_edge("COVID-19", "Sore Throat", relation="has_symptom")
        self.kg.add_edge("COVID-19", "Chest Pain", relation="has_symptom")
        self.kg.add_edge("COVID-19", "Rest", relation="treatable_by")
        self.kg.add_edge("COVID-19", "Fluids", relation="treatable_by")

        self.kg.add_edge("Bronchitis", "Cough", relation="has_symptom")
        self.kg.add_edge("Bronchitis", "Chest Pain", relation="has_symptom")
        self.kg.add_edge("Bronchitis", "Fatigue", relation="has_symptom")
        self.kg.add_edge("Bronchitis", "Rest", relation="treatable_by")
        self.kg.add_edge("Bronchitis", "Fluids", relation="treatable_by")


    def retrieve_knowledge(self, symptoms):
        """
        Retrieves relevant medical facts from the KG based on symptoms.
        Returns a list of formatted knowledge snippets.
        """
        relevant_facts = []
        symptom_nodes = [s for s in symptoms if s in self.kg.nodes and self.kg.nodes[s]["type"] == "symptom"]

        for symptom in symptom_nodes:
            # Find diseases that have this symptom
            for disease in self.kg.predecessors(symptom):
                if self.kg.nodes[disease]["type"] == "disease":
                    relevant_facts.append(f"Fact: Disease '{disease}' has symptom '{symptom}'.")
                    # Also retrieve other symptoms of this disease
                    for _, other_symptom in self.kg.out_edges(disease):
                        if self.kg.nodes[other_symptom]["type"] == "symptom" and other_symptom != symptom:
                            relevant_facts.append(f"Fact: Disease '{disease}' also commonly presents with '{other_symptom}'.")
                    # And treatments for this disease
                    for _, treatment in self.kg.out_edges(disease):
                        if self.kg.nodes[treatment]["type"] == "treatment":
                            relevant_facts.append(f"Fact: Treatment '{treatment}' is recommended for '{disease}'.")

        # Deduplicate facts
        return sorted(list(set(relevant_facts)))


class MedicalDiagnosticAssistant:
    """
    Integrates KG retrieval with an LLM for explainable medical diagnosis.
    """
    def __init__(self, kg, llm_model_name="gpt-3.5-turbo"):
        self.kg = kg
        # In a real application, you would initialize an LLM here
        # self.llm = ChatOpenAI(model=llm_model_name, temperature=0.0)
        # self.parser = StrOutputParser()
        print(f"Initializing MedicalDiagnosticAssistant (using dummy LLM response for demonstration).")

    def _generate_llm_response_dummy(self, prompt_text):
        """
        A dummy function to simulate LLM response for demonstration purposes.
        In a real application, this would be an actual LLM API call.
        """
        print(f"\n--- DUMMY LLM INPUT PROMPT ---\n{prompt_text}\n--- END DUMMY LLM INPUT ---")

        if "Fever" in prompt_text and "Cough" in prompt_text and "Fatigue" in prompt_text:
            if "Headache" in prompt_text:
                diagnosis = "Influenza"
                explanation = (
                    "Based on the symptoms of Fever, Cough, Fatigue, and Headache, and the knowledge that "
                    "Influenza presents with these symptoms, a possible diagnosis is Influenza. "
                    "KG Fact: Disease 'Influenza' has symptom 'Fever'. "
                    "KG Fact: Disease 'Influenza' has symptom 'Cough'. "
                    "KG Fact: Disease 'Influenza' has symptom 'Fatigue'. "
                    "KG Fact: Disease 'Influenza' has symptom 'Headache'. "
                    "KG Fact: Treatment 'Rest' is recommended for 'Influenza'. "
                    "KG Fact: Treatment 'Fluids' is recommended for 'Influenza'."
                )
            elif "Chest Pain" in prompt_text:
                diagnosis = "COVID-19 or Bronchitis"
                explanation = (
                    "Given Fever, Cough, Fatigue, and Chest Pain, both COVID-19 and Bronchitis are possibilities. "
                    "KG Fact: Disease 'COVID-19' has symptom 'Fever'. "
                    "KG Fact: Disease 'COVID-19' has symptom 'Cough'. "
                    "KG Fact: Disease 'COVID-19' has symptom 'Fatigue'. "
                    "KG Fact: Disease 'COVID-19' has symptom 'Chest Pain'. "
                    "KG Fact: Disease 'Bronchitis' has symptom 'Cough'. "
                    "KG Fact: Disease 'Bronchitis' has symptom 'Chest Pain'. "
                    "Further tests would be needed to differentiate."
                )
            else:
                diagnosis = "Common Cold or Early Influenza"
                explanation = (
                    "The combination of Fever, Cough, and Fatigue suggests a viral infection like the Common Cold or early Influenza. "
                    "KG Fact: Disease 'Common Cold' has symptom 'Fever'. "
                    "KG Fact: Disease 'Common Cold' has symptom 'Cough'. "
                    "KG Fact: Disease 'Common Cold' has symptom 'Fatigue'. "
                    "KG Fact: Disease 'Influenza' has symptom 'Fever'. "
                    "KG Fact: Disease 'Influenza' has symptom 'Cough'."
                )
        elif "Sore Throat" in prompt_text and "Fever" in prompt_text and "Nausea" in prompt_text:
            diagnosis = "Streptococcal Pharyngitis"
            explanation = (
                "The presence of Sore Throat, Fever, and Nausea strongly indicates Streptococcal Pharyngitis. "
                "KG Fact: Disease 'Streptococcal Pharyngitis' has symptom 'Sore Throat'. "
                "KG Fact: Disease 'Streptococcal Pharyngitis' has symptom 'Fever'. "
                "KG Fact: Disease 'Streptococcal Pharyngitis' has symptom 'Nausea'. "
                "KG Fact: Treatment 'Antibiotics' is recommended for 'Streptococcal Pharyngitis'."
            )
        elif "Headache" in prompt_text and "Nausea" in prompt_text:
            diagnosis = "Migraine"
            explanation = (
                "Given Headache and Nausea, a Migraine is a strong possibility. "
                "KG Fact: Disease 'Migraine' has symptom 'Headache'. "
                "KG Fact: Disease 'Migraine' has symptom 'Nausea'. "
                "KG Fact: Treatment 'Pain Relievers' is recommended for 'Migraine'."
            )
        else:
            diagnosis = "Undetermined - more information needed"
            explanation = (
                "The provided symptoms are too general or insufficient to make a specific diagnosis based on current knowledge. "
                "More detailed information or additional tests might be required."
            )

        return f"Diagnosis: {diagnosis}\n\nReasoning:\n{explanation}\n\n--- End of LLM Response ---"


    def diagnose(self, patient_symptoms, medical_history="None"):
        """
        Performs diagnosis by retrieving knowledge and guiding the LLM's reasoning.
        """
        print(f"\n--- Starting Diagnosis ---")
        print(f"Patient Symptoms: {', '.join(patient_symptoms)}")
        print(f"Medical History: {medical_history}")

        # Step 1: Retrieve relevant knowledge from the Medical Knowledge Graph
        retrieved_knowledge = self.kg.retrieve_knowledge(patient_symptoms)
        knowledge_context = "\n".join(retrieved_knowledge) if retrieved_knowledge else "No specific knowledge retrieved for these symptoms."

        print(f"\n--- Retrieved Knowledge from KG ---")
        for fact in retrieved_knowledge:
            print(f"- {fact}")
        if not retrieved_knowledge:
            print("- No specific facts found for the given symptoms.")

        # Step 2: Construct the prompt for the LLM
        # This prompt encourages Chain-of-Thought reasoning and grounding in facts.
        prompt_template = """
You are a Medical Diagnostic Assistant. Your task is to analyze patient symptoms and medical history,
and provide a possible diagnosis along with a detailed, step-by-step reasoning process.
Crucially, you must ground your reasoning in the provided medical knowledge facts and explicitly cite them.
Avoid making claims that are not supported by the provided facts.

Patient Symptoms: {symptoms}
Medical History: {history}

Relevant Medical Knowledge Facts (from Knowledge Graph):
{knowledge}

Your Output should be structured as follows:
Diagnosis: [Your most likely diagnosis]
Reasoning:
[Step-by-step explanation, explicitly referencing the 'Relevant Medical Knowledge Facts' provided above.
Each reasoning step should clearly show how it leads to the diagnosis and which fact supports it.]
"""
        prompt = prompt_template.format(
            symptoms=", ".join(patient_symptoms),
            history=medical_history,
            knowledge=knowledge_context
        )

        # Step 3: Get response from the LLM (using dummy for demonstration)
        # In a real scenario, you would use:
        # chain = prompt_template | self.llm | self.parser
        # llm_response = chain.invoke({"symptoms": ", ".join(patient_symptoms), "history": medical_history, "knowledge": knowledge_context})
        llm_response = self._generate_llm_response_dummy(prompt)

        # Step 4: Parse and return the diagnosis and reasoning
        diagnosis_line = ""
        reasoning_start = llm_response.find("Reasoning:")
        if reasoning_start != -1:
            diagnosis_line = llm_response[:reasoning_start].strip()
            reasoning_content = llm_response[reasoning_start + len("Reasoning:"):].strip()
        else:
            diagnosis_line = "Diagnosis: Error parsing LLM response."
            reasoning_content = llm_response

        # Clean up dummy markers if present
        diagnosis_line = diagnosis_line.replace("--- End of LLM Response ---", "").strip()
        reasoning_content = reasoning_content.replace("--- End of LLM Response ---", "").strip()


        return {
            "diagnosis": diagnosis_line,
            "reasoning": reasoning_content
        }


# --- Main execution flow for demonstration ---
if __name__ == "__main__":
    # 1. Initialize the Medical Knowledge Graph
    medical_kg = MedicalKnowledgeGraph()

    # 2. Initialize the Medical Diagnostic Assistant
    assistant = MedicalDiagnosticAssistant(medical_kg)

    # 3. Define patient cases
    patient_cases = [
        {
            "symptoms": ["Fever", "Cough", "Fatigue", "Headache"],
            "history": "No significant medical history."
        },
        {
            "symptoms": ["Sore Throat", "Fever", "Nausea"],
            "history": "No known allergies."
        },
        {
            "symptoms": ["Headache", "Nausea"],
            "history": "History of occasional severe headaches."
        },
        {
            "symptoms": ["Fever", "Cough", "Chest Pain"],
            "history": "Smokes occasionally."
        },
        {
            "symptoms": ["Runny Nose", "Sneezing"], # Symptoms not heavily represented in our mock KG
            "history": "No medical history."
        },
    ]

    # 4. Process patient cases and display results
    for i, case in enumerate(patient_cases):
        print(f"\n======== Patient Case {i+1} ========")
        result = assistant.diagnose(case["symptoms"], case["history"])
        print(f"\nProposed {result['diagnosis']}")
        print(f"\nDetailed {result['reasoning']}")
        print(f"\n=================================\n")
