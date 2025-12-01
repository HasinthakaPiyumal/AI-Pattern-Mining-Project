import random

class Tool:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def run(self, *args, **kwargs):
        raise NotImplementedError

class MedicalKnowledgeBaseTool(Tool):
    def __init__(self):
        super().__init__("MedicalKnowledgeBaseTool", "Queries a medical knowledge base for disease information based on symptoms or keywords.")
        self.mock_data = {
            "fatigue, joint pain": {"Lupus": "Autoimmune disease affecting joints, skin, kidneys. Common markers: positive ANA, high ESR.", "Rheumatoid Arthritis": "Chronic inflammatory disorder affecting joints. Common markers: high ESR, positive RF."}, 
            "muscle weakness, difficulty swallowing": {"Myasthenia Gravis": "Neuromuscular disease causing muscle weakness. Common markers: positive AChR antibodies."}, 
            "skin rash, fever": {"Psoriasis": "Chronic skin condition. Not typically fever related.", "Lyme Disease": "Bacterial infection spread by ticks. Can cause rash, fever, joint pain."}
        }

    def run(self, symptom_keywords):
        symptoms_str = ", ".join(sorted(symptom_keywords))
        result = self.mock_data.get(symptoms_str, {"Unknown": "No direct match in knowledge base for these symptoms."})
        return result

class ResearchPaperSearchTool(Tool):
    def __init__(self):
        super().__init__("ResearchPaperSearchTool", "Searches for relevant research papers based on potential diagnoses or specific patient markers.")
        self.mock_papers = {
            "Lupus diagnostic markers": ["Recent advances in ANA testing for SLE", "Genetic predispositions to Systemic Lupus Erythematosus"],
            "Rheumatoid Arthritis treatment": ["Biologics in RA management", "Early intervention in rheumatoid arthritis improves outcomes"],
            "Myasthenia Gravis antibodies": ["Acetylcholine receptor antibody detection methods", "MuSK antibody positive Myasthenia Gravis"],
            "rare autoimmune disease genetic links": ["Novel genetic markers for orphan autoimmune conditions", "The role of HLA in rare autoimmune disorders"]
        }

    def run(self, query):
        result = self.mock_papers.get(query, [f"No specific papers found for '{query}'."])
        return result

class PatientDataAnalysisTool(Tool):
    def __init__(self):
        super().__init__("PatientDataAnalysisTool", "Analyzes patient's structured data (e.g., lab results, genetic markers) to find patterns or confirm suspicions.")

    def run(self, patient_data, suspected_diseases=None):
        findings = []
        if "ANA_positive" in patient_data and patient_data["ANA_positive"]:
            findings.append("Patient is ANA positive, which is common in autoimmune diseases like Lupus.")
        if "ESR_high" in patient_data and patient_data["ESR_high"]:
            findings.append("Patient has high ESR, indicating inflammation, often seen in autoimmune conditions.")
        if "AChR_antibodies_positive" in patient_data and patient_data["AChR_antibodies_positive"]:
            findings.append("Patient has positive AChR antibodies, strongly suggesting Myasthenia Gravis.")
        if "genetic_marker_X_present" in patient_data and patient_data["genetic_marker_X_present"]:
            findings.append("Genetic marker X detected, potentially linked to a rare neurological disorder.")
        
        if not findings and suspected_diseases:
            findings.append(f"No definitive markers found in patient data to confirm/deny {', '.join(suspected_diseases)}.")
        elif not findings and not suspected_diseases:
            findings.append("No significant abnormal markers found in provided patient data.")

        return " ".join(findings)

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register_tool(self, tool):
        self._tools[tool.name] = tool

    def get_tool(self, name):
        return self._tools.get(name)

class TransparentDiagnosticAssistant:
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry
        self.reasoning_history = []

    def _log_reasoning(self, step_type, details):
        self.reasoning_history.append({"step_type": step_type, "details": details})

    def diagnose(self, symptoms, patient_data):
        self.reasoning_history = []  
        self._log_reasoning("Initial Request", {"symptoms": symptoms, "patient_data": patient_data})

        # Step 1: Initial assessment with Medical Knowledge Base
        self._log_reasoning("LLM's Rationale", "Starting with a broad search in the medical knowledge base based on patient symptoms to identify potential diseases.")
        medical_kb_tool = self.tool_registry.get_tool("MedicalKnowledgeBaseTool")
        symptom_keywords = sorted(list(set(symptoms)))
        self._log_reasoning("Tool Call", {"tool": medical_kb_tool.name, "parameters": {"symptom_keywords": symptom_keywords}})
        kb_result = medical_kb_tool.run(symptom_keywords)
        self._log_reasoning("Tool Result", {"output": kb_result})
        
        potential_diagnoses = list(kb_result.keys())
        if "Unknown" in potential_diagnoses:
            self._log_reasoning("Intermediate Hypothesis", "No direct disease match found based on symptoms. Will proceed to broader research and patient data analysis.")
            potential_diagnoses = []
        else:
            self._log_reasoning("Intermediate Hypothesis", f"Initial potential diagnoses identified: {', '.join(potential_diagnoses)}.")

        # Step 2: Refinement with Research Papers if potential diagnoses exist or a broad search is needed
        research_query = "rare autoimmune disease genetic links" 
        if potential_diagnoses:
            research_query = f"{potential_diagnoses[0]} diagnostic markers" 

        self._log_reasoning("LLM's Rationale", f"Searching for research papers to gather more specific information on {research_query}.")
        research_tool = self.tool_registry.get_tool("ResearchPaperSearchTool")
        self._log_reasoning("Tool Call", {"tool": research_tool.name, "parameters": {"query": research_query}})
        research_result = research_tool.run(research_query)
        self._log_reasoning("Tool Result", {"output": research_result})
        self._log_reasoning("Intermediate Hypothesis", f"Gathered relevant research papers related to {research_query}. Findings: {', '.join(research_result)}.")

        # Step 3: Patient Data Correlation
        self._log_reasoning("LLM's Rationale", "Analyzing patient-specific data to find corroborating evidence or rule out potential diagnoses.")
        patient_data_tool = self.tool_registry.get_tool("PatientDataAnalysisTool")
        self._log_reasoning("Tool Call", {"tool": patient_data_tool.name, "parameters": {"patient_data": patient_data, "suspected_diseases": potential_diagnoses}})
        patient_analysis_result = patient_data_tool.run(patient_data, potential_diagnoses)
        self._log_reasoning("Tool Result", {"output": patient_analysis_result})
        self._log_reasoning("Intermediate Hypothesis", f"Patient data analysis yielded: {patient_analysis_result}.")

        # Final Diagnosis Synthesis
        final_diagnosis = "Undetermined Rare Disease (requires further investigation)"
        confidence_score = 50.0
        justification_parts = []

        if "Myasthenia Gravis" in potential_diagnoses and "positive AChR antibodies" in patient_analysis_result:
            final_diagnosis = "Myasthenia Gravis"
            confidence_score = 95.0
            justification_parts.append("High confidence due to consistent symptoms and positive AChR antibodies in patient data.")
        elif "Lupus" in potential_diagnoses and "ANA positive" in patient_analysis_result and "high ESR" in patient_analysis_result:
            final_diagnosis = "Systemic Lupus Erythematosus (Lupus)"
            confidence_score = 90.0
            justification_parts.append("Strong indication of Lupus based on symptoms, positive ANA, and high ESR.")
        elif potential_diagnoses and "No definitive markers" not in patient_analysis_result and "No significant abnormal markers" not in patient_analysis_result:
            final_diagnosis = f"Probable {potential_diagnoses[0]} (further validation needed)"
            confidence_score = 75.0
            justification_parts.append(f"Symptoms align with {potential_diagnoses[0]}, and patient data provided some supporting observations.")
        else:
            justification_parts.append("Initial symptoms led to some potential rare diseases, but patient data did not provide sufficient confirmation. Further specialized tests are recommended.")
        
        final_justification = " ".join(justification_parts) + "\n\nFull Reasoning History:\n" + "\n".join([f"- {step['step_type']}: {step['details']}" for step in self.reasoning_history])

        return {"diagnosis": final_diagnosis, "confidence": confidence_score, "justification": final_justification, "reasoning_history": self.reasoning_history}

if __name__ == "__main__":
    # Setup Tool Registry and register tools
    tool_registry = ToolRegistry()
    tool_registry.register_tool(MedicalKnowledgeBaseTool())
    tool_registry.register_tool(ResearchPaperSearchTool())
    tool_registry.register_tool(PatientDataAnalysisTool())

    # Initialize the Diagnostic Assistant
    assistant = TransparentDiagnosticAssistant(tool_registry)

    # Scenario 1: Clear case of Myasthenia Gravis
    print("\n--- Scenario 1: Clear case of Myasthenia Gravis ---")
    symptoms_1 = ["muscle weakness", "difficulty swallowing"]
    patient_data_1 = {"ANA_positive": False, "ESR_high": False, "AChR_antibodies_positive": True}
    result_1 = assistant.diagnose(symptoms_1, patient_data_1)
    print(f"Diagnosis: {result_1['diagnosis']}")
    print(f"Confidence: {result_1['confidence']:.2f}%")
    print(f"Justification: {result_1['justification']}")

    # Scenario 2: Suspected Lupus with supporting data
    print("\n--- Scenario 2: Suspected Lupus with supporting data ---")
    symptoms_2 = ["fatigue", "joint pain", "skin rash"]
    patient_data_2 = {"ANA_positive": True, "ESR_high": True, "AChR_antibodies_positive": False}
    result_2 = assistant.diagnose(symptoms_2, patient_data_2)
    print(f"Diagnosis: {result_2['diagnosis']}")
    print(f"Confidence: {result_2['confidence']:.2f}%")
    print(f"Justification: {result_2['justification']}")

    # Scenario 3: Less clear case, requiring more investigation
    print("\n--- Scenario 3: Less clear case ---")
    symptoms_3 = ["unexplained weight loss", "intermittent fever"]
    patient_data_3 = {"ANA_positive": False, "ESR_high": False}
    result_3 = assistant.diagnose(symptoms_3, patient_data_3)
    print(f"Diagnosis: {result_3['diagnosis']}")
    print(f"Confidence: {result_3['confidence']:.2f}%")
    print(f"Justification: {result_3['justification']}")

    # Scenario 4: A less common combination, triggers generic research
    print("\n--- Scenario 4: Generic Research Trigger ---")
    symptoms_4 = ["muscle weakness", "skin rash"]
    patient_data_4 = {"genetic_marker_X_present": True}
    result_4 = assistant.diagnose(symptoms_4, patient_data_4)
    print(f"Diagnosis: {result_4['diagnosis']}")
    print(f"Confidence: {result_4['confidence']:.2f}%")
    print(f"Justification: {result_4['justification']}")
