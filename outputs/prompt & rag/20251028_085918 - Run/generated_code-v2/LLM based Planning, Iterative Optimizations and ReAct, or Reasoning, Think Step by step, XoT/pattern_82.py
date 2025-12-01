from typing import Dict, Any
import json

class MockLLM:
    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if "initial diagnosis" in prompt.lower():
            return "Initial diagnosis: Viral pharyngitis. Treatment: Rest, fluids, symptomatic relief. Potential concern: Could be bacterial, consider rapid strep test if symptoms worsen."
        elif "critique" in prompt.lower():
            if "viral pharyngitis" in prompt.lower():
                return "Critique: The initial diagnosis of viral pharyngitis seems plausible, but the suggestion to consider a rapid strep test for bacterial infection is crucial. Are there specific symptoms that strongly suggest strep throat, like tonsillar exudates or swollen lymph nodes? Also, verify if symptomatic relief includes over-the-counter pain relievers and throat lozenges. Check for drug interactions if the patient is on other medications. Could there be any other less common but severe differential diagnoses to rule out based on the presented symptoms?"
            else:
                return "Critique: The initial response is too vague. Needs more specific details and consideration of external verification."
        elif "refine" in prompt.lower():
            if "bacterial infection" in prompt.lower() and "rapid strep test" in prompt.lower():
                return "Refined Diagnosis: Likely Viral Pharyngitis. Treatment: Rest, hydration, acetaminophen/ibuprofen for pain, throat lozenges. Recommendation: If symptoms (especially fever >101F, severe sore throat, purulent tonsillar exudates) persist beyond 48 hours or worsen, perform a rapid strep test to rule out Group A Streptococcus. Consult clinical guidelines for atypical presentations. No significant drug interactions identified with common OTCs."
            else:
                return "Refined Diagnosis: Could not refine effectively with provided information. More data needed."
        return "Mock LLM response."

class MockInternetSearchTool:
    def run(self, query: str) -> str:
        if "strep throat symptoms" in query.lower():
            return "Search Result: Common strep throat symptoms include sudden sore throat, pain when swallowing, fever, red and swollen tonsils sometimes with white patches or streaks of pus, tiny red spots on the soft or hard palate (petechiae), headache, nausea, vomiting. Rarely, a fine, sandpaper-like rash (scarlet fever) may be present."
        elif "viral pharyngitis symptoms" in query.lower():
            return "Search Result: Viral pharyngitis typically presents with a sore throat, runny nose, cough, hoarseness, conjunctivitis, and sometimes a low-grade fever. It's often associated with common cold viruses."
        return f"Search result for '{query}': No specific medical information found in mock database."

class MockMedicalKnowledgeBaseTool:
    def run(self, query: str) -> str:
        if "acetaminophen ibuprofen interaction" in query.lower():
            return "KB Result: Acetaminophen and Ibuprofen can generally be taken together, but caution should be exercised regarding total daily dosage of each to avoid toxicity, especially liver damage with acetaminophen and kidney issues/GI bleed with NSAIDs like ibuprofen. No direct drug-drug interaction preventing co-administration at therapeutic doses."
        elif "amoxicillin dosage strep" in query.lower():
            return "KB Result: Amoxicillin for strep throat (adults): 500 mg 2-3 times a day for 10 days, or 875 mg twice a day for 10 days. Pediatric dosages are weight-based."
        return f"KB result for '{query}': No specific drug or interaction information found in mock database."

class MockClinicalGuidelinesTool:
    def run(self, query: str) -> str:
        if "strep throat management guidelines" in query.lower():
            return "Guidelines Result: Clinical guidelines for Group A Streptococcal Pharyngitis recommend penicillin or amoxicillin as first-line treatment for 10 days to prevent acute rheumatic fever. Rapid antigen detection tests (RADTs) or throat culture are recommended for diagnosis. Symptomatic management includes analgesics and antipyretics."
        elif "viral pharyngitis management" in query.lower():
            return "Guidelines Result: Management for viral pharyngitis is primarily supportive, focusing on symptomatic relief. This includes rest, hydration, analgesics (e.g., acetaminophen, ibuprofen), and throat lozenges. Antibiotics are ineffective and not recommended."
        return f"Guidelines result for '{query}': No specific guidelines found in mock database."

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.llm = MockLLM()
        self.internet_search_tool = MockInternetSearchTool()
        self.medical_kb_tool = MockMedicalKnowledgeBaseTool()
        self.clinical_guidelines_tool = MockClinicalGuidelinesTool()

    def diagnose(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        symptoms = patient_data.get("symptoms", "")
        medical_history = patient_data.get("medical_history", "")
        test_results = patient_data.get("test_results", "")

        initial_prompt = f"Patient presents with: Symptoms: {symptoms}, Medical History: {medical_history}, Test Results: {test_results}. Provide an initial diagnosis and treatment plan."
        initial_response = self.llm.generate(initial_prompt)

        critique_prompt = f"Critique the following diagnosis and treatment plan for potential errors or areas of uncertainty: {initial_response}. Consider differential diagnoses, drug interactions, and evidence-based practices."
        critique_response = self.llm.generate(critique_prompt)

        tool_results = []
        if "strep throat" in critique_response.lower() or "bacterial infection" in critique_response.lower() or "rapid strep test" in critique_response.lower():
            tool_results.append(self.internet_search_tool.run("strep throat symptoms"))
            tool_results.append(self.clinical_guidelines_tool.run("strep throat management guidelines"))
        if "drug interactions" in critique_response.lower() or "medications" in critique_response.lower():
            tool_results.append(self.medical_kb_tool.run("acetaminophen ibuprofen interaction"))

        refined_prompt = f"Refine the initial diagnosis and treatment plan based on the following critique and tool results. Initial Response: {initial_response}\nCritique: {critique_response}\nTool Results: {'\n'.join(tool_results) if tool_results else 'No additional tool results.'}. Provide a final, evidence-based diagnosis and treatment plan."
        final_response = self.llm.generate(refined_prompt)

        return {
            "initial_diagnosis": initial_response,
            "critique": critique_response,
            "tool_information": tool_results,
            "final_diagnosis": final_response,
            "confidence_score": "High (due to self-correction)"
        }

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    patient_data_1 = {
        "symptoms": "Sore throat, mild fever, runny nose, cough",
        "medical_history": "No significant medical history",
        "test_results": "None"
    }

    print("\n--- Patient 1 Diagnosis ---")
    diagnosis_output_1 = assistant.diagnose(patient_data_1)
    print(json.dumps(diagnosis_output_1, indent=2))

    patient_data_2 = {
        "symptoms": "Sudden severe sore throat, high fever, swollen tonsils with white patches, headache",
        "medical_history": "No known allergies",
        "test_results": "None"
    }

    print("\n--- Patient 2 Diagnosis ---")
    diagnosis_output_2 = assistant.diagnose(patient_data_2)
    print(json.dumps(diagnosis_output_2, indent=2))