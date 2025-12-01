from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool


class DiseaseSearchInput(BaseModel):
    disease_name: str = Field(description="The name of the disease to search for.")


class MedicalKnowledgeBaseTool:
    def run(self, disease_name: str) -> str:
        medical_data = {
            "common cold": "A viral infection of the nose and throat. Symptoms include runny nose, sore throat, cough, congestion. Treatment is supportive care.",
            "influenza": "A viral infection that attacks the respiratory system. Symptoms include fever, body aches, headache, cough, fatigue. Treatment involves antivirals or supportive care.",
            "strep throat": "A bacterial infection of the throat and tonsils. Symptoms include sore throat, difficulty swallowing, fever. Treatment is antibiotics.",
            "pneumonia": "An infection that inflames air sacs in one or both lungs. Symptoms include cough with phlegm, fever, chills, difficulty breathing. Treatment involves antibiotics or antivirals."
        }
        return medical_data.get(disease_name.lower(), "Disease not found in knowledge base.")


class ImagingAnalysisInput(BaseModel):
    image_description: str = Field(description="A description of the medical image (e.g., 'chest X-ray showing lungs', 'MRI of knee').")


class DiagnosticImagingAnalysisTool:
    def run(self, image_description: str) -> str:
        if "chest x-ray showing lungs" in image_description.lower():
            return "Simulated report: Chest X-ray indicates clear lung fields with no signs of acute pathology."
        elif "mri of knee" in image_description.lower() and "ligament tear" in image_description.lower():
            return "Simulated report: MRI of the knee suggests a partial tear of the ACL."
        return f"Simulated report: Analysis for '{image_description}' is inconclusive or not specifically configured."


class LabTestInput(BaseModel):
    test_name: str = Field(description="The name of the lab test (e.g., 'blood glucose', 'CBC').")
    test_value: Optional[float] = Field(None, description="The numerical result of the lab test, if applicable.")


class LabTestInterpretationTool:
    def run(self, test_name: str, test_value: Optional[float] = None) -> str:
        if test_name.lower() == "blood glucose":
            if test_value is None:
                return "Please provide a blood glucose value for interpretation."
            if test_value < 70:
                return "Simulated interpretation: Blood glucose is low (hypoglycemia)."
            elif 70 <= test_value <= 99:
                return "Simulated interpretation: Blood glucose is within normal range."
            else:
                return "Simulated interpretation: Blood glucose is elevated (hyperglycemia)."
        elif test_name.lower() == "cbc":
            return "Simulated interpretation: Complete Blood Count (CBC) results show no significant abnormalities, indicating general good health."
        return f"Simulated interpretation: Lab test '{test_name}' is not specifically configured or value '{test_value}' unrecognized."


class DrugInteractionInput(BaseModel):
    drugs: List[str] = Field(description="A list of drug names to check for interactions.")


class DrugInteractionCheckerTool:
    def run(self, drugs: List[str]) -> str:
        drugs_lower = [d.lower() for d in drugs]
        if "aspirin" in drugs_lower and "warfarin" in drugs_lower:
            return "Severe interaction: Increased risk of bleeding with Aspirin and Warfarin."
        elif "ibuprofen" in drugs_lower and "lisinopril" in drugs_lower:
            return "Moderate interaction: NSAIDs like Ibuprofen can reduce the effectiveness of ACE inhibitors like Lisinopril and increase kidney risk."
        elif len(drugs) > 1:
            return f"Simulated interaction check: No significant interactions found between {', '.join(drugs)}."
        return "Please provide at least two drugs to check for interactions."


medical_kb_tool = Tool(
    name="MedicalKnowledgeBase",
    func=MedicalKnowledgeBaseTool().run,
    description="Useful for looking up information about diseases, symptoms, causes, and standard treatments.",
    args_schema=DiseaseSearchInput,
)

imaging_analysis_tool = Tool(
    name="DiagnosticImagingAnalysis",
    func=DiagnosticImagingAnalysisTool().run,
    description="Useful for simulating the analysis of medical images (e.g., X-rays, MRIs) and getting a diagnostic report.",
    args_schema=ImagingAnalysisInput,
)

lab_test_tool = Tool(
    name="LabTestInterpretation",
    func=LabTestInterpretationTool().run,
    description="Useful for interpreting numerical lab test results (e.g., blood glucose, CBC) and getting an interpretation.",
    args_schema=LabTestInput,
)

drug_interaction_tool = Tool(
    name="DrugInteractionChecker",
    func=DrugInteractionCheckerTool().run,
    description="Useful for checking potential interactions between a list of prescribed drugs.",
    args_schema=DrugInteractionInput,
)


llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")

tools = [
    medical_kb_tool,
    imaging_analysis_tool,
    lab_test_tool,
    drug_interaction_tool,
]


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)


def main():
    print("Welcome to the AI Medical Diagnosis and Treatment Recommendation System!")
    print("How can I help you today? (Type 'exit' to quit)")

    while True:
        patient_query = input("\nEnter patient symptoms and medical history: ")
        if patient_query.lower() == 'exit':
            break

        try:
            response = agent.run(patient_query)
            print("\n--- Diagnosis and Recommendation ---")
            print(response)
            print("------------------------------------")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try rephrasing your query or check the environment setup (e.g., OPENAI_API_KEY).")


if __name__ == "__main__":
    main()