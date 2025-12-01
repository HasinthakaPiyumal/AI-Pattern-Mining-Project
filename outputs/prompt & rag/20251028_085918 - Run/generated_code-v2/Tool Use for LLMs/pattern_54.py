from pydantic import BaseModel, Field
from typing import Any, List, Dict, Union, Iterable

from langchain.tools import Tool
from langchain.agents import AgentExecutor, AgentType, LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import BasePromptTemplate, PromptTemplate

# --- 1. Mock LLM Implementation ---
class MockLLM(BaseLLM):
    response_map: Dict[str, str]

    def __init__(self, response_map: Dict[str, str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.response_map = response_map if response_map is not None else {}

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        # Simple rule-based response for demonstration
        if "medical journal" in prompt.lower() and "hypertension" in prompt.lower():
            return "Action: MedicalJournalSearch\nAction Input: {'query': 'latest guidelines hypertension'}"
        elif "calculate dosage" in prompt.lower():
            return "Action: DrugDosageCalculator\nAction Input: {'drug_name': 'Amlodipine', 'patient_weight_kg': 70, 'dose_per_kg': 0.05}"
        elif "patient history" in prompt.lower() or "lab results" in prompt.lower():
            return "Action: EHRQuery\nAction Input: {'patient_id': 'P12345'}"
        elif "ICD code" in prompt.lower() or "drug interaction" in prompt.lower():
            return "Action: ICDDrugInteractionLookup\nAction Input: {'diagnosis_keywords': 'Type 2 Diabetes', 'medications': ['Metformin', 'Simvastatin']}"
        elif "Final Answer:" in prompt:
            return prompt.split("Final Answer:", 1)[1].strip()
        else:
            # Default behavior for other prompts, trying to simulate a simple thought process
            if "Thought:" not in prompt:
                 return "Thought: I need to determine the best tool to answer the user's request. " \
                        "If I cannot find a specific tool, I will try to answer based on general knowledge or state I need more information.\n" \
                        f"Observation: User asked: {prompt}\n" \
                        "Thought: It seems like I can directly answer this without a specific tool, or I might be missing a tool for this. For now, I will provide a general response.\n" \
                        f"Final Answer: I understand your query about '{prompt}'. To provide a precise answer, could you specify what kind of medical information or task you need?"
            else:
                return "Thought: I have evaluated the input and previous observations. It seems I have processed the request or reached a point where I can formulate a final answer based on the available information.\nFinal Answer: Based on my understanding and the tools available, I can provide information relevant to your medical query. Please refine your request if you need specific details."


    @property
    def _llm_type(self) -> str:
        return "mock_llm"

# --- 2. Tool Input Schemas (Pydantic) ---
class MedicalJournalSearchInput(BaseModel):
    query: str = Field(description="The medical query to search for in journal databases.")

class DrugDosageCalculatorInput(BaseModel):
    drug_name: str = Field(description="The name of the drug.")
    patient_weight_kg: float = Field(description="Patient's weight in kilograms.")
    dose_per_kg: float = Field(description="Recommended dose per kilogram of body weight.")

class EHRQueryInput(BaseModel):
    patient_id: str = Field(description="The unique identifier for the patient.")

class ICDDrugInteractionLookupInput(BaseModel):
    diagnosis_keywords: str = Field(description="Keywords describing the diagnosis (e.g., 'Type 2 Diabetes').")
    medications: List[str] = Field(description="List of current medications for drug interaction check.")

# --- 3. Mock Tool Implementations ---
def medical_journal_search(query: str) -> str:
    if "hypertension" in query.lower():
        return (
            "Found latest guidelines for hypertension management (2023 update):\n" \
            "- Lifestyle modifications (diet, exercise)\n" \
            "- First-line medications: ACE inhibitors, ARBs, Thiazide diuretics, CCBs.\n" \
            "- Personalized treatment based on comorbidities."
        )
    elif "diabetes" in query.lower():
        return "Found recent articles on Type 2 Diabetes treatment: Metformin remains first-line, SGLT2 inhibitors and GLP-1 agonists for cardiovascular benefits."
    return f"No specific recent articles found for '{query}'."

def drug_dosage_calculator(drug_name: str, patient_weight_kg: float, dose_per_kg: float) -> str:
    calculated_dose = patient_weight_kg * dose_per_kg
    return f"Calculated dosage for {drug_name}: {calculated_dose:.2f} mg (for patient weight {patient_weight_kg} kg at {dose_per_kg} mg/kg)."

def ehr_query(patient_id: str) -> str:
    if patient_id == "P12345":
        return (
            "Patient ID: P12345\n" \
            "Name: Jane Doe\n" \
            "DOB: 1970-05-15\n" \
            "Diagnoses: Type 2 Diabetes, Hypertension\n" \
            "Medications: Metformin 500mg BID, Amlodipine 5mg QD, Simvastatin 20mg QD\n" \
            "Lab Results (last 3 months): HbA1c 7.2%, BP 138/85 mmHg, LDL 105 mg/dL."
        )
    return f"Patient with ID '{patient_id}' not found in EHR."

def icd_drug_interaction_lookup(diagnosis_keywords: str, medications: List[str]) -> str:
    response = f"ICD-10 codes for '{diagnosis_keywords}':\n"
    if "type 2 diabetes" in diagnosis_keywords.lower():
        response += " - E11.9 (Type 2 diabetes mellitus without complications)\n"
    if "hypertension" in diagnosis_keywords.lower():
        response += " - I10 (Essential (primary) hypertension)\n"
    
    response += "\nDrug Interactions for current medications:\n"
    if "metformin" in [m.lower() for m in medications] and "simvastatin" in [m.lower() for m in medications]:
        response += " - No significant direct interaction between Metformin and Simvastatin commonly noted.\n"
    elif "amlodipine" in [m.lower() for m in medications] and "simvastatin" in [m.lower() for m in medications]:
        response += " - Co-administration of Amlodipine and Simvastatin may increase Simvastatin levels. Consider lower Simvastatin dose.\n"
    else:
        response += " - No specific interactions found for the provided combination (mock data).\n"
    return response

# --- 4. Create langchain.Tool instances ---
tools = [
    Tool(
        name="MedicalJournalSearch",
        func=medical_journal_search,
        description=(
            "Useful for searching up-to-date medical journal articles and clinical guidelines "
            "for evidence-based information. Input should be a specific medical query."
        ),
        args_schema=MedicalJournalSearchInput,
    ),
    Tool(
        name="DrugDosageCalculator",
        func=drug_dosage_calculator,
        description=(
            "Useful for calculating precise drug dosages based on patient weight and recommended dose per kg. "
            "Input requires drug_name, patient_weight_kg, and dose_per_kg."
        ),
        args_schema=DrugDosageCalculatorInput,
    ),
    Tool(
        name="EHRQuery",
        func=ehr_query,
        description=(
            "Useful for retrieving patient's electronic health records, including history, "
            "diagnoses, medications, and lab results. Input requires patient_id."
        ),
        args_schema=EHRQueryInput,
    ),
    Tool(
        name="ICDDrugInteractionLookup",
        func=icd_drug_interaction_lookup,
        description=(
            "Useful for looking up ICD-10/11 codes for diagnoses and checking for "
            "potential drug-drug interactions. Input requires diagnosis_keywords and a list of medications."
        ),
        args_schema=ICDDrugInteractionLookupInput,
    ),
]

# --- 5. Initialize Mock LLM ---
# The response_map in MockLLM is simplified and will primarily use the logic in _call
# for determining which tool to call based on keywords.
llm = MockLLM()

# --- 6. Create Langchain Agent ---
# Using AgentType.ZERO_SHOT_REACT_DESCRIPTION for simplicity
agent = AgentExecutor.from_agent_and_tools(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, # Set to True to see the thought process of the agent
    handle_parsing_errors=True,
)

# --- 7. Simple Command-Line Interface ---
def main():
    print("Welcome to the Medical Diagnosis and Treatment Recommendation System!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        user_query = input("\nHealthcare Professional: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting system. Goodbye!")
            break
        try:
            # The agent's __call__ method takes an input dictionary
            response = agent.run(user_query)
            print(f"System: {response}")
        except ValueError as e:
            print(f"Error processing request: {e}")
            print("Please try rephrasing your query or providing more specific details.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
