from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


class MedicalDiagnosticAssistant:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.5)
        self.output_parser = StrOutputParser()
        self.medical_context = ""

        self.thot_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a highly intelligent medical diagnostic assistant. Your task is to analyze medical information using a 'Thread of Thought' approach. Walk through the provided context and patient information step-by-step, summarizing and analyzing as you go to arrive at a differential diagnosis and treatment recommendations."),
                ("human", "Patient Information:\n{patient_info}\n\nMedical Context:\n{medical_context}\n\nWalk me through this context in manageable parts step by step, summarizing and analyzing as we go to provide a differential diagnosis and treatment recommendations.")
            ]
        )

        self.chain = self.thot_prompt_template | self.llm | self.output_parser

    def ingest_medical_data(self, data: str):
        self.medical_context = data
        print("Medical data ingested (simulated).")

    def diagnose_patient(self, patient_info: str) -> str:
        print("Analyzing patient information with Thread-of-Thought...")
        response = self.chain.invoke({"patient_info": patient_info, "medical_context": self.medical_context})
        return response


if __name__ == "__main__":
    # In a real application, you would load your OPENAI_API_KEY securely
    # For demonstration, replace "YOUR_OPENAI_API_KEY" with your actual key or use environment variables.
    # For example: import os; openai_api_key = os.getenv("OPENAI_API_KEY")
    assistant = MedicalDiagnosticAssistant(openai_api_key="YOUR_OPENAI_API_KEY")

    # Simulate ingesting complex medical data (e.g., from a document loader, text splitter, vector store)
    # This data would typically be retrieved based on patient symptoms in a real RAG setup.
    simulated_context = """
    Medical Guideline for Diabetes Type 2:
    Symptoms often include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Risk factors include obesity, physical inactivity, family history, age, and certain ethnicities. Diagnosis involves blood tests like HbA1c (>=6.5%), Fasting Plasma Glucose (>=126 mg/dL), or Oral Glucose Tolerance Test (>=200 mg/dL).
    Treatment typically involves lifestyle modifications (diet, exercise), metformin as a first-line drug, and potentially other medications like sulfonylureas, GLP-1 receptor agonists, SGLT2 inhibitors, or insulin.

    Patient Case Study - Mr. John Doe (fictional):
    Age: 55, Gender: Male, Weight: 220 lbs, Height: 5'10".
    Presenting complaints: Increased thirst for 3 months, frequent urination (especially at night), general fatigue. No unexplained weight loss reported. History of hypertension, controlled with medication. Family history positive for Type 2 Diabetes (father).
    Initial lab results (fictional): Fasting Plasma Glucose: 180 mg/dL, HbA1c: 8.2%.
    """
    assistant.ingest_medical_data(simulated_context)

    patient_description = "Patient is a 55-year-old male with increased thirst, frequent urination, and fatigue. He has a history of hypertension and a family history of Type 2 Diabetes. Recent fasting glucose is 180 mg/dL and HbA1c is 8.2%."

    diagnosis_and_recommendations = assistant.diagnose_patient(patient_description)
    print("\n--- Differential Diagnosis and Treatment Recommendations ---")
    print(diagnosis_and_recommendations)
