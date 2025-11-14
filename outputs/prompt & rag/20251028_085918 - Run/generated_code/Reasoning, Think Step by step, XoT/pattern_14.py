from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
import os

# --- Configuration and Setup ---

# Placeholder for LLM - replace with your actual API key or model initialization
# For demonstration, we'll use a mock if OPENAI_API_KEY is not set.
# In a real application, you would securely load this.
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
llm = ChatOpenAI(model="gpt-4", temperature=0.5) # You can use a specific model or a mock

# Embedding model for RAG
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Sample Medical Knowledge Base (In-memory ChromaDB for demonstration)
medical_docs = [
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. It is commonly caused by bacteria or viruses.",
    "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness, and at times can lead to death. Symptoms include fever, cough, sore throat, body aches, headache, and fatigue.",
    "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out, and shortness of breath.",
    "Myocardial infarction (heart attack) occurs when blood flow to a part of your heart is blocked for a long enough time, that part of the heart muscle is damaged or dies. Symptoms include chest pain, shortness of breath, pain in the left arm, and lightheadedness.",
    "Diabetes mellitus is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. High blood glucose is a common effect of uncontrolled diabetes and over time leads to serious damage to many of the body's systems.",
    "Migraine is a headache of varying intensity, often accompanied by nausea and sensitivity to light and sound. It can be triggered by stress, certain foods, or hormonal changes."
]

vectorstore = Chroma.from_texts(medical_docs, embeddings)
retriever = vectorstore.as_retriever()

# --- Prompt Templates ---

# 1. Symptom Analysis and Problem Decomposition Prompt
symptom_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a medical diagnostic assistant. Your task is to analyze patient symptoms, identify potential affected body systems, and decompose the problem into key areas for further investigation. Focus on breaking down the case logically."),
        ("human", "Analyze the following patient information and outline a step-by-step diagnostic approach:\nPatient Symptoms: {patient_symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}"),
    ]
)

# 2. Differential Diagnosis (Chain-of-Thought) Prompt
differential_diagnosis_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Based on the initial symptom analysis and medical facts, generate a list of possible differential diagnoses. For each diagnosis, provide a brief Chain-of-Thought explanation connecting it to the patient's symptoms and why it's a plausible option. Also, mention any key distinguishing factors."),
        ("human", "Initial Analysis: {initial_analysis}\nRelevant Medical Facts: {medical_facts}\nPatient Symptoms: {patient_symptoms}\nGenerate differential diagnoses with reasoning:"),
    ]
)

# 3. Verification Prompt
verification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a medical verifier. Your task is to critically evaluate the proposed differential diagnoses and their reasoning against provided medical facts. Identify any inconsistencies, unsupported claims, or areas needing further clarification. Rate the confidence (1-5) for each proposed diagnosis."),
        ("human", "Proposed Diagnoses and Reasoning: {differential_diagnoses}\nRelevant Medical Facts: {medical_facts}\nPatient Symptoms: {patient_symptoms}\nPerform a verification:"),
    ]
)

# --- LangChain Runnable Chains ---

def get_relevant_facts(inputs: Dict[str, Any]) -> List[str]:
    """Retrieves medical facts based on patient symptoms."""
    query = inputs["patient_symptoms"]
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]

# Chain 1: Symptom Analysis and Problem Decomposition
symptom_analysis_chain = (
    symptom_analysis_prompt
    | llm
    | StrOutputParser()
)

# Chain 2: Differential Diagnosis (includes RAG and CoT)
differential_diagnosis_chain = (
    RunnablePassthrough.assign(
        medical_facts=get_relevant_facts
    ) 
    | differential_diagnosis_prompt
    | llm
    | StrOutputParser()
)

# Chain 3: Verification
verification_chain = (
    RunnablePassthrough.assign(
        medical_facts=get_relevant_facts
    ) 
    | verification_prompt
    | llm
    | StrOutputParser()
)

# --- Main Diagnostic Flow --- 

def run_diagnostic_assistant(
    patient_symptoms: str,
    medical_history: str = "",
    lab_results: str = ""
) -> Dict[str, Any]:
    """Orchestrates the diagnostic process."""
    print("\n--- Running Diagnostic Assistant ---")
    print(f"Patient Symptoms: {patient_symptoms}")

    # Step 1: Symptom Analysis and Problem Decomposition
    print("\n--- Step 1: Symptom Analysis ---")
    initial_analysis = symptom_analysis_chain.invoke({
        "patient_symptoms": patient_symptoms,
        "medical_history": medical_history,
        "lab_results": lab_results
    })
    print("Initial Analysis and Decomposition:\n", initial_analysis)

    # Step 2: Differential Diagnosis with Chain-of-Thought and RAG
    print("\n--- Step 2: Generating Differential Diagnoses (CoT + RAG) ---")
    differential_diagnoses_output = differential_diagnosis_chain.invoke({
        "patient_symptoms": patient_symptoms,
        "initial_analysis": initial_analysis
    })
    print("Proposed Differential Diagnoses:\n", differential_diagnoses_output)

    # Step 3: Verification
    print("\n--- Step 3: Verifying Diagnoses ---")
    verification_output = verification_chain.invoke({
        "patient_symptoms": patient_symptoms,
        "differential_diagnoses": differential_diagnoses_output
    })
    print("Verification Results:\n", verification_output)

    return {
        "initial_analysis": initial_analysis,
        "differential_diagnoses": differential_diagnoses_output,
        "verification_results": verification_output
    }

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Respiratory symptoms
    patient_case_1 = {
        "patient_symptoms": "Severe cough with yellow phlegm, fever of 102°F, shortness of breath, and chills for 3 days.",
        "medical_history": "Has a history of seasonal allergies.",
        "lab_results": "Chest X-ray shows consolidation in the lower left lung lobe."
    }
    print("\n========================================")
    print("    MEDICAL CASE 1: RESPIRATORY ISSUE    ")
    print("========================================")
    run_diagnostic_assistant(**patient_case_1)

    print("\n\n")

    # Example 2: General malaise and fatigue
    patient_case_2 = {
        "patient_symptoms": "Persistent fatigue, increased thirst and urination, unexplained weight loss over the past month.",
        "medical_history": "No significant medical history.",
        "lab_results": "Fasting blood glucose: 250 mg/dL."
    }
    print("\n========================================")
    print("    MEDICAL CASE 2: METABOLIC ISSUE    ")
    print("========================================")
    run_diagnostic_assistant(**patient_case_2)

    # Example 3: Headache with visual disturbances
    patient_case_3 = {
        "patient_symptoms": "Intense throbbing headache on one side of the head, nausea, sensitivity to light and sound, with visual aura.",
        "medical_history": "Family history of migraines.",
        "lab_results": "Normal neurological exam."
    }
    print("\n========================================")
    print("    MEDICAL CASE 3: NEUROLOGICAL ISSUE    ")
    print("========================================")
    run_diagnostic_assistant(**patient_case_3)
