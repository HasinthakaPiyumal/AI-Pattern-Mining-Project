from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# --- 1. Simulated External Medical Knowledge Base ---
medical_knowledge_base = {
    "fever and cough": [
        "Common cold: Viral infection, self-limiting, rest and fluids.",
        "Influenza: Viral infection, may require antivirals, flu shot recommended.",
        "Pneumonia: Bacterial or viral lung infection, can be serious, requires antibiotics for bacterial cases."
    ],
    "chest pain and shortness of breath": [
        "Myocardial Infarction (Heart Attack): Medical emergency, seek immediate help.",
        "Angina: Chest pain due to reduced blood flow to the heart, often triggered by exertion.",
        "Pleurisy: Inflammation of the lining of the lungs, causes sharp chest pain."
    ],
    "headache and stiff neck": [
        "Meningitis: Inflammation of the membranes surrounding the brain and spinal cord, serious, urgent medical attention.",
        "Tension Headache: Common, often stress-related, responds to OTC pain relievers.",
        "Migraine: Severe headache, often with throbbing pain, sensitivity to light/sound, nausea."
    ],
    "abdominal pain and nausea": [
        "Gastroenteritis (Stomach Flu): Viral infection, vomiting and diarrhea, self-limiting.",
        "Appendicitis: Inflammation of the appendix, requires surgery, characterized by right lower quadrant pain.",
        "Gallstones: Hardened deposits in the gallbladder, can cause severe pain, especially after fatty meals."
    ]
}

def retrieve_medical_info(query: str) -> str:
    """Simulates retrieval of relevant medical information based on keywords."""
    retrieved_docs = []
    query_keywords = query.lower().split()
    for key, docs in medical_knowledge_base.items():
        if any(keyword in key for keyword in query_keywords):
            retrieved_docs.extend(docs)
    return "\n".join(retrieved_docs) if retrieved_docs else "No specific external medical information found for this query."

# --- 2. LLM for Chain of Thought Generation and Editing ---
llm = ChatGoogleGenerativeAI(model="gemini-pro") # Initialize Gemini Pro LLM

# Prompt for initial Chain of Thought generation
cot_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly intelligent medical diagnostic assistant. Your task is to generate several distinct diagnostic reasoning paths (Chain of Thought) based on patient symptoms and history. Each path should lead to a potential diagnosis and outline the reasoning process."),
    ("user", "Patient Symptoms: {symptoms}\nMedical History: {medical_history}\n\nGenerate 2-3 distinct diagnostic reasoning paths.")
])

# Prompt for verifying and editing Chain of Thought with external information
verify_edit_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical expert tasked with verifying and editing diagnostic reasoning paths. You will be provided with an initial diagnostic path and relevant external medical information. Your goal is to review the path, incorporate the external information to improve its accuracy, add missing steps, correct errors, and confirm/refine the diagnosis. If the external information strongly suggests an alternative, incorporate that."),
    ("user", "Initial Diagnostic Path:\n{initial_cot}\n\nExternal Medical Information:\n{external_info}\n\nReview and refine the diagnostic path based on the external information. Provide the refined path and a clear, concise final diagnosis.")
])

# --- 3. Orchestration with LangChain ---

def get_patient_input():
    """Simulates getting patient symptoms and medical history from a user."""
    print("\n--- Medical Diagnosis Assistant ---")
    symptoms = input("Enter patient symptoms (e.g., 'fever and cough'): ")
    medical_history = input("Enter patient medical history (e.g., 'no significant history'): ")
    return {"symptoms": symptoms, "medical_history": medical_history}

def diagnose_patient():
    patient_data = get_patient_input()
    symptoms = patient_data["symptoms"]
    medical_history = patient_data["medical_history"]

    # Step 1: Generate initial Chains of Thought
    print("\nGenerating initial diagnostic reasoning paths...")
    cot_generator_chain = cot_generation_prompt | llm
    initial_cots_response = cot_generator_chain.invoke({"symptoms": symptoms, "medical_history": medical_history})
    initial_cots_raw = initial_cots_response.content.strip()
    initial_cots_list = [path.strip() for path in initial_cots_raw.split("Diagnostic Path") if path.strip()]

    if not initial_cots_list:
        print("Could not generate initial diagnostic paths. Please try again with different input.")
        return

    print("Initial Paths Generated:")
    for i, cot in enumerate(initial_cots_list):
        print(f"\n--- Initial Path {i+1} ---\n{cot}")

    # Step 2: Select paths for verification, retrieve info, and edit
    print("\nVerifying and editing diagnostic paths with external knowledge...")
    verified_diagnoses = []

    for i, initial_cot in enumerate(initial_cots_list):
        print(f"\nProcessing Path {i+1} for verification...")

        # Retrieve external information based on the initial symptoms
        # In a real system, this would be more sophisticated, perhaps extracting keywords from the CoT itself.
        external_info = retrieve_medical_info(symptoms)
        print(f"Retrieved external info for Path {i+1}:\n{external_info}")

        # Verify and Edit chain
        verify_edit_chain = verify_edit_prompt | llm
        edited_cot_response = verify_edit_chain.invoke({
            "initial_cot": initial_cot,
            "external_info": external_info
        })
        edited_cot = edited_cot_response.content.strip()
        verified_diagnoses.append(f"Path {i+1}:\n{edited_cot}")

    # Step 3: Present final verified diagnoses
    print("\n--- Final Verified Diagnoses and Reasoning Paths ---")
    for diagnosis_output in verified_diagnoses:
        print(f"\n{diagnosis_output}")

if __name__ == "__main__":
    diagnose_patient()