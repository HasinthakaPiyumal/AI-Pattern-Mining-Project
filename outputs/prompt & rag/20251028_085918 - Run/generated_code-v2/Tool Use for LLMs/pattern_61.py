import pandas as pd
import numpy as np
import random

# Placeholder for ML models
class MockModel:
    def predict(self, data):
        return np.random.rand(len(data))

mock_risk_model = MockModel()
mock_efficacy_model = MockModel()

# --- 1. Data Ingestion & Preprocessing Layer ---
def ingest_patient_data(filepath='dummy_patient_data.csv'):
    # Simulate data ingestion
    data = pd.DataFrame({
        'patient_id': range(1, 11),
        'age': np.random.randint(40, 80, 10),
        'gender': np.random.choice(['Male', 'Female'], 10),
        'diagnosis': np.random.choice(['Diabetes', 'Hypertension', 'Asthma', 'Arthritis'], 10),
        'medications': [random.sample(['Metformin', 'Lisinopril', 'Albuterol', 'Aspirin', 'Insulin'], random.randint(1, 3)) for _ in range(10)],
        'allergies': [random.sample(['Penicillin', 'Sulfonamides', 'NSAIDs'], random.randint(0, 2)) if i % 2 == 0 else [] for i in range(10)],
        'symptoms': [random.sample(['Fatigue', 'Dizziness', 'Shortness of Breath', 'Joint Pain'], random.randint(1, 3)) for _ in range(10)],
        'vitals_bp_systolic': np.random.randint(120, 180, 10),
        'vitals_bp_diastolic': np.random.randint(70, 110, 10),
    })
    return data

def preprocess_patient_data(df):
    # Basic preprocessing: one-hot encode categorical features for mock models
    df_processed = df.copy()
    df_processed['num_medications'] = df_processed['medications'].apply(len)
    df_processed['num_allergies'] = df_processed['allergies'].apply(len)
    df_processed['num_symptoms'] = df_processed['symptoms'].apply(len)

    # Simulate NLP feature extraction for diagnosis (e.g., embedding or simple one-hot)
    for diag in ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis']:
        df_processed[f'diag_{diag.lower()}'] = df_processed['diagnosis'].apply(lambda x: 1 if diag in x else 0)

    # Select numerical features for mock models
    features = ['age', 'num_medications', 'num_allergies', 'num_symptoms', 'vitals_bp_systolic', 'vitals_bp_diastolic'] + \
               [f'diag_{d.lower()}' for d in ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis']]
    return df_processed[features], df_processed['patient_id']

# --- 2. Patient Profile & Disease Modeling Layer ---
def create_patient_profile(patient_data):
    # This would involve more sophisticated feature engineering and NLP for real data
    profile = {
        'patient_id': patient_data['patient_id'].iloc[0],
        'age': patient_data['age'].iloc[0],
        'gender': patient_data['gender'].iloc[0],
        'diagnosis': patient_data['diagnosis'].iloc[0],
        'medications': patient_data['medications'].iloc[0],
        'allergies': patient_data['allergies'].iloc[0],
        'symptoms': patient_data['symptoms'].iloc[0],
        'vitals': {
            'bp_systolic': patient_data['vitals_bp_systolic'].iloc[0],
            'bp_diastolic': patient_data['vitals_bp_diastolic'].iloc[0],
        }
    }
    return profile

def predict_disease_risk(processed_features):
    # Mock risk prediction
    risk_score = mock_risk_model.predict(processed_features)
    return float(risk_score[0]) # Return a single float for a single patient

# --- 3. ToRA Core - Treatment Recommendation & Optimization Layer ---
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# Mock API Key for demonstration, replace with actual key or environment variable
OPENAI_API_KEY = "sk-..."

llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)

@tool
def medical_knowledge_base_query(query: str) -> str:
    """Queries a medical knowledge base for information on diseases, treatments, or guidelines."""
    if "diabetes" in query.lower():
        return "For diabetes, common treatments include Metformin, insulin, and lifestyle modifications like diet and exercise. Regular blood sugar monitoring is crucial."
    elif "hypertension" in query.lower():
        return "Hypertension treatment often involves ACE inhibitors, ARBs, diuretics, beta-blockers, and lifestyle changes such as reduced sodium intake and exercise."
    else:
        return f"Information for '{query}' is not readily available in the mock knowledge base."

@tool
def drug_interaction_check(medications: list[str]) -> str:
    """Checks for potential drug-drug interactions between a list of medications."""
    interactions = []
    if "Metformin" in medications and "Lisinopril" in medications:
        interactions.append("Potential for increased risk of kidney issues with Metformin and Lisinopril. Monitor renal function.")
    if "Albuterol" in medications and "Lisinopril" in medications:
        interactions.append("Albuterol may reduce the antihypertensive effect of Lisinopril. Monitor blood pressure.")
    if not interactions:
        return "No significant drug-drug interactions found among the provided medications."
    return ". ".join(interactions)

@tool
def patient_similarity_search(patient_profile_summary: str) -> str:
    """Finds similar patient cases based on a summary of the current patient's profile to provide insights into effective treatments."""
    # In a real system, this would use embeddings and a vector database
    if "diabetes" in patient_profile_summary.lower() and "hypertension" in patient_profile_summary.lower():
        return "Found 3 similar patients with both diabetes and hypertension. They responded well to a combination of Metformin, Lisinopril, and a supervised exercise program."
    elif "diabetes" in patient_profile_summary.lower():
        return "Found 5 similar patients with diabetes. Many showed good control with Metformin and dietary changes."
    else:
        return "No highly similar patient cases found for the given profile."

@tool
def treatment_efficacy_prediction(treatment_plan: str, patient_profile_summary: str) -> str:
    """Predicts the likely efficacy of a proposed treatment plan for the given patient profile."""
    # This would call a trained ML model (mock_efficacy_model)
    # For simplicity, we'll use a rule-based mock prediction here.
    if "Metformin" in treatment_plan and "Diabetes" in patient_profile_summary:
        efficacy_score = 0.85 # High efficacy
    elif "Lisinopril" in treatment_plan and "Hypertension" in patient_profile_summary:
        efficacy_score = 0.75 # Moderate-high efficacy
    else:
        efficacy_score = 0.60 # Baseline efficacy
    return f"Predicted efficacy score for this plan: {efficacy_score:.2f} (higher is better)."

def initialize_tora_agent():
    tools = [
        medical_knowledge_base_query,
        drug_interaction_check,
        patient_similarity_search,
        treatment_efficacy_prediction
    ]

    # Use AgentType.OPENAI_FUNCTIONS for best results with OpenAI models
    agent_executor = initialize_agent(
        tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True
    )
    return agent_executor

def run_tora_workflow(patient_profile, disease_risk_score, agent_executor):
    patient_summary = f"Patient ID: {patient_profile['patient_id']}, Age: {patient_profile['age']}, Gender: {patient_profile['gender']}, Diagnosis: {patient_profile['diagnosis']}, Medications: {', '.join(patient_profile['medications'])}, Allergies: {', '.join(patient_profile['allergies'])}, Symptoms: {', '.join(patient_profile['symptoms'])}, Disease Risk Score: {disease_risk_score:.2f}."

    initial_prompt = f"Given the patient profile: {patient_summary}, and a predicted disease exacerbation risk of {disease_risk_score:.2f}. Propose a personalized treatment plan. Consider the patient's diagnosis, current medications, allergies. Use the available tools to gather information, check for interactions, find similar cases, and predict efficacy. Provide the final recommended treatment plan with justification and potential alternatives."

    response = agent_executor.run(initial_prompt)
    return response

# --- 4. Explanation & User Interface Layer (Streamlit) ---
import streamlit as st

st.set_page_config(layout="wide")
st.title("AI-powered Personalized Treatment Recommendation System")
st.subheader("For Chronic Diseases using Tool-Integrated Reasoning Agent (ToRA)")

# Initialize agent once
@st.cache_resource
def get_agent():
    return initialize_tora_agent()

agent_executor = get_agent()

st.write("### Patient Data Input")

# Simulate getting a patient from the dummy data
dummy_data = ingest_patient_data()
patient_ids = dummy_data['patient_id'].tolist()
selected_patient_id = st.selectbox("Select Patient ID", patient_ids)

if st.button("Generate Treatment Recommendation"):
    if selected_patient_id:
        patient_raw_data = dummy_data[dummy_data['patient_id'] == selected_patient_id].iloc[0]
        st.write("#### Raw Patient Data:")
        st.json(patient_raw_data.to_dict())

        processed_features_df, _ = preprocess_patient_data(dummy_data[dummy_data['patient_id'] == selected_patient_id])
        patient_profile = create_patient_profile(dummy_data[dummy_data['patient_id'] == selected_patient_id])
        disease_risk = predict_disease_risk(processed_features_df)

        st.write("#### Processed Patient Profile:")
        st.json(patient_profile)
        st.write(f"**Predicted Disease Risk Score:** {disease_risk:.2f}")

        st.write("#### Running ToRA Agent for Recommendations...")
        with st.spinner("The AI agent is reasoning and using tools to generate recommendations. This may take a moment..."):
            try:
                recommendation = run_tora_workflow(patient_profile, disease_risk, agent_executor)
                st.success("Recommendation Generated!")
                st.write("#### Recommended Treatment Plan:")
                st.markdown(recommendation)
            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")
                st.warning("Please ensure your OpenAI API key is correctly set up if you are encountering authentication issues.")
    else:
        st.warning("Please select a patient ID.")
