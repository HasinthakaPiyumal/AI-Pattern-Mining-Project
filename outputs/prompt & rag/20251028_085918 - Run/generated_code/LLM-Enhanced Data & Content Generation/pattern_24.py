import streamlit as st
import os
from openai import OpenAI

# Mock Medical Knowledge Graph (simplified for demonstration)
MEDICAL_KNOWLEDGE_GRAPH = {
    "Common Cold": {
        "symptoms": ["runny nose", "sore throat", "cough", "sneezing", "mild headache", "fatigue"],
        "treatments": ["rest", "fluids", "over-the-counter cold medicine"],
        "description": "A viral infection of the nose and throat."
    },
    "Influenza (Flu)": {
        "symptoms": ["fever", "body aches", "chills", "fatigue", "cough", "sore throat", "headache"],
        "treatments": ["rest", "fluids", "antiviral medication (if prescribed)", "pain relievers"],
        "description": "A contagious respiratory illness caused by influenza viruses."
    },
    "Streptococcal Pharyngitis (Strep Throat)": {
        "symptoms": ["sore throat", "difficulty swallowing", "fever", "red spots on roof of mouth", "swollen lymph nodes"],
        "treatments": ["antibiotics"],
        "description": "A bacterial infection that can make your throat feel sore and scratchy."
    },
    "Migraine": {
        "symptoms": ["severe headache", "pulsating head pain", "nausea", "vomiting", "sensitivity to light", "sensitivity to sound"],
        "treatments": ["pain relievers", "triptans", "rest in a dark room"],
        "description": "A severe type of headache often accompanied by other symptoms."
    },
    "Allergic Rhinitis (Hay Fever)": {
        "symptoms": ["sneezing", "runny nose", "itchy eyes", "nasal congestion"],
        "treatments": ["antihistamines", "nasal corticosteroids", "avoiding allergens"],
        "description": "An allergic reaction to airborne allergens, such as pollen or dust mites."
    }
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def retrieve_relevant_knowledge(symptoms_input):
    relevant_conditions = set()
    retrieved_info = []

    for condition, data in MEDICAL_KNOWLEDGE_GRAPH.items():
        condition_symptoms = [s.lower() for s in data["symptoms"]]
        for symptom in symptoms_input:
            if symptom.lower() in condition_symptoms:
                relevant_conditions.add(condition)
                break

    for condition in relevant_conditions:
        data = MEDICAL_KNOWLEDGE_GRAPH[condition]
        retrieved_info.append(f"Condition: {condition}")
        retrieved_info.append(f"Symptoms: {', '.join(data['symptoms'])}")
        retrieved_info.append(f"Treatments: {', '.join(data['treatments'])}")
        retrieved_info.append(f"Description: {data['description']}")
        retrieved_info.append("---")

    if not retrieved_info:
        return "No specific conditions found based on provided symptoms in the knowledge graph."
    return "\n".join(retrieved_info)

def reason_and_recommend(symptoms_input, patient_data=""):
    retrieved_knowledge = retrieve_relevant_knowledge(symptoms_input)

    prompt = f"""
    You are a medical diagnosis and treatment recommendation AI. Your goal is to provide potential diagnoses and treatment suggestions based on the provided symptoms and medical knowledge. Always explain your reasoning.

    Patient Symptoms: {', '.join(symptoms_input)}
    Patient Data: {patient_data if patient_data else 'None provided.'}

    Relevant Medical Knowledge:
    {retrieved_knowledge}

    Based on the above information, what are the potential diagnoses, and what treatment recommendations would you suggest? Please provide a detailed explanation of your reasoning. If the symptoms do not directly match known conditions, provide general medical advice or suggest seeing a doctor.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while communicating with the AI: {e}. Please ensure your OPENAI_API_KEY is correctly set."

st.set_page_config(page_title="Medical Diagnosis & Treatment System")
st.title("🩺 Medical Diagnosis & Treatment Recommendation System")

st.markdown("Enter your symptoms below to get potential diagnoses and treatment recommendations.")

user_symptoms_input = st.text_area(
    "Enter your symptoms (comma-separated, e.g., 'fever, cough, sore throat'):",
    height=100
)

user_patient_data = st.text_area(
    "Optional: Enter any relevant patient data (e.g., 'age 35, no known allergies, takes blood pressure medication'):",
    height=70
)

if st.button("Get Recommendations"):
    if user_symptoms_input:
        symptoms_list = [s.strip() for s in user_symptoms_input.split(',') if s.strip()]
        if symptoms_list:
            with st.spinner("Analyzing symptoms and generating recommendations..."):
                recommendations = reason_and_recommend(symptoms_list, user_patient_data)
                st.subheader("Diagnosis and Treatment Recommendations:")
                st.write(recommendations)
        else:
            st.warning("Please enter at least one symptom.")
    else:
        st.warning("Please enter your symptoms.")

st.markdown("""
---
**Disclaimer:** This system is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")

