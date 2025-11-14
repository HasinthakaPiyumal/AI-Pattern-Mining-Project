import streamlit as st
import pandas as pd
import random
from collections import Counter

# LangChain imports (using simulated LLM for demonstration)
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM

# --- Configuration --- 
NUM_ENSEMBLE_RUNS = 5 # Number of times to run the LLM with different exemplar subsets
NUM_FEW_SHOT_EXEMPLARS = 3 # Number of exemplars to include in each prompt

# --- Dummy Patient Case Exemplars (replace with a real dataset in production) ---
patient_exemplars_data = [
    {"symptoms": "Fever, rash, joint pain, fatigue", "diagnosis": "Lupus"},
    {"symptoms": "Severe headache, stiff neck, sensitivity to light", "diagnosis": "Meningitis"},
    {"symptoms": "Chronic cough, night sweats, weight loss", "diagnosis": "Tuberculosis"},
    {"symptoms": "Sudden weakness on one side of the body, difficulty speaking, confusion", "diagnosis": "Stroke"},
    {"symptoms": "Extreme tiredness, muscle weakness, dry eyes, dry mouth", "diagnosis": "Sjogren's Syndrome"},
    {"symptoms": "Swelling in hands and feet, high blood pressure, protein in urine", "diagnosis": "Preeclampsia"},
    {"symptoms": "Frequent urination, increased thirst, unexplained weight loss", "diagnosis": "Diabetes Mellitus Type 1"},
    {"symptoms": "Chest pain, shortness of breath, dizziness", "diagnosis": "Heart Attack"},
    {"symptoms": "Persistent sore throat, difficulty swallowing, ear pain", "diagnosis": "Throat Cancer"},
    {"symptoms": "Yellow skin, dark urine, abdominal pain", "diagnosis": "Hepatitis"},
]
pdf_exemplars = pd.DataFrame(patient_exemplars_data)

# --- Simulated LLM (replace with actual LLM like OpenAI's GPT in production) ---
def get_simulated_llm():
    # In a real application, you would initialize a ChatOpenAI or similar here:
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4", api_key="YOUR_API_KEY")
    
    # For demonstration, we use FakeListLLM to simulate responses.
    # The responses are designed to show how different prompts might lead to different (or similar) outputs.
    # In a real scenario, the LLM would generate these based on the prompt.
    fake_responses = [
        "Lupus", "Meningitis", "Sjogren's Syndrome", "Stroke", "Tuberculosis", # Common from exemplars
        "Fibromyalgia", "Chronic Fatigue Syndrome", "Multiple Sclerosis", # Similar but different
        "Undetermined - requires further testing", # Ambiguous
    ]
    return FakeListLLM(responses=fake_responses * 2) # Repeat to have enough responses

llm = get_simulated_llm()

# --- Prompt Generation Function ---
def generate_few_shot_prompt(user_symptoms: str, exemplars: pd.DataFrame) -> str:
    exemplar_string = ""
    for _, row in exemplars.iterrows():
        exemplar_string += f"Symptoms: {row['symptoms']}\nDiagnosis: {row['diagnosis']}\n\n"
    
    template = PromptTemplate(
        input_variables=["exemplars", "user_symptoms"],
        template=(
            "You are a highly skilled medical diagnosis assistant. Your task is to provide a probable diagnosis "
            "for a patient based on their symptoms. Use the provided few-shot examples as guidance to infer the diagnosis. "
            "If the symptoms do not clearly match a known condition, provide the most plausible rare disease or indicate uncertainty.\n\n"
            "--- Exemplar Patient Cases ---\n"
            "{exemplars}"
            "--- New Patient ---\n"
            "Symptoms: {user_symptoms}\n"
            "Diagnosis: "
        ),
    )
    return template.format(exemplars=exemplar_string, user_symptoms=user_symptoms)

# --- Ensembling Function ---
def ensemble_diagnoses(diagnoses: list) -> str:
    if not diagnoses:
        return "No diagnosis could be determined."
    
    # Majority voting for diagnoses
    diagnosis_counts = Counter(diagnoses)
    most_common_diagnosis = diagnosis_counts.most_common(1)[0][0]
    
    # You could also consider a confidence score if your LLM provided one
    # For simplicity, we stick to majority voting here.
    
    return f"Ensembled Diagnosis (Most Common): {most_common_diagnosis}"

# --- Streamlit Application ---
st.set_page_config(page_title="Rare Disease Diagnosis Support", layout="centered")
st.title("🩺 Rare Disease Diagnosis Support System")
st.markdown("--- Developed using **Demonstration Ensembling (DENSE)** ---")

st.write(
    "This system assists medical professionals in diagnosing rare diseases by leveraging a FewShot Prompting approach "
    "with an LLM. To improve accuracy and reduce variance, it aggregates diagnostic suggestions from multiple prompts, "
    "each with distinct subsets of anonymized patient case exemplars."
)

# User Input
user_symptoms = st.text_area(
    "Enter patient symptoms (e.g., 'Chronic fatigue, muscle weakness, difficulty concentrating'):",
    height=150,
    placeholder="e.g., Persistent joint pain, skin rash, extreme fatigue, dry eyes"
)

if st.button("Get Diagnosis"):
    if user_symptoms:
        with st.spinner("Generating multiple diagnostic suggestions..."):
            all_diagnoses = []
            for i in range(NUM_ENSEMBLE_RUNS):
                st.info(f"Running ensemble iteration {i+1}/{NUM_ENSEMBLE_RUNS}...")
                
                # Select a random subset of exemplars for this run
                # Ensure we don't pick more exemplars than available
                num_exemplars_to_sample = min(NUM_FEW_SHOT_EXEMPLARS, len(pdf_exemplars))
                sampled_exemplars = pdf_exemplars.sample(n=num_exemplars_to_sample, random_state=random.randint(0, 10000))
                
                # Generate prompt
                prompt_text = generate_few_shot_prompt(user_symptoms, sampled_exemplars)
                
                # Send prompt to LLM and get response
                try:
                    # In a real scenario, you'd use invoke or stream method from the actual LLM instance
                    # For FakeListLLM, we directly get a response.
                    llm_response = llm.invoke(prompt_text)
                    current_diagnosis = llm_response.strip()
                    all_diagnoses.append(current_diagnosis)
                    st.write(f"  Iteration {i+1} suggested: **{current_diagnosis}**")
                except Exception as e:
                    st.error(f"Error during LLM call for iteration {i+1}: {e}")
                    st.write(f"  Prompt used in iteration {i+1}:\n```\n{prompt_text}\n```")
            
            if all_diagnoses:
                st.subheader("Individual LLM Suggestions:")
                for diag in all_diagnoses:
                    st.write(f"- {diag}")
                
                # Ensembling
                final_diagnosis = ensemble_diagnoses(all_diagnoses)
                st.subheader("Final Ensembled Diagnosis:")
                st.success(f"**{final_diagnosis}**")
            else:
                st.error("Could not generate any diagnoses. Please try again or refine your symptoms.")
    else:
        st.warning("Please enter patient symptoms to get a diagnosis.")

st.markdown("---")
st.info(
    "Note: This is a demonstration. The LLM is simulated, and the exemplar dataset is small and dummy. "
    "A production system would use a robust LLM (e.g., OpenAI GPT-4), a large, curated, and anonymized "
    "medical dataset, and potentially more sophisticated ensembling techniques."
)

