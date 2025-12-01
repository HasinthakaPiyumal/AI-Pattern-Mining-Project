import streamlit as st
import time
import threading
import random

def initial_response_generator(symptoms):
    time.sleep(0.5)  # Simulate very low latency
    responses = [
        f"Based on your symptoms, it sounds like you might have a common cold. Please note this is a preliminary assessment.",
        f"Initial thought: Your symptoms suggest a possible mild viral infection. Further analysis needed.",
        f"Preliminary finding: Could be allergies, given your symptoms. This is a quick check."
    ]
    return random.choice(responses)

def refined_response_generator(symptoms):
    time.sleep(5)  # Simulate higher latency for detailed analysis
    detailed_responses = {
        "fever, cough, fatigue": "Refined Report: Your symptoms (fever, cough, fatigue) are highly indicative of Influenza. We recommend consulting a doctor for proper diagnosis and treatment. Consider resting and staying hydrated.",
        "sore throat, runny nose, sneezing": "Refined Report: A more accurate analysis suggests you likely have Rhinovirus (common cold). Rest, fluids, and over-the-counter medication should help. Watch for worsening symptoms.",
        "headache, stiff neck, sensitivity to light": "Refined Report: These symptoms (headache, stiff neck, sensitivity to light) could indicate Meningitis. This is a serious condition, and immediate medical attention is strongly advised. Do not wait.",
        "rash, itching, swelling": "Refined Report: Your symptoms (rash, itching, swelling) strongly point towards an Allergic Reaction. Try to identify potential allergens. If severe, seek immediate medical care."
    }
    default_refined = "Refined Report: Based on a comprehensive analysis of your symptoms, a more precise diagnosis is difficult without additional information. We strongly recommend consulting a healthcare professional for a definitive diagnosis and personalized advice."
    
    # Simple keyword matching for demo purposes
    for key_symptoms, response in detailed_responses.items():
        if all(symptom in symptoms.lower() for symptom in key_symptoms.split(', ')):
            return response
    return default_refined

def call_refined_generator_thread(symptoms):
    st.session_state.refined_response = refined_response_generator(symptoms)
    st.session_state.refined_response_ready = True
    st.session_state.is_processing_refined = False
    st.experimental_rerun() # Rerun Streamlit to update UI

st.set_page_config(page_title="MediCheck AI Assistant", layout="centered")
st.title("MediCheck AI Assistant")
st.markdown("Enter your symptoms below to get a preliminary assessment and a more refined diagnostic report.")

# Initialize session state
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = ""
if 'initial_response' not in st.session_state:
    st.session_state.initial_response = None
if 'refined_response' not in st.session_state:
    st.session_state.refined_response = None
if 'is_processing_refined' not in st.session_state:
    st.session_state.is_processing_refined = False
if 'refined_response_ready' not in st.session_state:
    st.session_state.refined_response_ready = False

symptoms_input = st.text_area("Describe your symptoms (e.g., 'fever, cough, fatigue'):", value=st.session_state.symptoms, height=150)

if st.button("Get Assessment"):
    if symptoms_input:
        st.session_state.symptoms = symptoms_input
        st.session_state.initial_response = None
        st.session_state.refined_response = None
        st.session_state.is_processing_refined = True
        st.session_state.refined_response_ready = False
        
        # Generate and display initial response immediately
        st.session_state.initial_response = initial_response_generator(st.session_state.symptoms)
        st.subheader("Preliminary Assessment (Immediate):")
        st.info(st.session_state.initial_response)
        
        # Start refined response generation in a separate thread
        thread = threading.Thread(target=call_refined_generator_thread, args=(st.session_state.symptoms,))
        thread.start()
        st.warning("A more detailed and accurate report is being processed... Please wait or view the preliminary assessment above.")
        
    else:
        st.error("Please enter your symptoms to get an assessment.")

# Display current status and responses
if st.session_state.initial_response and not st.session_state.is_processing_refined:
    st.subheader("Preliminary Assessment (Immediate):")
    st.info(st.session_state.initial_response)

if st.session_state.is_processing_refined and not st.session_state.refined_response_ready:
    st.warning("A more detailed and accurate report is being processed... Please wait or view the preliminary assessment above.")
    
if st.session_state.refined_response_ready:
    st.subheader("Refined Diagnostic Report (Detailed):")
    st.success(st.session_state.refined_response)
    st.session_state.is_processing_refined = False # Ensure processing flag is off after display
