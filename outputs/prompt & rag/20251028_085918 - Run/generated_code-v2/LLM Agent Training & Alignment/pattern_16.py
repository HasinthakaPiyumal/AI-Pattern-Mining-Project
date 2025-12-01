import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import random
import re

# --- 1. Configuration and Constitutional Principles ---

class ConstitutionalPrinciples:
    FACTUALITY = "The response must be evidence-based and verifiable from reputable medical sources. Do not invent facts."
    HARMLESSNESS = "Avoid providing direct medical advice or diagnoses. Always recommend consulting healthcare professionals for any health concerns or before making medical decisions."
    NON_DISCRIMINATION = "Ensure recommendations and information are free from bias related to age, gender, ethnicity, socioeconomic status, or any other demographic. Provide universally applicable information unless a specific demographic context is explicitly requested and medically relevant."
    PRIVACY = "Do not request, store, or process sensitive personal health information beyond what is strictly necessary for the current interaction, and always assume user data is private."
    TRANSPARENCY = "Explain the limitations of AI, state that this is not a substitute for professional medical advice, and encourage critical thinking and verification."

    ALL_PRINCIPLES = [
        FACTUALITY,
        HARMLESSNESS,
        NON_DISCRIMINATION,
        PRIVACY,
        TRANSPARENCY
    ]

# --- 2. Mock Health Knowledge Base (Simplified RAG Source) ---

HEALTH_KB = {
    "common cold symptoms": "The common cold typically presents with symptoms like runny nose, sore throat, coughing, congestion, slight body aches, and sneezing. It is caused by viruses, and antibiotics are ineffective.",
    "diabetes prevention": "Preventing type 2 diabetes often involves maintaining a healthy weight, regular physical activity, and a balanced diet rich in fruits, vegetables, and whole grains. Regular check-ups are also important.",
    "headache remedies": "For tension headaches, over-the-counter pain relievers like ibuprofen or acetaminophen can help. Rest, hydration, and stress reduction techniques are also beneficial. Persistent or severe headaches require medical evaluation.",
    "heart disease risks": "Key risk factors for heart disease include high blood pressure, high cholesterol, diabetes, smoking, obesity, physical inactivity, and a family history of heart disease.",
    "healthy diet tips": "A healthy diet emphasizes whole, unprocessed foods, including a variety of fruits, vegetables, lean proteins, and whole grains. Limiting processed foods, sugary drinks, and excessive saturated/trans fats is crucial.",
    "vaccination benefits": "Vaccinations protect individuals from infectious diseases and help achieve herd immunity, reducing the spread of diseases within communities. They are a safe and effective public health measure."
}

def retrieve_health_information(query: str) -> str:
    """Simulates retrieval from a health knowledge base based on keywords."""
    relevant_info = []
    query_lower = query.lower()
    for topic, info in HEALTH_KB.items():
        if any(keyword in query_lower for keyword in topic.split()):
            relevant_info.append(info)
    return " ".join(relevant_info) if relevant_info else "No directly relevant information found in the knowledge base."

# --- 3. Mock Large Language Model (LLM) ---

def mock_llm_generate(prompt: str) -> str:
    """Simulates an LLM generating a response."""
    # A very simple mock. In a real system, this would be an actual LLM call.
    if "common cold" in prompt.lower():
        return "Based on your query, the common cold is a viral infection causing symptoms such as a runny nose, sore throat, and cough. Rest and fluids are often recommended. Please consult a doctor for diagnosis."
    elif "diabetes" in prompt.lower():
        return "Regarding diabetes, particularly type 2, lifestyle changes like diet and exercise are key for prevention. Regular medical check-ups are vital. This information is for general awareness."
    elif "headache" in prompt.lower():
        return "If you're experiencing a headache, basic pain relief and rest might help. However, I cannot diagnose. If headaches are severe or persistent, seek professional medical advice immediately."
    elif "heart disease" in prompt.lower():
        return "Risk factors for heart disease include high blood pressure and cholesterol. Maintaining a healthy lifestyle is important. For personalized advice, consult a healthcare provider."
    else:
        return "I can provide general health information. Please remember I am an AI and cannot offer medical advice. Always consult a healthcare professional for personalized guidance."

# --- 4. Constitutional AI Layer (Critique and Revision Simulation) ---

def critique_response(response: str, user_query: str) -> List[str]:
    """Simulates an AI critique model identifying principle violations."""
    violations = []

    # HARMLESSNESS check
    if "diagnose" in response.lower() or "treat" in response.lower() or \
       re.search(r'take (this|that) medicine', response.lower()):
        violations.append(ConstitutionalPrinciples.HARMLESSNESS)

    # FACTUALITY (very basic check, assumes LLM sometimes 'hallucinates' direct advice)
    # In a real system, this would compare against KB or external tools
    if "definitely cure" in response.lower() or "magic pill" in response.lower():
        violations.append(ConstitutionalPrinciples.FACTUALITY)

    # NON-DISCRIMINATION (very conceptual without user demographics)
    # This is hard to simulate without explicit bias in mock_llm_generate
    # For demonstration, assume a very subtle bias might be detected:
    if re.search(r'women over 50 should only', response.lower()): # Example of a biased statement
         violations.append(ConstitutionalPrinciples.NON_DISCRIMINATION)

    # PRIVACY (conceptual, focuses on output not input)
    if "what is your full name and address for medical record?" in response.lower():
        violations.append(ConstitutionalPrinciples.PRIVACY)

    # TRANSPARENCY
    if "I am an AI and cannot offer medical advice" not in response and \
       "consult a healthcare professional" not in response:
        violations.append(ConstitutionalPrinciples.TRANSPARENCY)

    return violations

def revise_response(original_response: str, violations: List[str], user_query: str) -> str:
    """Simulates the LLM revising its response based on critiques."""
    revised_response = original_response

    for violation_principle in violations:
        if violation_principle == ConstitutionalPrinciples.HARMLESSNESS:
            revised_response = re.sub(r'\b(diagnose|treat|cure)\b', 'provide information on', revised_response, flags=re.IGNORECASE)
            if "consult a healthcare professional" not in revised_response:
                revised_response += " Always consult a healthcare professional for diagnosis or treatment."
        elif violation_principle == ConstitutionalPrinciples.FACTUALITY:
            # For mock, we'll just generalize if factuality is an issue
            revised_response = "Please verify any health information with a medical professional. " + revised_response
        elif violation_principle == ConstitutionalPrinciples.NON_DISCRIMINATION:
            # Remove potentially biased phrases (very simplistic for mock)
            revised_response = re.sub(r'women over 50 should only', 'individuals should consider', revised_response, flags=re.IGNORECASE)
        elif violation_principle == ConstitutionalPrinciples.PRIVACY:
            # Remove any privacy-violating questions
            revised_response = re.sub(r'what is your full name and address for medical record\?', '', revised_response, flags=re.IGNORECASE)
        elif violation_principle == ConstitutionalPrinciples.TRANSPARENCY:
            if "I am an AI and cannot offer medical advice" not in revised_response:
                revised_response = "As an AI, I provide general information and cannot offer medical advice. " + revised_response

    # Ensure harmlessness and transparency disclaimer is always present after revision
    if ConstitutionalPrinciples.HARMLESSNESS in violations or ConstitutionalPrinciples.TRANSPARENCY in violations or not (ConstitutionalPrinciples.HARMLESSNESS in ConstitutionalPrinciples.ALL_PRINCIPLES and ConstitutionalPrinciples.TRANSPARENCY in ConstitutionalPrinciples.ALL_PRINCIPLES and not (ConstitutionalPrinciples.HARMLESSNESS in violations or ConstitutionalPrinciples.TRANSPARENCY in violations)):
        if "consult a healthcare professional" not in revised_response:
            revised_response += " It's crucial to consult a qualified healthcare professional for any medical concerns."
        if "As an AI, I provide general information" not in revised_response:
            revised_response = "As an AI, I provide general information and cannot offer medical advice. " + revised_response

    return revised_response.strip()

# --- 5. FastAPI Backend API --- 

app = FastAPI(
    title="Ethical AI Health Assistant",
    description="An AI assistant providing personalized health information aligned with ethical principles."
)

class QueryRequest(BaseModel):
    user_query: str

class HealthResponse(BaseModel):
    original_llm_response: str
    critiques: List[str]
    ethical_response: str
    explanation: str

@app.post("/ask_health_ai", response_model=HealthResponse)
async def ask_health_ai(request: QueryRequest):
    user_query = request.user_query

    # 1. RAG - Retrieve relevant information
    retrieved_info = retrieve_health_information(user_query)
    rag_prompt = f"Based on the following information: '{retrieved_info}'. User query: '{user_query}'. Provide general health information, avoiding direct medical advice."

    # 2. LLM Generation
    original_response = mock_llm_generate(rag_prompt)

    # 3. Constitutional AI - Critique
    violations = critique_response(original_response, user_query)

    ethical_response = original_response
    explanation = "Original LLM response generated."

    # 4. Constitutional AI - Revision Loop (if violations found)
    if violations:
        ethical_response = revise_response(original_response, violations, user_query)
        explanation = f"Original response critiqued for violating: {', '.join([p.split(':')[0] for p in violations])}. Revised to ensure ethical alignment."

    # Ensure final response always includes a disclaimer about AI limitations and seeking professional advice
    if "As an AI, I provide general information" not in ethical_response:
        ethical_response = "As an AI, I provide general information and cannot offer medical advice. " + ethical_response
    if "consult a qualified healthcare professional" not in ethical_response:
        ethical_response += " It is crucial to consult a qualified healthcare professional for any medical concerns."

    return HealthResponse(
        original_llm_response=original_response,
        critiques=[p.split(':')[0] for p in violations],
        ethical_response=ethical_response,
        explanation=explanation
    )

# --- 6. Streamlit Frontend (for demonstration) ---

# To run the Streamlit app, save this file as 'app.py' and run 'streamlit run app.py'
# To run the FastAPI app, run 'uvicorn ethical_ai_health_assistant:app --reload'
# For this combined file, you would typically run FastAPI and then Streamlit as separate processes or use an iframe/component if embedding.

# For simplicity, we'll put the Streamlit code here, but in a real project,
# this would be in a separate file (e.g., `frontend.py`) and interact with the FastAPI backend.

# To make this file directly runnable as a Streamlit app (and thus also implicitly run FastAPI for the demo)
# we will detect if it's being run by Streamlit.

if __name__ == "__main__":
    # This block allows running the FastAPI app directly for testing the API
    # You can also run it via `uvicorn ethical_ai_health_assistant:app --reload`
    print("\n--- Running FastAPI Backend ---")
    print("Access FastAPI interactive docs at: http://127.0.0.1:8000/docs")
    print("To interact with the Streamlit frontend, run: streamlit run ethical_ai_health_assistant.py (in a separate terminal)")

    # For the purpose of running both in one script for demonstration clarity,
    # we will use a conditional import for streamlit
    try:
        import streamlit as st
        import requests

        # Check if the script is being run by Streamlit
        if st._is_running_with_streamlit:
            st.set_page_config(layout="wide")
            st.title("🩺 Ethical AI Health Assistant")
            st.write("Get general health information aligned with ethical principles.")

            st.sidebar.header("Constitutional Principles")
            for principle in ConstitutionalPrinciples.ALL_PRINCIPLES:
                st.sidebar.markdown(f"- {principle}")

            user_input = st.text_input("Ask me a health-related question:", "What are the symptoms of the common cold?")

            if st.button("Get Ethical Health Info") and user_input:
                with st.spinner("Processing your request..."):
                    try:
                        # Call the FastAPI backend
                        api_url = "http://127.0.0.1:8000/ask_health_ai"
                        response = requests.post(api_url, json={"user_query": user_input})
                        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                        data = response.json()

                        st.subheader("Your Ethical Health Information")
                        st.success(data["ethical_response"])

                        with st.expander("See Details (Original Response & Critiques)"):
                            st.write("**Original LLM Response:**")
                            st.info(data["original_llm_response"])

                            if data["critiques"]:
                                st.write("**Critiques (Violated Principles):**")
                                for critique in data["critiques"]:
                                    st.warning(f"- {critique}")
                            else:
                                st.success("No ethical violations detected in the original response.")
                            st.write("**Explanation of Revision:**")
                            st.code(data["explanation"])

                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the FastAPI backend. Please ensure it's running (e.g., using 'uvicorn ethical_ai_health_assistant:app --reload').")
                    except requests.exceptions.RequestException as e:
                        st.error(f"An error occurred during the API call: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
        else:
            # If not run by Streamlit, run FastAPI directly
            uvicorn.run(app, host="127.0.0.1", port=8000)

    except ImportError:
        print("\n--- Streamlit not found ---")
        print("To run the full demo with UI, install Streamlit: pip install streamlit requests")
        print("Then run this file using Streamlit: streamlit run ethical_ai_health_assistant.py")
        print("Currently running FastAPI backend only.")
        uvicorn.run(app, host="127.0.0.1", port=8000)

