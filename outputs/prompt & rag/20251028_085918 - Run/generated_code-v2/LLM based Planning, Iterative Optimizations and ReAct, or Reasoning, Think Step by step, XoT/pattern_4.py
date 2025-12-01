import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# llm_service.py content
def get_diagnosis_with_cot(symptoms: str, openai_api_key: str):
    if not openai_api_key:
        raise ValueError("OpenAI API key is required.")

    chat = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

    prompt = f"""You are a highly intelligent medical diagnostic assistant. Your task is to analyze patient symptoms and provide a potential diagnosis along with a clear, step-by-step reasoning process. First, 'think step-by-step' through the problem, considering differential diagnoses and ruling out possibilities. Then, state your final diagnosis and suggested next steps.

Patient Symptoms: {symptoms}

Think step-by-step: """

    messages = [HumanMessage(content=prompt)]
    response = chat.invoke(messages)
    response_text = response.content

    reasoning = "No reasoning found."
    diagnosis = "No diagnosis found."
    next_steps = "No next steps found."

    # Attempt to parse the CoT response
    if "Think step-by-step:" in response_text:
        parts = response_text.split("Think step-by-step:", 1)
        if len(parts) > 1:
            reasoning_and_diagnosis_part = parts[1].strip()

            if "Final Diagnosis:" in reasoning_and_diagnosis_part:
                reasoning_parts = reasoning_and_diagnosis_part.split("Final Diagnosis:", 1)
                reasoning = reasoning_parts[0].strip()
                diagnosis_and_next_steps_part = reasoning_parts[1].strip()

                if "Suggested next steps:" in diagnosis_and_next_steps_part:
                    diagnosis_parts = diagnosis_and_next_steps_part.split("Suggested next steps:", 1)
                    diagnosis = diagnosis_parts[0].strip()
                    next_steps = diagnosis_parts[1].strip()
                else:
                    diagnosis = diagnosis_and_next_steps_part
            else:
                reasoning = reasoning_and_diagnosis_part
    else:
        # Fallback if the explicit marker is not found, assume the whole response is reasoning/diagnosis
        reasoning = response_text

    return {"reasoning": reasoning, "diagnosis": diagnosis, "next_steps": next_steps}

# app.py content
st.set_page_config(page_title="Medical Diagnosis Assistant (CoT)", layout="wide")
st.title("🧠 Medical Diagnosis Assistant (Chain of Thought)")
st.markdown("Aids healthcare professionals by providing step-by-step reasoning for potential diagnoses.")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

symptoms_input = st.text_area(
    "Describe the patient's symptoms:",
    height=200,
    placeholder="e.g., 'Patient presents with persistent cough for 3 weeks, low-grade fever (100.5°F), and mild shortness of breath. No history of asthma or smoking.'"
)

if st.button("Get Diagnosis"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not symptoms_input:
        st.warning("Please describe the patient's symptoms.")
    else:
        with st.spinner("Thinking step-by-step and generating diagnosis..."):
            try:
                result = get_diagnosis_with_cot(symptoms_input, openai_api_key)

                st.subheader("🧠 AI's Reasoning Process (Chain of Thought):")
                st.info(result.get("reasoning", "No detailed reasoning provided."))

                st.subheader("🩺 Potential Diagnosis:")
                st.success(result.get("diagnosis", "Could not determine a diagnosis."))

                st.subheader("📝 Suggested Next Steps:")
                st.warning(result.get("next_steps", "No specific next steps suggested."))

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please ensure your OpenAI API key is correct and you have sufficient credits.")

st.sidebar.markdown(
    """
    --- 
    **How to use:**
    1. Enter your OpenAI API Key.
    2. Describe patient symptoms.
    3. Click 'Get Diagnosis'.
    The AI will provide a step-by-step reasoning process, a potential diagnosis, and suggested next steps.
    """
)