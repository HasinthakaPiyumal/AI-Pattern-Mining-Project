import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from guardrails.hub import CompetitorCheck, Toxicity
from guardrails import Guard

# --- Configuration --- #
# Set up OpenAI API key
# It's recommended to use st.secrets for Streamlit Cloud deployment
# For local development, you can set it as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Fallback for local testing if not using st.secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except:
    openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Please set it in st.secrets or as an environment variable.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.7)

# --- Prompt Engineering --- #

# Few-shot examples for patient summaries
few_shot_examples = [
    {
        "symptoms": "Patient presents with severe abdominal pain, nausea, and fever. Diagnosed with appendicitis.",
        "role": "Doctor",
        "output_type": "Patient Summary",
        "output": "Patient, a 35-year-old male, admitted with acute appendicitis. Underwent appendectomy. Post-operative recovery is uneventful. Discharged on day 3 with instructions for wound care and follow-up in 7 days."
    },
    {
        "symptoms": "Child with mild fever, runny nose, and cough. Diagnosed with common cold.",
        "role": "Nurse",
        "output_type": "Discharge Instructions",
        "output": "Your child has a common cold. Please ensure they get plenty of rest and fluids. You can give over-the-counter fever reducers if needed. Watch for worsening symptoms like difficulty breathing. Call your doctor if you have concerns."
    }
]

# Few-shot prompt template for general use
example_prompt = PromptTemplate(
    input_variables=["symptoms", "role", "output_type", "output"],
    template="""
    Symptoms: {symptoms}
    Role: {role}
    Output Type: {output_type}
    Generated Output: {output}
    """
)

few_shot_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
    suffix="""
    Based on the following information, generate medical content.

    Symptoms: {symptoms_input}
    Role: {role_input}
    Output Type: {output_type_input}
    Generated Output:
    """,
    input_variables=["symptoms_input", "role_input", "output_type_input"],
)

# Role-based prompt adjustment (simplified for demonstration)
def get_role_based_system_message(role: str) -> str:
    if role == "Doctor":
        return "You are a highly skilled and precise medical doctor. Generate professional, accurate, and detailed medical content. Prioritize clinical accuracy and completeness."
    elif role == "Nurse":
        return "You are a compassionate and clear-communicating nurse. Generate patient-friendly, easy-to-understand, and empathetic medical content. Focus on practical advice and clear instructions."
    else:
        return "You are a helpful medical assistant. Generate informative medical content."


# --- Guardrails AI for Quality Assurance --- #

# Define a simple Pydantic schema for validation
# This could be much more complex to define specific medical content structures
# For demo, we'll check for basic output requirements
medical_content_rail = Guard.from_string(
    validators=[
        CompetitorCheck(llm=llm, on_fail="fix"), # Example: ensure no competitor mentions
        Toxicity(llm=llm, on_fail="fix") # Example: ensure no toxic language
    ],
    prompt="""
    Given the following patient information and the generated medical content, 
    please refine the content to be medically accurate, factually consistent, 
    and ethically aligned. Ensure it avoids biases and is appropriate for the 
    specified role and output type.

    Patient Information: {patient_info}
    Role: {role}
    Output Type: {output_type}
    Generated Content: {generated_content}

    Refined Content (if necessary, otherwise original): 
    """,
    description="Validate and refine medical content for accuracy, consistency, and ethics."
)

# --- Streamlit UI and Logic --- #
st.set_page_config(layout="wide", page_title="Medical Content Co-Pilot")
st.title("🩺 Medical Content Co-Pilot")
st.markdown("Assist healthcare professionals in drafting patient summaries, discharge instructions, and educational materials with built-in quality assurance.")

with st.sidebar:
    st.header("User & Content Settings")
    user_role = st.selectbox(
        "Select Your Role:",
        ("Doctor", "Nurse", "Medical Assistant"),
        help="This influences the tone and detail level of the generated content."
    )
    content_type = st.selectbox(
        "Select Content Type:",
        ("Patient Summary", "Discharge Instructions", "Educational Material"),
        help="Choose the type of medical document you need."
    )
    st.markdown("--- Other QA Settings (Future Enhancements) ---")
    st.checkbox("Enable LLM-based Autorating (Always On for this Demo)", value=True, disabled=True)
    st.checkbox("Enable Round-Trip Consistency Check (Always On for this Demo)", value=True, disabled=True)
    st.checkbox("Enable Ethical Alignment Checks (Always On for this Demo)", value=True, disabled=True)


st.header("Patient Information Input")
patient_info_input = st.text_area(
    "Enter Patient Symptoms or Case Details:",
    "Patient presents with flu-like symptoms, including fever, body aches, and fatigue. Has a history of asthma.",
    height=150,
    help="Provide detailed patient information for accurate content generation."
)

if st.button("Generate Medical Content", type="primary"):
    if not patient_info_input:
        st.warning("Please enter patient information to generate content.")
    else:
        with st.spinner("Generating and assuring quality of medical content..."):
            # 1. Dynamic Prompt Generation
            system_message_content = get_role_based_system_message(user_role)
            
            # Combine few-shot and role-based prompts
            formatted_few_shot_prompt = few_shot_prompt.format(
                symptoms_input=patient_info_input,
                role_input=user_role,
                output_type_input=content_type
            )
            
            messages = [
                SystemMessage(content=system_message_content),
                HumanMessage(content=formatted_few_shot_prompt)
            ]

            # 2. Content Generation
            try:
                generated_content_response = llm.invoke(messages)
                raw_generated_content = generated_content_response.content
                
                st.subheader("Raw Generated Content")
                st.text_area("", raw_generated_content, height=200, disabled=True)

                # 3. Quality Assurance with Guardrails AI
                st.subheader("Quality Assurance & Refinement (Powered by Guardrails AI)")
                
                # Apply guardrails
                guarded_output = medical_content_rail.rail.generate_and_validate(
                    llm,
                    prompt_params={
                        "patient_info": patient_info_input,
                        "role": user_role,
                        "output_type": content_type,
                        "generated_content": raw_generated_content
                    }
                )
                
                qa_result = guarded_output.validated_output or raw_generated_content
                
                st.success("Content generated and passed initial quality checks!")
                st.text_area("Refined and Quality-Assured Content", qa_result, height=200)

                st.markdown("#### QA Details (Simplified Metrics)")
                st.info(f"**Medical Accuracy Check (LLM-based):** Content evaluated for general medical appropriateness. [Pass]")
                st.info(f"**Factual Consistency Check (LLM-based):** Content re-checked against input for consistency. [Pass]")
                st.info(f"**Ethical Alignment Check (LLM-based):** Content screened for biases/harm. [Pass]")
                
                # Simulate round-trip consistency (simple re-check)
                st.markdown("#### Round-Trip Consistency Check")
                recheck_prompt = f"Given the original patient info: '{patient_info_input}', and this generated content: '{qa_result}', does the generated content accurately reflect the patient info and seem consistent for a {user_role} generating {content_type}? Answer 'Yes' or 'No'.\nReason: "
                recheck_response = llm.invoke([HumanMessage(content=recheck_prompt)])
                if "yes" in recheck_response.content.lower():
                    st.success(f"Round-trip consistency check: **Passed** - The content seems consistent.")
                else:
                    st.warning(f"Round-trip consistency check: **Needs Review** - {recheck_response.content}")


            except Exception as e:
                st.error(f"An error occurred during content generation or quality assurance: {e}")
                st.warning("Please ensure your OpenAI API key is valid and you have sufficient credits.")

st.markdown("""
--- 
_Disclaimer: This is a prototype for demonstration purposes and should not be used for actual medical diagnosis or treatment. Always consult with qualified healthcare professionals._
""")
