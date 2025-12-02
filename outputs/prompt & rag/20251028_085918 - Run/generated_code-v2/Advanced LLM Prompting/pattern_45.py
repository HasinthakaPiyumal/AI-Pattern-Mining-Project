import streamlit as st

def simulate_llm_response(prompt):
    if "INITIAL_TRANSLATION_REQUEST" in prompt:
        # Simulate initial translation and question generation
        medical_text = prompt.split("Medical Text:")[1].strip().split("\n")[0].strip()
        initial_translation = f"Initial Spanish translation of '{medical_text}': 'El paciente presenta síntomas confusos.'"
        questions = [
            "Could you clarify what 'síntomas confusos' means specifically? Are they vague, contradictory, or complex?",
            "Is there any specific context for these 'síntomas' (e.g., location, onset, severity)?"
        ]
        return {"initial_translation": initial_translation, "questions": questions}
    elif "FINAL_TRANSLATION_REQUEST" in prompt:
        # Simulate final translation with clarifications
        original_text = prompt.split("Original Text:")[1].split("Initial Translation:")[0].strip()
        human_clarifications = prompt.split("Human Clarifications:")[1].strip().split("\n")
        
        # Simple logic to show clarification was used
        if "vague" in human_clarifications[0].lower():
            final_translation = f"Final Spanish translation of '{original_text}' (clarified): 'El paciente presenta síntomas vagos y de naturaleza incierta.'"
        elif "contradictory" in human_clarifications[0].lower():
            final_translation = f"Final Spanish translation of '{original_text}' (clarified): 'El paciente presenta síntomas contradictorios y complejos.'"
        else:
            final_translation = f"Final Spanish translation of '{original_text}' (clarified based on input): 'El paciente presenta síntomas que requieren más detalles.'"
            
        return {"final_translation": final_translation}
    return {}

st.set_page_config(layout="wide", page_title="Medical Translation Assistant")
st.title("🩺 Medical Translation Assistant (ICP)")
st.markdown("This assistant uses Interactive Chain Prompting to clarify ambiguous medical translations with human input.")

if 'step' not in st.session_state:
    st.session_state.step = 0
if 'medical_text' not in st.session_state:
    st.session_state.medical_text = ""
if 'initial_translation' not in st.session_state:
    st.session_state.initial_translation = ""
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'human_answers' not in st.session_state:
    st.session_state.human_answers = ["", ""]
if 'final_translation' not in st.session_state:
    st.session_state.final_translation = ""

# --- Step 1: Input Medical Text and Get Initial Translation/Questions ---
if st.session_state.step == 0:
    st.header("1. Enter Medical Text for Translation")
    st.session_state.medical_text = st.text_area(
        "Please enter the medical text you need translated (e.g., patient symptoms, medical report excerpt):",
        height=150,
        value=st.session_state.medical_text
    )

    if st.button("Translate and Identify Ambiguities"):
        if st.session_state.medical_text:
            with st.spinner("Generating initial translation and clarifying questions..."):
                llm_prompt = f"INITIAL_TRANSLATION_REQUEST\nMedical Text: {st.session_state.medical_text}\nTranslate this medical text and identify any ambiguities by generating clarifying questions."
                response = simulate_llm_response(llm_prompt)
                st.session_state.initial_translation = response.get("initial_translation", "")
                st.session_state.questions = response.get("questions", [])
            st.session_state.step = 1
            st.experimental_rerun()
        else:
            st.warning("Please enter some medical text to proceed.")

# --- Step 2: Human Clarification ---
if st.session_state.step == 1:
    st.header("2. Review Initial Translation and Provide Clarifications")
    st.markdown("**Original Text:**")
    st.info(st.session_state.medical_text)
    st.markdown("**Initial Translation:**")
    st.success(st.session_state.initial_translation)

    if st.session_state.questions:
        st.markdown("--- \n **The AI has identified potential ambiguities and needs your clarification:**")
        for i, question in enumerate(st.session_state.questions):
            st.session_state.human_answers[i] = st.text_input(f"**Question {i+1}:** {question}", key=f"q_{i}", value=st.session_state.human_answers[i])

        if st.button("Generate Final Translation with Clarifications"):
            with st.spinner("Generating final translation with human input..."):
                human_answers_str = "\n".join(st.session_state.human_answers)
                llm_prompt = (
                    f"FINAL_TRANSLATION_REQUEST\n"
                    f"Original Text: {st.session_state.medical_text}\n"
                    f"Initial Translation: {st.session_state.initial_translation}\n"
                    f"Human Clarifications:\n{human_answers_str}\n"
                    f"Using the above clarifications, refine the translation to be highly accurate and contextually appropriate."
                )
                response = simulate_llm_response(llm_prompt)
                st.session_state.final_translation = response.get("final_translation", "")
            st.session_state.step = 2
            st.experimental_rerun()
    else:
        st.info("No ambiguities identified. Click below to finalize the translation based on the initial translation.")
        if st.button("Finalize Translation (No Clarifications Needed)"):
            st.session_state.final_translation = st.session_state.initial_translation
            st.session_state.step = 2
            st.experimental_rerun()

# --- Step 3: Display Final Translation ---
if st.session_state.step == 2:
    st.header("3. Final Accurate Medical Translation")
    st.markdown("**Original Text:**")
    st.info(st.session_state.medical_text)
    st.markdown("**Final Translation:**")
    st.success(st.session_state.final_translation)

    if st.button("Start New Translation"):
        st.session_state.step = 0
        st.session_state.medical_text = ""
        st.session_state.initial_translation = ""
        st.session_state.questions = []
        st.session_state.human_answers = ["", ""]
        st.session_state.final_translation = ""
        st.experimental_rerun()