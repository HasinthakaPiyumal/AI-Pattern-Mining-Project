import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# --- Environment Setup ---
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Replace with actual key or set as env var

# --- Medical Knowledge Base (Simulated) ---
# In a real application, this would be loaded from external sources
def load_medical_knowledge():
    docs = [
        "Symptoms of common cold include runny nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, low-grade fever, and a general feeling of being unwell (malaise). Treatment often involves rest, fluids, and over-the-counter medications.",
        "Influenza (flu) symptoms are similar to a cold but often more severe and come on suddenly. They include fever or feeling feverish/chills, cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue. Antiviral drugs can be prescribed.",
        "Strep throat is a bacterial infection that can cause a sore throat, fever, tiny red spots on the roof of the mouth, and swollen tonsils, sometimes with white patches. It is treated with antibiotics like penicillin or amoxicillin.",
        "Allergic reactions can cause symptoms like sneezing, runny nose, itchy eyes, skin rashes (hives), and sometimes difficulty breathing or swelling. Antihistamines are commonly used.",
        "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm or pus, fever, chills, and difficulty breathing. Treatment depends on the cause (bacterial, viral, fungal) and may involve antibiotics.",
        "Migraines are severe headaches often accompanied by throbbing pain, sensitivity to light and sound, nausea, and vomiting. Triptans and pain relievers are common treatments.",
        "Appendicitis symptoms include sudden pain that begins on the right side of the lower abdomen, sudden pain that begins around your navel and shifts to your lower right abdomen, nausea and vomiting, loss of appetite, fever. Requires surgery."
    ]
    return docs

# Initialize Embedding Model and ChromaDB
medical_docs = load_medical_knowledge()
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts=medical_docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# --- LLMs --- 
llm_primary = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
llm_verifier = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3) # A lighter model for verification

# --- LangChain Chains --- 

# 1. Core Reasoning Chain (Chain-of-Thought & Problem Decomposition)
reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly analytical medical diagnostic assistant. Your task is to provide a preliminary diagnosis and explain your reasoning step-by-step. Break down the problem, consider different possibilities, and integrate provided medical knowledge."),
    ("user", "Patient Symptoms: {symptoms}\nMedical History: {history}\nRelevant Medical Knowledge: {context}\n\nBased on this information, first, list potential conditions with brief justifications. Then, analyze each condition against the symptoms and history, explaining why it is or isn't a strong candidate. Finally, provide your most likely diagnosis and initial treatment recommendations. Think step-by-step.")
])

reasoning_chain = (
    RunnablePassthrough.assign(context=retriever)
    | reasoning_prompt
    | llm_primary
    | StrOutputParser()
)

# 2. Tree-of-Thoughts (Simplified: Generate multiple hypotheses for initial consideration)
# This is a simplification. A true ToT would involve branching and self-evaluation.
hypothesis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical brainstorming assistant. Given patient symptoms and history, propose 2-3 distinct potential diagnoses that a doctor might consider, along with a very brief (1-2 sentence) reason for each. These are initial thoughts, not final diagnoses."),
    ("user", "Patient Symptoms: {symptoms}\nMedical History: {history}\n\nPropose 2-3 distinct potential diagnoses with brief reasons.")
])

hypothesis_chain = (
    hypothesis_prompt
    | llm_primary
    | StrOutputParser()
)

# 3. Verification Chain (Chain-of-Verification)
verification_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical verifier. Your role is to critically evaluate a diagnostic reasoning process and its conclusion. Check for factual accuracy against the provided medical knowledge, logical consistency, and completeness. Point out any discrepancies or areas for improvement. If the diagnosis seems correct and well-justified, state that."),
    ("user", "Proposed Diagnosis and Reasoning: {reasoning}\n\nPatient Symptoms: {symptoms}\nMedical History: {history}\nRelevant Medical Knowledge (for cross-referencing): {context}\n\nCritique the proposed diagnosis and reasoning. Is it accurate, consistent, and well-supported by the provided knowledge and patient information? What are its strengths and weaknesses? Provide a concise verification statement.")
])

verification_chain = (
    RunnablePassthrough.assign(context=retriever)
    | verification_prompt
    | llm_verifier
    | StrOutputParser()
)

# --- Streamlit UI --- 
st.set_page_config(page_title="MediDiag-VR: Verified Medical Assistant", layout="wide")
st.title("🩺 MediDiag-VR: Verified Medical Diagnostic Assistant")
st.markdown("Input patient information below to receive a preliminary diagnosis with structured and verified reasoning.")

with st.sidebar:
    st.header("About")
    st.info("This application demonstrates the Structured and Verified Reasoning (SVR) pattern for medical diagnosis. It uses LangChain, OpenAI LLMs, and ChromaDB for knowledge retrieval, offering step-by-step reasoning and a verification step to enhance accuracy and transparency.")
    st.warning("Disclaimer: This AI tool is for informational and demonstrative purposes only and should NOT be used for actual medical diagnosis or treatment. Always consult with a qualified medical professional.")

symptoms = st.text_area("Enter Patient Symptoms (e.g., 'Sudden onset of severe headache, sensitivity to light, nausea, throbbing pain on one side of the head')", height=150)
medical_history = st.text_area("Enter Patient Medical History (e.g., 'No significant past medical history, occasionally suffers from tension headaches')", height=100)

if st.button("Get Diagnosis and Verified Reasoning"): # Pass symptoms and history here
    if not symptoms:
        st.error("Please enter patient symptoms to proceed.")
    else:
        with st.spinner("Generating initial hypotheses..."):
            # Generate initial hypotheses (simplified ToT)
            hypotheses = hypothesis_chain.invoke({"symptoms": symptoms, "history": medical_history})
            st.subheader("🔬 Initial Diagnostic Hypotheses")
            st.write(hypotheses)
        
        st.write("\n---")
        with st.spinner("Generating structured reasoning and diagnosis..."):
            # Core Reasoning
            reasoning_output = reasoning_chain.invoke({"symptoms": symptoms, "history": medical_history})
            st.subheader("🧠 Proposed Diagnosis & Reasoning")
            st.markdown(reasoning_output)

        st.write("\n---")
        with st.spinner("Verifying diagnosis and reasoning..."):
            # Verification
            verification_output = verification_chain.invoke({"reasoning": reasoning_output, "symptoms": symptoms, "history": medical_history})
            st.subheader("✅ Verification Report")
            st.markdown(verification_output)

        st.success("Diagnosis and Verification Complete!")
