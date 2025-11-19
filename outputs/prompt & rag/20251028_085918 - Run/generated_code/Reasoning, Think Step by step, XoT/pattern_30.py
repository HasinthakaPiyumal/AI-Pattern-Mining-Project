import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

# Placeholder for API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 1. LLM and Embedding Model Initialization
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
embeddings = OpenAIEmbeddings()

# 2. Medical Knowledge Base (Simplified in-memory ChromaDB for demonstration)
# In a real application, this would be loaded from persistent storage.
docs = [
    "Symptoms of influenza often include fever, cough, sore throat, and body aches.",
    "A common treatment for strep throat is antibiotics like penicillin or amoxicillin.",
    "Diabetes mellitus type 2 is characterized by insulin resistance and relative insulin deficiency.",
    "High blood pressure (hypertension) increases the risk of heart disease and stroke.",
    "Migraine headaches are typically severe, throbbing pain, usually on one side of the head, and can be accompanied by nausea and light sensitivity.",
    "Appendicitis usually presents with pain starting around the navel and moving to the lower right abdomen."
]
vectorstore = Chroma.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever()

# 3. Prompt Templates
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if necessary and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = (
    "You are a highly experienced Medical Diagnostic AI Assistant. "
    "Your goal is to provide a comprehensive diagnosis and reasoning based on the patient's symptoms, medical history, and lab results. "
    "Always provide a step-by-step reasoning process (Chain-of-Thought) to arrive at the diagnosis. "
    "Break down complex problems into subtasks (Decomposed Prompting). "
    "Consider multiple diagnostic paths (Tree-of-Thoughts concept by exploring alternatives if initial path is weak). "
    "Reference the provided medical context to support your reasoning and diagnosis. "
    "If you are unsure, state the uncertainty and suggest further tests or specialist consultation. "
    "Final Answer should include a 'Diagnosis:' and 'Reasoning:' section."+
    "\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# 4. Rule-Based Verifier (Simplified)
def rule_based_verifier(diagnosis: str, symptoms: str) -> bool:
    if "appendicitis" in diagnosis.lower() and "lower right abdomen pain" not in symptoms.lower():
        return False, "Rule violation: Appendicitis usually involves lower right abdomen pain."
    if "strep throat" in diagnosis.lower() and "antibiotics" not in diagnosis.lower() and "treatment" in diagnosis.lower():
        return False, "Rule violation: Strep throat treatment should mention antibiotics."
    return True, "No obvious rule violations found."

# 5. LLM-based Self-Correction/Verifier
def llm_verifier(llm_model, reasoning: str, diagnosis: str, symptoms: str, medical_context: str) -> str:
    verifier_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical reasoning verifier. Evaluate the following diagnosis and reasoning for accuracy, consistency, and completeness based on the provided symptoms and medical context. Identify any potential hallucinations or logical flaws. If there are issues, suggest improvements. Provide a confidence score (0-100)."),
        ("human", f"Symptoms: {symptoms}\nMedical Context: {medical_context}\nProposed Diagnosis: {diagnosis}\nProposed Reasoning: {reasoning}\n\nIs this diagnosis and reasoning sound? Provide feedback and a confidence score.")
    ])
    verifier_chain = verifier_prompt | llm_model | StrOutputParser()
    response = verifier_chain.invoke({"symptoms": symptoms, "medical_context": medical_context, "diagnosis": diagnosis, "reasoning": reasoning})
    return response

# 6. Main Diagnostic Chain
def create_diagnostic_chain(llm_model, retriever_model):
    rag_chain = create_retrieval_chain(retriever_model, qa_prompt)
    return rag_chain

full_diagnostic_chain = create_diagnostic_chain(llm, history_aware_retriever)

# 7. Streamlit UI
st.set_page_config(page_title="Medical Diagnostic Assistant", layout="wide")
st.title("🩺 Medical Diagnostic Assistant with Verified Reasoning")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("About")
    st.write("This assistant uses advanced AI reasoning patterns (Chain-of-Thought, Decomposed Prompting, Tree-of-Thoughts) and verification mechanisms to provide structured and verified diagnostic support.")
    st.write("Disclaimer: This tool is for informational purposes only and should not replace professional medical advice.")

user_question = st.text_area("Enter patient symptoms, medical history, or lab results:", height=150)
submit_button = st.button("Get Diagnosis")

if submit_button and user_question:
    with st.spinner("Generating diagnosis and reasoning..."):
        response = full_diagnostic_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_question
        })
        
        generated_answer = response["answer"]
        retrieved_context = "\n".join([doc.page_content for doc in response["context"]])

        st.subheader("Generated Diagnosis and Reasoning:")
        st.write(generated_answer)

        # Extract diagnosis and reasoning for verification (simplified parsing)
        diagnosis_match = "Diagnosis:"
        reasoning_match = "Reasoning:"
        
        diagnosis_start = generated_answer.find(diagnosis_match)
        reasoning_start = generated_answer.find(reasoning_match)

        extracted_diagnosis = "N/A"
        extracted_reasoning = "N/A"

        if diagnosis_start != -1 and reasoning_start != -1:
            if diagnosis_start < reasoning_start:
                extracted_diagnosis = generated_answer[diagnosis_start + len(diagnosis_match):reasoning_start].strip()
                extracted_reasoning = generated_answer[reasoning_start + len(reasoning_match):].strip()
            else:
                extracted_reasoning = generated_answer[reasoning_start + len(reasoning_match):diagnosis_start].strip()
                extracted_diagnosis = generated_answer[diagnosis_start + len(diagnosis_match):].strip()
        elif diagnosis_start != -1:
             extracted_diagnosis = generated_answer[diagnosis_start + len(diagnosis_match):].strip()
        elif reasoning_start != -1:
             extracted_reasoning = generated_answer[reasoning_start + len(reasoning_match):].strip()

        st.subheader("Verification Steps:")
        
        # Rule-based verification
        is_valid_rule, rule_feedback = rule_based_verifier(extracted_diagnosis, user_question)
        if is_valid_rule:
            st.success(f"✅ Rule-Based Verification: {rule_feedback}")
        else:
            st.warning(f"⚠️ Rule-Based Verification: {rule_feedback}")

        # LLM-based verification
        llm_verify_feedback = llm_verifier(llm, extracted_reasoning, extracted_diagnosis, user_question, retrieved_context)
        st.info(f"🧠 LLM-Based Verifier Feedback:\n{llm_verify_feedback}")

        st.session_state.chat_history.extend([
            HumanMessage(content=user_question),
            AIMessage(content=generated_answer)
        ])

        st.subheader("Chat History")
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.write(f"**You:** {message.content}")
            elif isinstance(message, AIMessage):
                st.write(f"**Assistant:** {message.content}")

        st.subheader("Retrieved Medical Context:")
        st.expander("Show context").write(retrieved_context)

