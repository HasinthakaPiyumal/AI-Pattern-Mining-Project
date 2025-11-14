import os
from dotenv import load_dotenv
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI # Or other LLM providers

# 1. Configuration and Environment Variables
load_dotenv()

# For demonstration, use OpenAI. Replace with your preferred LLM provider (e.g., Google Generative AI)
# Ensure OPENAI_API_KEY is set in your .env file or environment
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# 3. Knowledge Base and Retrieval Augmented Generation (RAG)
# For a real application, this would be populated with extensive medical data.
# Here, we create a small, in-memory vector store for demonstration.
medical_documents = [
    "Symptoms of influenza often include fever, body aches, headache, and fatigue. It is a viral respiratory illness.",
    "Diabetes mellitus is a chronic condition that affects how your body turns food into energy. Common symptoms include increased thirst, frequent urination, and unexplained weight loss.",
    "Hypertension, or high blood pressure, often has no symptoms. Regular monitoring is crucial for diagnosis and management.",
    "Appendicitis typically presents with pain that begins around the navel and then shifts to the lower right abdomen, often accompanied by nausea, vomiting, and loss of appetite.",
    "Migraine headaches are severe, throbbing headaches, often on one side of the head, accompanied by sensitivity to light and sound, and sometimes nausea.",
    "Common cold symptoms include runny nose, sore throat, cough, and congestion. It is generally milder than the flu."
]

# Initialize embedding model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a Chroma vector store (in-memory for this example)
vdb = Chroma.from_texts(
    texts=medical_documents,
    embedding=embeddings,
    collection_name="medical_knowledge"
)

retriever = vdb.as_retriever()

# 2. Core LLM & Reasoning Engine - Chain-of-Thought (CoT) Implementation
# Prompt template for Chain-of-Thought reasoning
cot_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI-powered clinical diagnostic assistant. Your goal is to provide a differential diagnosis and clear reasoning based on patient data and medical knowledge. \n"
         "Follow these steps carefully:"),
        ("human",
         "Patient Data:\n"
         "Symptoms: {symptoms}\n"
         "Medical History: {medical_history}\n"
         "Lab Results: {lab_results}\n\n"
         "Relevant Medical Knowledge:\n"
         "{context}\n\n"
         "**Step 1: Analyze Key Findings**\n"
         "Based on the patient data, identify and list all key symptoms, signs, and relevant historical or lab findings. Explain why each finding is significant.\n\n"
         "**Step 2: Generate Differential Diagnoses**\n"
         "Propose a list of at least 3-5 potential differential diagnoses that could explain the key findings. Briefly justify each one based on the presented evidence and retrieved knowledge.\n\n"
         "**Step 3: Evaluate and Prioritize Diagnoses**\n"
         "For each differential diagnosis, critically evaluate its likelihood. Explain how well it matches or contradicts the patient's data and the provided medical knowledge. Prioritize them from most to least likely.\n\n"
         "**Step 4: Suggest Further Investigations/Management**\n"
         "Based on your prioritized diagnoses, suggest relevant further diagnostic tests (e.g., imaging, specific lab tests) or initial management steps to confirm or rule out the top diagnoses.\n\n"
         "**Step 5: Formulate Most Likely Diagnosis and Summary**\n"
         "State the single most likely diagnosis and provide a concise summary of the reasoning that led to this conclusion, referencing key evidence. Include a confidence level (e.g., High, Medium, Low) in your diagnosis.\n\n"
         "**Step 6: Self-Correction/Verification**\n"
         "Review your entire reasoning process (Steps 1-5). Are there any inconsistencies? Have you considered all critical information? Are there alternative interpretations? Adjust your final diagnosis or reasoning if necessary, explaining any changes made."
        )
    ]
)

# Create the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "symptoms": RunnablePassthrough(), "medical_history": RunnablePassthrough(), "lab_results": RunnablePassthrough()}  # Pass all inputs for context and prompt
    | cot_prompt_template
    | llm
    | StrOutputParser()
)

# 5. User Interface (UI) Layer with Gradio
def diagnose_patient(symptoms: str, medical_history: str, lab_results: str):
    """
    Generates a diagnostic report for a patient using the LLM and RAG.
    """
    # Create a single dictionary to pass all inputs to the rag_chain
    inputs = {
        "symptoms": symptoms,
        "medical_history": medical_history,
        "lab_results": lab_results
    }

    # The rag_chain expects all these keys, but the `context` will be dynamically retrieved.
    # For the CoT prompt, we pass symptoms, medical_history, lab_results directly.

    response = rag_chain.invoke(inputs)
    return response

if __name__ == "__main__":
    # Gradio Interface
    iface = gr.Interface(
        fn=diagnose_patient,
        inputs=[
            gr.Textbox(label="Patient Symptoms", placeholder="e.g., severe abdominal pain, nausea, loss of appetite"),
            gr.Textbox(label="Medical History", placeholder="e.g., no significant medical history, occasional heartburn"),
            gr.Textbox(label="Lab Results", placeholder="e.g., WBC 15,000, CRP elevated"),
        ],
        outputs=gr.Markdown(label="Diagnostic Report"),
        title="AI-Powered Clinical Diagnostic Assistant",
        description="Enter patient details to receive a differential diagnosis and detailed reasoning based on advanced LLM reasoning and medical knowledge. This tool is for informational purposes only and should not replace professional medical advice."
    )

    print("Starting Gradio interface...")
    iface.launch(share=True)
