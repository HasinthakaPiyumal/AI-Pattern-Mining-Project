import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 1. Simulate Medical Knowledge Base ---
# In a real application, this would be comprehensive medical literature, drug databases, etc.
medical_knowledge = [
    "Symptoms of influenza include fever, cough, sore throat, muscle aches, and fatigue. Treatment often involves rest, fluids, and antiviral medications.",
    "Diabetes is a chronic condition characterized by high blood sugar levels. Type 1 diabetes is an autoimmune disease, while Type 2 is often linked to lifestyle. Management includes diet, exercise, and medication like insulin or metformin.",
    "Hypertension (high blood pressure) can lead to heart disease and stroke. Symptoms are often subtle, but headaches and dizziness can occur. Treatment involves lifestyle changes and medication such as ACE inhibitors or diuretics.",
    "Rare disease: Fibrodysplasia Ossificans Progressiva (FOP) is a very rare genetic disorder where soft tissues progressively turn into bone. There is no cure, but treatments focus on managing symptoms and preventing injury.",
    "Drug interaction: Taking warfarin (an anticoagulant) with ibuprofen (an NSAID) can increase the risk of bleeding.",
    "Drug interaction: Statin medications (for cholesterol) can interact with grapefruit juice, leading to increased drug levels and potential side effects."
]

# Create dummy files for Langchain document loader
for i, doc in enumerate(medical_knowledge):
    with open(f"doc_{i}.txt", "w") as f:
        f.write(doc)

# Load documents
document_paths = [f"doc_{i}.txt" for i in range(len(medical_knowledge))]
loaders = [TextLoader(path) for path in document_paths]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = text_splitter.split_documents(docs)

# --- 2. Initialize Embeddings and Vector Store (Chroma) ---
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunked_docs, embeddings)

# --- 3. Initialize Language Model (LLM) ---
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o") # Using gpt-4o for better reasoning

# --- 4. Create Retrieval-Augmented Generation (RAG) Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # Simple stuffing of all retrieved documents into the prompt
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# --- 5. Medical Diagnostic Function ---
def diagnose_patient(patient_symptoms: str, patient_history: str = "No significant history provided.") -> str:
    # Combine symptoms and history into a single query for the RAG chain
    query = f"Patient presents with the following symptoms: {patient_symptoms}. Patient history: {patient_history}. What are potential diagnoses and recommended treatments, considering relevant medical facts and drug interactions?"
    
    result = qa_chain({"query": query})
    
    response_text = result["result"]
    # Optionally, include source documents for transparency
    # if result.get("source_documents"):
    #     response_text += "\n\n--- Relevant Medical Information ---"
    #     for doc in result["source_documents"]:
    #         response_text += f"\n- {doc.page_content}"
            
    return response_text

# --- 6. Gradio Interface ---
if __name__ == "__main__":
    # Clean up dummy files after use (optional, for a cleaner demo)
    def cleanup_files():
        for path in document_paths:
            if os.path.exists(path):
                os.remove(path)
    
    # Register cleanup to run on exit or after usage
    # import atexit
    # atexit.register(cleanup_files)

    interface = gr.Interface(
        fn=diagnose_patient,
        inputs=[
            gr.Textbox(label="Patient Symptoms (e.g., 'fever, cough, sore throat')"),
            gr.Textbox(label="Patient History (e.g., 'on warfarin for heart condition')", value="No significant history provided.")
        ],
        outputs=gr.Textbox(label="Diagnosis and Treatment Suggestions"),
        title="AI Medical Diagnostic Assistant",
        description="Enter patient symptoms and history to get potential diagnoses and treatment suggestions based on integrated medical knowledge."
    )
    interface.launch()

    # Manual cleanup call if atexit is not used or for testing
    cleanup_files()
