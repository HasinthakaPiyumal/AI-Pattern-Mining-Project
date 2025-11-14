import os
import random
import time
from typing import List, Dict

# Placeholder for real LLM and embedding models (replace with actual model loading)
# For demonstration, we'll use a simple placeholder or a small local model if possible.

# Try to import necessary libraries, provide stubs if not available for code generation.
try:
    from transformers import pipeline
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    import gradio as gr
except ImportError:
    print("One or more required libraries not found. Please install them:")
    print("pip install transformers langchain-community chromadb sentence-transformers gradio")
    # Define stubs for the functions/classes to avoid NameError during code generation
    class pipeline:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return [{"generated_text": "Placeholder LLM response."}]
    class HuggingFacePipeline:
        def __init__(self, *args, **kwargs): pass
        def invoke(self, *args, **kwargs): return "Placeholder LLM response."
    class HuggingFaceEmbeddings:
        def __init__(self, *args, **kwargs): pass
        def embed_documents(self, texts: List[str]) -> List[List[float]]: return [[0.0]*768 for _ in texts]
        def embed_query(self, text: str) -> List[float]: return [0.0]*768
    class Chroma:
        def __init__(self, *args, **kwargs): pass
        def add_documents(self, documents): pass
        def as_retriever(self, *args, **kwargs): return type('Retriever', (object,), {'get_relevant_documents': lambda self, query: [{'page_content': 'Placeholder retrieved document.'}]})()
    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs): pass
        def split_documents(self, documents): return documents # Simplified for stub
        def create_documents(self, texts: List[str]): return [{'page_content': t} for t in texts]
    class RetrievalQA:
        def __init__(self, *args, **kwargs): pass
        def invoke(self, query: str) -> Dict: return {"result": "Placeholder RAG response."}
    class PromptTemplate:
        def __init__(self, *args, **kwargs): pass
        def format(self, *args, **kwargs): return "Placeholder prompt."
    class gr:
        Blocks = type('Blocks', (object,), {})
        Markdown = type('Markdown', (object,), {})
        Textbox = type('Textbox', (object,), {})
        Button = type('Button', (object,), {})
        Interface = type('Interface', (object,), {'launch': lambda *args, **kwargs: None})
        ChatInterface = type('ChatInterface', (object,), {'launch': lambda *args, **kwargs: None})

# --- Configuration ---
# You might want to use environment variables for actual models/APIs
# os.environ["HF_HOME"] = "./hf_cache" # Uncomment to set a cache directory for HuggingFace models

# Define a small LLM for local demonstration if a powerful one is not available
# For production, consider larger models, e.g., 'mistralai/Mistral-7B-Instruct-v0.2'
GENERATION_LLM_MODEL = "distilgpt2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"

# --- Simulated External Data Sources ---

def get_simulated_ehr_data(patient_id: str) -> Dict:
    """Simulates fetching Electronic Health Record (EHR) data for a patient."""
    print(f"Fetching EHR for patient: {patient_id}")
    ehr_data = {
        "P1001": {
            "name": "Alice Smith",
            "age": 45,
            "gender": "Female",
            "past_medical_history": [
                "Hypertension (diagnosed 5 years ago)",
                "Type 2 Diabetes (diagnosed 2 years ago, controlled with Metformin)",
                "Seasonal allergies"
            ],
            "current_medications": ["Lisinopril 10mg", "Metformin 500mg"],
            "allergies": ["Penicillin"],
            "recent_visits": [
                {"date": "2023-10-15", "reason": "Routine check-up", "findings": "Blood pressure elevated (145/90)"},
                {"date": "2024-01-20", "reason": "Persistent cough", "findings": "Mild bronchitis, prescribed cough syrup"}
            ]
        },
        "P1002": {
            "name": "Bob Johnson",
            "age": 60,
            "gender": "Male",
            "past_medical_history": [
                "Coronary Artery Disease (CABG 10 years ago)",
                "Hyperlipidemia"
            ],
            "current_medications": ["Aspirin 81mg", "Atorvastatin 20mg", "Metoprolol 25mg"],
            "allergies": [],
            "recent_visits": [
                {"date": "2023-11-01", "reason": "Chest pain evaluation", "findings": "Stable angina, advised lifestyle changes"},
                {"date": "2024-02-10", "reason": "Follow-up", "findings": "Blood work stable"}
            ]
        }
    }
    return ehr_data.get(patient_id, {"error": "Patient not found. This is simulated data."})

def get_simulated_medical_literature(query: str) -> List[str]:
    """Simulates fetching relevant medical literature based on a query."""
    print(f"Searching medical literature for: '{query}'")
    # In a real system, this would hit PubMed, research databases, etc.
    # For simulation, we have some hardcoded "literature snippets".
    literature_snippets = {
        "lupus nephritis": [
            "Lupus nephritis is a severe complication of systemic lupus erythematosus (SLE) characterized by inflammation of the kidneys.",
            "Diagnosis typically involves kidney biopsy and serological tests (e.g., anti-dsDNA, ANA).",
            "Treatment often includes corticosteroids and immunosuppressants like cyclophosphamide or mycophenolate mofetil.",
            "Early diagnosis and aggressive treatment are crucial to prevent progression to end-stage renal disease."
        ],
        "pulmonary fibrosis": [
            "Idiopathic pulmonary fibrosis (IPF) is a chronic, progressive lung disease characterized by the scarring of lung tissue.",
            "Symptoms include shortness of breath, dry cough, and clubbing of the fingers.",
            "Diagnosis involves high-resolution CT scans, lung function tests, and sometimes surgical lung biopsy.",
            "Antifibrotic drugs such as pirfenidone and nintedanib can slow disease progression."
        ],
        "glioblastoma treatment": [
            "Glioblastoma (GBM) is the most aggressive type of primary brain tumor.",
            "Standard treatment involves surgery, radiation therapy, and chemotherapy with temozolomide.",
            "Newer therapies, including tumor treating fields (TTFields) and targeted therapies, are being investigated.",
            "Prognosis for GBM remains poor, highlighting the need for novel therapeutic strategies."
        ],
        "hypertension management": [
            "Lifestyle modifications (diet, exercise, weight loss) are foundational for hypertension management.",
            "Common antihypertensive medications include ACE inhibitors, ARBs, diuretics, beta-blockers, and calcium channel blockers.",
            "Regular blood pressure monitoring is essential.",
            "Personalized treatment plans are crucial, considering patient comorbidities and risk factors."
        ]
    }
    # Simple keyword matching for simulation
    found_literature = []
    for keyword, snippets in literature_snippets.items():
        if keyword in query.lower():
            found_literature.extend(snippets)
    if not found_literature:
        # Fallback for general queries
        if "disease" in query.lower() or "diagnosis" in query.lower():
            return ["Medical research continues to uncover new insights into various diseases.", "Consulting specialized medical databases can provide detailed diagnostic criteria and treatment protocols."]
    return found_literature if found_literature else ["No specific literature found for this exact query in our simulated database."]

def get_simulated_medical_database_entry(condition_or_drug: str) -> List[str]:
    """Simulates querying a specialized medical database (e.g., for drug info, rare disease profiles)."""
    print(f"Querying medical database for: '{condition_or_drug}'")
    db_entries = {
        "myasthenia gravis": [
            "Myasthenia gravis is a chronic autoimmune neuromuscular disease characterized by varying degrees of weakness of the skeletal (voluntary) muscles.",
            "It is caused by a breakdown in the normal communication between nerves and muscles.",
            "Symptoms include weakness in eye muscles, eyelids, facial expression, chewing, talking, and swallowing.",
            "Diagnosis involves neurological exam, blood tests for antibodies, and electrodiagnostic tests (EMG).",
            "Treatment includes anticholinesterase agents, corticosteroids, immunosuppressants, IVIG, and plasma exchange."
        ],
        "hemophilia a": [
            "Hemophilia A is a genetic disorder caused by a deficiency in clotting factor VIII, leading to prolonged bleeding.",
            "It primarily affects males and is typically inherited in an X-linked recessive pattern.",
            "Symptoms include spontaneous bleeding, easy bruising, and prolonged bleeding after injury or surgery.",
            "Treatment involves replacement therapy with factor VIII concentrate, either on demand or as prophylaxis."
        ],
        "temozolomide": [
            "Temozolomide is an oral chemotherapy drug used to treat certain types of brain tumors, including glioblastoma.",
            "It works by damaging the DNA of cancer cells, leading to their death.",
            "Common side effects include nausea, vomiting, fatigue, hair loss, and myelosuppression (reduced blood cell counts).",
            "It is often given in conjunction with radiation therapy."
        ]
    }
    return db_entries.get(condition_or_drug.lower(), [f"No detailed entry found for '{condition_or_drug}' in simulated database."])

# --- LLM and Embeddings Setup ---

# Initialize embedding model
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize LLM for text generation
print(f"Loading LLM for generation: {GENERATION_LLM_MODEL}")
# Using a small pre-trained model suitable for text generation for local execution
# You might need to adjust 'max_new_tokens' based on the model and desired output length
try:
    llm_pipeline = pipeline(
        "text-generation",
        model=GENERATION_LLM_MODEL,
        torch_dtype="auto",
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") else -1, # Use GPU if available
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        num_return_sequences=1,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
except Exception as e:
    print(f"Could not load HuggingFacePipeline for {GENERATION_LLM_MODEL}: {e}")
    print("Falling back to a placeholder LLM for demonstration.")
    class PlaceholderLLM:
        def invoke(self, prompt: str) -> str:
            if "diagnose" in prompt.lower():
                return "Based on the provided information, a potential diagnosis could be [Disease Name]. Further investigation with [Test] is recommended."
            elif "treatment" in prompt.lower():
                return "For the identified condition, treatment options include [Treatment A] and [Treatment B], along with supportive care."
            return "As an AI assistant, I can provide information, but a medical professional should always be consulted for diagnosis and treatment."
    llm = PlaceholderLLM()

# --- ChromaDB and RAG Setup ---

# Initialize ChromaDB persistent client
print(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

def populate_knowledge_base():
    """Populates the vector store with initial medical knowledge."""
    print("Populating initial medical knowledge base...")
    # Example documents for the vector store
    initial_docs = [
        "Systemic lupus erythematosus (SLE) is a chronic autoimmune disease that can affect various organs, including the kidneys (lupus nephritis), joints, skin, and brain.",
        "Symptoms of SLE are diverse and can include fatigue, joint pain, skin rashes (e.g., butterfly rash), and fever.",
        "Diagnosis of SLE involves a combination of clinical criteria and laboratory tests, such as ANA, anti-dsDNA, and anti-Sm antibodies.",
        "Treatment for SLE aims to control symptoms and prevent organ damage, often involving anti-malarials (hydroxychloroquine), corticosteroids, and immunosuppressants.",
        "Rheumatoid arthritis (RA) is a chronic inflammatory disorder affecting many joints, including those in the hands and feet. It causes painful swelling that can eventually result in bone erosion and joint deformity.",
        "Psoriasis is a common, chronic, noncommunicious disease characterized by well-demarcated erythematous plaques with silvery scales.",
        "Type 1 Diabetes Mellitus is an autoimmune condition in which the the body's immune system mistakenly attacks and destroys the insulin-producing beta cells in the pancreas.",
        "Type 2 Diabetes Mellitus is a chronic condition that affects the way the body processes blood sugar (glucose).",
        "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
        "A myocardial infarction (heart attack) occurs when blood flow to a part of your heart is blocked for a long enough time that part of the heart muscle is damaged or dies.",
        "Stroke occurs when the blood supply to part of your brain is interrupted or severely reduced, depriving brain tissue of oxygen and nutrients."
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents(initial_docs)
    vectorstore.add_documents(docs)
    print(f"Added {len(docs)} documents to the knowledge base.")

    # Add some simulated "new research findings"
    new_findings = [
        "Recent study suggests a novel biomarker for early detection of Alzheimer's disease via cerebrospinal fluid analysis.",
        "Clinical trials are exploring new gene therapies for Duchenne Muscular Dystrophy, showing promising early results in animal models.",
        "Updated guidelines for managing severe sepsis recommend early administration of broad-spectrum antibiotics and fluid resuscitation."
    ]
    new_docs = text_splitter.create_documents(new_findings)
    vectorstore.add_documents(new_docs)
    print(f"Added {len(new_docs)} new findings to the knowledge base.")

    print("Knowledge base population complete.")

# Ensure the ChromaDB is populated on startup
# It's better to check if it's already populated to avoid re-adding on every run.
# For this example, we'll just populate it.
try:
    # Attempt to retrieve a document to check if the store is not empty
    if not vectorstore.get()['ids']:
        populate_knowledge_base()
    else:
        print("ChromaDB already contains data. Skipping initial population.")
except Exception as e:
    print(f"Error checking ChromaDB or it's empty: {e}. Populating knowledge base.")
    populate_knowledge_base()

# Set up the RAG chain
# We define a custom prompt template to guide the LLM
QA_CHAIN_PROMPT = PromptTemplate.from_template(
    """You are a highly knowledgeable medical diagnostic assistant. Your goal is to help doctors by providing comprehensive and accurate information based on the context provided.
    Use the following pieces of retrieved context and patient history to answer the question.
    If you don't know the answer, just say that you don't have enough information, don't try to make up an answer.

    Patient History:
    {patient_history}

    Medical Context:
    {context}

    Question: {question}

    Detailed Answer:"""
)

# Create a retriever for the vector store
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}) # MMR for diversity

# Create the RAG chain
# We'll use RetrievalQA.from_chain_type for simplicity in this example.
# For more complex flows, LCEL (LangChain Expression Language) is recommended.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'stuff' means cram all retrieved documents into the prompt
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# --- Medical Diagnostic Assistant Class ---

class MedicalDiagnosticAssistant:
    def __init__(self, qa_chain, patient_data_fetcher):
        self.qa_chain = qa_chain
        self.patient_data_fetcher = patient_data_fetcher

    def diagnose_and_assist(self, patient_id: str, symptoms: str, additional_info: str = "") -> str:
        """
        Assists in diagnosis by combining patient data, external medical knowledge, and LLM reasoning.
        """
        print(f"\n--- Processing Request for Patient ID: {patient_id} ---")

        # 1. Retrieve patient historical data
        ehr_data = self.patient_data_fetcher(patient_id)
        if "error" in ehr_data:
            patient_history_str = f"Patient ID: {patient_id}. Error: {ehr_data['error']}"
            print(patient_history_str)
        else:
            patient_history_str = f"Patient Name: {ehr_data.get('name', 'N/A')}, Age: {ehr_data.get('age', 'N/A')}, Gender: {ehr_data.get('gender', 'N/A')}\n" \
                                  f"Past Medical History: {', '.join(ehr_data.get('past_medical_history', []))}\n" \
                                  f"Current Medications: {', '.join(ehr_data.get('current_medications', []))}\n" \
                                  f"Allergies: {', '.join(ehr_data.get('allergies', []))}\n" \
                                  f"Recent Visits: {ehr_data.get('recent_visits', 'N/A')}"
            print("Patient History retrieved.")

        # 2. Augment query with symptoms and additional info for more relevant retrieval
        full_query = f"Patient symptoms: {symptoms}. Additional context: {additional_info}. " \
                     f"Considering patient history: {patient_history_str}. What is the most likely diagnosis and recommended next steps?"

        # 3. Perform RAG query
        print("Invoking RAG chain...")
        result = self.qa_chain.invoke({"query": full_query, "patient_history": patient_history_str})
        response_text = result["result"]
        source_documents = result.get("source_documents", [])

        # 4. Integrate real-time literature/database if specific keywords are detected
        # This part demonstrates tool integration on top of RAG
        # We can analyze the LLM's initial response or the user's symptoms for keywords
        # For simplicity, let's just check for specific keywords in symptoms
        potential_conditions = []
        if "weakness" in symptoms.lower() and "eyes" in symptoms.lower():
            potential_conditions.append("myasthenia gravis")
        if "bleeding" in symptoms.lower() and "joint" in symptoms.lower():
            potential_conditions.append("hemophilia a")
        if "kidney" in symptoms.lower() and ("lupus" in symptoms.lower() or "rash" in symptoms.lower()):
            potential_conditions.append("lupus nephritis")
        if "cough" in symptoms.lower() and "shortness of breath" in symptoms.lower():
             potential_conditions.append("pulmonary fibrosis")

        for condition in potential_conditions:
            literature = get_simulated_medical_literature(condition)
            db_entry = get_simulated_medical_database_entry(condition)
            if literature:
                response_text += f"\n\n--- Real-time Literature for '{condition}' ---\n" + "\n".join(literature)
            if db_entry:
                response_text += f"\n\n--- Medical Database for '{condition}' ---\n" + "\n".join(db_entry)
            print(f"Integrated real-time data for: {condition}")

        # 5. Format the final output
        formatted_output = f"**Medical Diagnostic Assistant Report**\n\n" \
                           f"**Patient ID:** {patient_id}\n" \
                           f"**Symptoms:** {symptoms}\n" \
                           f"**Additional Info:** {additional_info if additional_info else 'N/A'}\n\n" \
                           f"**Patient History Context:**\n{patient_history_str}\n\n" \
                           f"**AI Assistant's Insights (Augmented LLM Response):**\n{response_text}\n\n"

        if source_documents:
            formatted_output += "**Relevant Knowledge Base Sources:**\n"
            for i, doc in enumerate(source_documents):
                formatted_output += f"- Source {i+1}: {doc.page_content[:150]}...\n" # Truncate for display

        formatted_output += "\n---\n*Disclaimer: This is an AI assistant for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional.*"

        print("Request processed successfully.")
        return formatted_output

# Instantiate the assistant
assistant = MedicalDiagnosticAssistant(qa_chain=qa_chain, patient_data_fetcher=get_simulated_ehr_data)

# --- Gradio Interface ---

def medical_diagnosis_interface(patient_id: str, symptoms: str, additional_info: str) -> str:
    """Gradio interface function to wrap the assistant's logic."""
    if not patient_id or not symptoms:
        return "Please provide both Patient ID and Symptoms."
    return assistant.diagnose_and_assist(patient_id, symptoms, additional_info)

# Launch the Gradio app
if __name__ == "__main__":
    print("\nStarting Gradio interface...")
    gr.Interface(
        fn=medical_diagnosis_interface,
        inputs=[
            gr.Textbox(label="Patient ID (e.g., P1001, P1002)", placeholder="Enter patient ID"),
            gr.Textbox(label="Patient Symptoms (e.g., 'fever, cough, shortness of breath')", placeholder="Describe symptoms"),
            gr.Textbox(label="Additional Information (e.g., lab results, imaging findings)", placeholder="Any other relevant details", lines=3)
        ],
        outputs=gr.Markdown(label="Medical Diagnostic Assistant Report"),
        title="🏥 Dynamic Knowledge-Augmented Medical Diagnostic Assistant 🏥",
        description="This AI assistant leverages an LLM, real-time medical literature, patient EHRs, and specialized databases to aid in diagnosing complex diseases. "
                    "It integrates dynamic external knowledge to enhance factual accuracy and reasoning. "
                    "**Note:** This is a simulated demonstration. Always consult a medical professional."
    ).launch(share=True)