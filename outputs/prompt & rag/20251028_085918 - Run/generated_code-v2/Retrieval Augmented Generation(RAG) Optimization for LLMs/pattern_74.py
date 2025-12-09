import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Define paths
DB_DIR = "./chroma_db"
DOCS_DIR = "./medical_docs"

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)

# Sample medical documents (replace with your actual document loading logic)
sample_docs = [
    {
        "filename": "diabetes_overview.txt",
        "content": "Diabetes is a chronic condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let blood sugar into your body’s cells for use as energy. With diabetes, your body doesn’t make enough insulin or can’t use the insulin it makes as well as it should. When there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, that can cause serious health problems, such as heart disease, kidney disease, and nerve damage. The main types of diabetes are type 1, type 2, and gestational diabetes."
    },
    {
        "filename": "hypertension_treatment.txt",
        "content": "Hypertension, or high blood pressure, can be managed through lifestyle changes and medication. Lifestyle modifications include a healthy diet low in sodium, regular physical activity, maintaining a healthy weight, limiting alcohol intake, and quitting smoking. Common medications for hypertension include diuretics, ACE inhibitors, ARBs, beta-blockers, and calcium channel blockers. It\\'s crucial to consult a doctor to determine the best treatment plan, as uncontrolled high blood pressure can lead to serious complications like heart attack, stroke, and kidney failure."
    },
    {
        "filename": "fever_management.txt",
        "content": "A fever is a temporary increase in your body temperature, often due to an illness. For adults, a fever may be uncomfortable, but usually isn\\'t a cause for concern unless it reaches 103 F (39.4 C) or higher. For infants and toddlers, any fever should be reported to a doctor. To manage a fever, rest, drink plenty of fluids, and use over-the-counter medications like acetaminophen (Tylenol) or ibuprofen (Advil, Motrin) as directed. Avoid overdressing, which can trap body heat. Seek medical attention if a fever is accompanied by severe headache, stiff neck, shortness of breath, or rash."
    },
    {
        "filename": "covid19_symptoms.txt",
        "content": "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Common symptoms include fever or chills, cough, fatigue, muscle or body aches, headache, sore throat, congestion or runny nose, nausea or vomiting, and diarrhea. Some people may also experience a loss of taste or smell. Symptoms can range from mild to severe, and usually appear 2-14 days after exposure to the virus. If you suspect you have COVID-19, it\\'s recommended to get tested and isolate yourself to prevent further spread. Vaccination is highly effective in preventing severe illness and death from COVID-19."
    }
]

# Write sample documents to files
print(f"Saving sample documents to {DOCS_DIR}...")
for doc in sample_docs:
    with open(os.path.join(DOCS_DIR, doc["filename"]), "w", encoding="utf-8") as f:
        f.write(doc["content"])
print("Sample documents saved.")

# Load documents from the directory
print(f"Loading documents from {DOCS_DIR}...")
loader = TextLoader
documents = []
for filename in os.listdir(DOCS_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCS_DIR, filename)
        loader_instance = loader(filepath, encoding="utf-8")
        documents.extend(loader_instance.load())
print(f"Loaded {len(documents)} documents.")

# Split documents into chunks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)
print(f"Split into {len(splits)} chunks.")

# Initialize embeddings model
print("Initializing SentenceTransformer embeddings (all-MiniLM-L6-v2)...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings model initialized.")

# Create and persist the vector store (Chroma DB)
print(f"Creating and persisting Chroma DB at {DB_DIR}...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=DB_DIR
)
vectorstore.persist()
print("Chroma DB created and persisted successfully.")
print("Run 'medical_rag_app.py' next to interact with the Q&A system.")