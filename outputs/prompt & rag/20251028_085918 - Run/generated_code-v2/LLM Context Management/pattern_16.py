import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define the directory for medical documents
DOC_DIR = "./medical_documents"
VECTOR_DB_DIR = "./chroma_db"

def setup_documents_and_vector_db():
    # Create a directory for dummy medical documents if it doesn't exist
    os.makedirs(DOC_DIR, exist_ok=True)

    # Create some dummy medical documents for demonstration
    # In a real application, these would be actual clinical guidelines, drug info, etc.
    with open(os.path.join(DOC_DIR, "clinical_guideline_diabetes.txt"), "w") as f:
        f.write("### Clinical Guideline: Type 2 Diabetes Management\n\n"
                "**Diagnosis:** Fasting plasma glucose >= 126 mg/dL or A1C >= 6.5%.\n"
                "**First-line Treatment:** Metformin, lifestyle modifications.\n"
                "**Second-line Treatment:** Add SGLT2 inhibitors or GLP-1 receptor agonists if glycemic targets not met.\n"
                "**Monitoring:** Regular A1C checks (every 3-6 months), blood pressure, lipid profile.\n"
                "**Complications:** Nephropathy, retinopathy, neuropathy, cardiovascular disease. Early detection is key.\n"
                "**Source:** American Diabetes Association (ADA) Guidelines 2023.")

    with open(os.path.join(DOC_DIR, "drug_info_metformin.txt"), "w") as f:
        f.write("### Drug Information: Metformin\n\n"
                "**Class:** Biguanide (oral hypoglycemic agent).\n"
                "**Mechanism of Action:** Decreases hepatic glucose production, decreases intestinal absorption of glucose, improves insulin sensitivity.\n"
                "**Indications:** Type 2 Diabetes Mellitus.\n"
                "**Contraindications:** Severe renal impairment (eGFR <30 mL/min/1.73 m2), metabolic acidosis.\n"
                "**Common Side Effects:** Diarrhea, nausea, abdominal discomfort. Take with food to minimize.\n"
                "**Dosage:** Initial 500 mg once or twice daily with meals. Max dose 2550 mg/day.\n"
                "**Source:** FDA Drug Label, Lexicomp Database.")

    with open(os.path.join(DOC_DIR, "treatment_protocol_hypertension.txt"), "w") as f:
        f.write("### Treatment Protocol: Hypertension\n\n"
                "**Diagnosis:** Blood pressure >= 130/80 mmHg (Stage 1) or >= 140/90 mmHg (Stage 2).\n"
                "**Lifestyle Modifications:** Diet (DASH), exercise, reduced sodium intake, limited alcohol.\n"
                "**Pharmacological Treatment (Stage 1):** Single agent (ACE inhibitor, ARB, Thiazide diuretic, CCB). \n"
                "**Pharmacological Treatment (Stage 2):** Two agents from different classes.\n"
                "**Target BP:** <130/80 mmHg for most adults.\n"
                "**Monitoring:** Regular BP checks, electrolyte balance, renal function.\n"
                "**Source:** American Heart Association (AHA) Guidelines 2023.")

    print(f"Loaded {len(os.listdir(DOC_DIR))} dummy medical documents into '{DOC_DIR}'.")

    # 1. Load documents
    documents = []
    for filename in os.listdir(DOC_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DOC_DIR, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # 3. Initialize embeddings model
    # Using a local HuggingFace embedding model for self-contained execution
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and persist the Chroma vector store
    # This will store the embeddings and document chunks locally
    db = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_DB_DIR)
    db.persist()
    print(f"Chroma vector database created and persisted at '{VECTOR_DB_DIR}'.")

if __name__ == "__main__":
    setup_documents_and_vector_db()