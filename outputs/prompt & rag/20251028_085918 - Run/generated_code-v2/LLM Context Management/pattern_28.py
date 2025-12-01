import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables (for OpenAI API key)
load_dotenv()

# --- 1. Knowledge Base (Non-Parametric Memory) ---

# Simulate medical documents
medical_documents = [
    "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain. It is also used to treat inflammatory conditions such as arthritis, and to reduce the risk of heart attack and stroke.",
    "Metformin is a first-line medication for type 2 diabetes, primarily working by decreasing glucose production by the liver and improving insulin sensitivity. Common side effects include nausea and diarrhea.",
    "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle changes, including diet and exercise, are often recommended, alongside medications like ACE inhibitors or diuretics.",
    "The recommended dosage for adult ibuprofen for pain relief is typically 200-400 mg every 4-6 hours as needed, not exceeding 1200 mg in a 24-hour period. It should be taken with food or milk to reduce stomach upset.",
    "Penicillin allergies can manifest in various ways, from mild rashes to severe anaphylaxis. It is crucial to document a patient's allergic reactions thoroughly. Alternative antibiotics for penicillin-allergic patients include macrolides or clindamycin.",
    "COVID-19 symptoms commonly include fever, cough, fatigue, and loss of taste or smell. Severe cases can lead to pneumonia, acute respiratory distress syndrome (ARDS), and multi-organ failure. Vaccination is highly effective in preventing severe disease.",
    "Type 1 diabetes is an autoimmune condition where the body's immune system destroys the insulin-producing beta cells in the pancreas. It requires lifelong insulin therapy. Unlike type 2 diabetes, it is not primarily associated with lifestyle factors.",
    "Migraine headaches are characterized by severe throbbing pain or a pulsing sensation, usually on one side of the head. They are often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Triptans are a common class of drugs used to treat migraines.",
    "The Human Papillomavirus (HPV) vaccine protects against infections that can cause certain cancers, including cervical, anal, and oral cancers. It is recommended for adolescents, ideally before they become sexually active.",
    "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, shortness of breath, chest tightness, and coughing. Inhalers are commonly used for both quick relief and long-term control."
]

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 50,
    length_function = len,
    is_separator_regex = False,
)

texts = text_splitter.create_documents(medical_documents)

# Embedding Model
# Using a SentenceTransformer model via HuggingFaceEmbeddings
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Vector Store (ChromaDB)
# Initialize ChromaDB from documents and embeddings
print("Initializing ChromaDB with medical documents...")
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("ChromaDB initialized.")

# --- 2. Retrieval Mechanism ---
# Configure the retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# --- 3. Generative Language Model (Parametric Memory) ---
# Initialize the LLM (using OpenAI's Chat model)
# Ensure OPENAI_API_KEY is set in your environment variables
print("Initializing Generative Language Model...")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

# --- Alternative: Local HuggingFace Model (uncomment to use) ---
# from langchain_community.llms import HuggingFacePipeline
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
#
# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
#
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     device=0 # or -1 for CPU
# )
#
# llm = HuggingFacePipeline(pipeline=pipe)
# print("Local HuggingFace model initialized.")

# --- 4. Orchestration (RAG Pipeline) ---
# Create the RetrievalQA chain
print("Creating RAG pipeline...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # Combines all retrieved documents into one prompt
    retriever=retriever,
    return_source_documents=True
)
print("RAG pipeline created. Ready to answer queries.")

# --- 5. Example Usage (Conceptual UI for Clinicians) ---
def medical_assistant_query(query: str):
    print(f"\nClinician Query: {query}")
    response = qa_chain.invoke({"query": query})
    print("\n--- Medical Assistant Response ---")
    print(f"Answer: {response['result']}")
    print("\nSource Documents:")
    for i, doc in enumerate(response['source_documents']):
        print(f"  Document {i+1}:\n    Content: {doc.page_content}\n    Metadata: {doc.metadata}")
    print("----------------------------------")

if __name__ == "__main__":
    # Example Queries
    medical_assistant_query("What is the first-line treatment for type 2 diabetes?")
    medical_assistant_query("Can you tell me about the symptoms and prevention of COVID-19?")
    medical_assistant_query("What are the common side effects of ibuprofen and its recommended dosage?")
    medical_assistant_query("What are the characteristics of a migraine headache?")
    medical_assistant_query("What are alternatives to penicillin for allergic patients?")

    # To clean up the ChromaDB directory (optional)
    # import shutil
    # if os.path.exists("./chroma_db"):
    #     shutil.rmtree("./chroma_db")
    #     print("ChromaDB directory removed.")
