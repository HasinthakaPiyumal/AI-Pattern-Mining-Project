import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

# Set your OpenAI API key as an environment variable (or replace directly)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Simulated Medical Knowledge Base
# In a real application, this would be loaded from a database, API, or files.
medical_documents = [
    "Aspirin is commonly used for pain relief and to reduce fever. It can also be used as an anti-inflammatory and to prevent blood clots.",
    "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). Symptoms include increased thirst, frequent urination, and blurred vision.",
    "The flu, or influenza, is a contagious respiratory illness caused by influenza viruses. Symptoms are fever, cough, sore throat, and body aches. Vaccination is recommended annually.",
    "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle changes and medication can help manage it.",
    "Antibiotics are medicines that fight bacterial infections in people and animals. They work by killing the bacteria or preventing them from reproducing. They are not effective against viral infections like the common cold or flu.",
    "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Symptoms range from mild to severe, including fever, cough, fatigue, and loss of taste or smell. Vaccines are highly effective in preventing severe illness.",
    "Cancer is a disease in which some of the body's cells grow uncontrollably and spread to other parts of the body. Treatment options include chemotherapy, radiation, surgery, and immunotherapy.",
    "Headaches are a common condition that most people will experience many times in their lives. The main symptom of a headache is a pain in your head or face. They can be mild, moderate, or severe.",
    "Allergies occur when your immune system reacts to a foreign substance — such as pollen, bee venom, or pet dander — that doesn't cause a reaction in most people. Symptoms can include sneezing, itching, rashes, and swelling."
]

# 2. Initialize Embedding Model
# We use a SentenceTransformer model for generating embeddings.
print("Loading embedding model...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# 3. Create Vector Store
# First, split documents into smaller chunks for better retrieval.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.create_documents(medical_documents)

print("Creating Chroma vector store...")
# Create a Chroma vector store from the split documents and embeddings
# This simulates storing the knowledge base in an efficient, searchable format.
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
print("Chroma vector store created.")

# 4. Initialize LLM
# Ensure OPENAI_API_KEY is set in your environment variables
if os.getenv("OPENAI_API_KEY") is None:
    print("Warning: OPENAI_API_KEY not found. Please set it as an environment variable or replace 'ChatOpenAI' with a local LLM if running without OpenAI.")
    # For demonstration, you might want to uncomment the line below and use a local LLM or mock it.
    llm = None # Placeholder, will cause an error if used without API key
else:
    llm = ChatOpenAI(temperature=0.7) # Using OpenAI's GPT model

if llm:
    # 5. Create RetrievalQA Chain
    # This chain orchestrates the retrieval of relevant documents and then passes them to the LLM.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' combines all retrieved documents into one prompt
        retriever=vectorstore.as_retriever(),
        return_source_documents=True # To see which documents were retrieved
    )

    print("\n--- Medical Information Assistant Ready ---")

    # Example Usage
    query = "What are the uses of Aspirin and what are its main functions?"
    print(f"User Query: {query}")

    result = qa_chain.invoke({"query": query})

    print("\n--- AI Response ---")
    print(result["result"])

    print("\n--- Retrieved Source Documents ---")
    for i, doc in enumerate(result["source_documents"]):
        print(f"Document {i+1}: {doc.page_content}")

    print("\n---\n")

    query_2 = "How is Type 2 diabetes characterized and what are its symptoms?"
    print(f"User Query: {query_2}")

    result_2 = qa_chain.invoke({"query": query_2})

    print("\n--- AI Response ---")
    print(result_2["result"])

    print("\n--- Retrieved Source Documents ---")
    for i, doc in enumerate(result_2["source_documents"]):
        print(f"Document {i+1}: {doc.page_content}")

else:
    print("Cannot run LLM-based QA chain without a valid LLM initialization. Please set OPENAI_API_KEY or configure a local LLM.")
