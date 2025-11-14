import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
# Set your LLM model. Ensure Ollama is running and has 'llama2' model pulled, or use OpenAI API key.
# If using OpenAI, uncomment the relevant lines and set OPENAI_API_KEY environment variable.
# from langchain_openai import ChatOpenAI
LLM_MODEL = os.getenv("LLM_MODEL", "llama2") # e.g., "llama2", "gpt-3.5-turbo"

# --- 1. Prepare Medical Knowledge Base (Dummy Data) ---
# In a real application, this would come from databases, research papers, etc.
medical_documents = [
    """Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing. Various organisms, including bacteria, viruses, and fungi, can cause pneumonia.""",
    """Common symptoms of influenza (flu) include fever, muscle aches, headache, cough, and sore throat. It is caused by influenza viruses and can lead to complications like pneumonia.""",
    """Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). Symptoms include increased thirst, frequent urination, increased hunger, unintended weight loss, fatigue, blurred vision, and slow-healing sores.""",
    """Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Often, there are no symptoms.""",
    """Migraine is a severe headache often accompanied by symptoms such as throbbing pain on one side of the head, nausea, vomiting, and extreme sensitivity to light and sound. Triggers can include stress, certain foods, and hormonal changes."""
]

# --- 2. Initialize Embeddings and Vector Store ---
# Using HuggingFaceEmbeddings for local embedding generation. Can be replaced with OpenAIEmbeddings.
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Split documents into chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.create_documents(medical_documents)

# Create or load the Chroma vector store
# In a real system, you might persist this to disk
print("Creating/loading vector store...")
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
retriever = vectorstore.as_retriever()

# --- 3. Initialize LLM ---
# Using Ollama with llama2. Ensure Ollama server is running and model is pulled.
# Alternatively, use ChatOpenAI if you have an API key set up.
print(f"Initializing LLM: {LLM_MODEL}...")
if LLM_MODEL.startswith("gpt"):
    # llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.7)
    # print("WARNING: OpenAI model selected, but code for it is commented out. Please uncomment and ensure OPENAI_API_KEY is set.")
    raise NotImplementedError("OpenAI LLM is not fully implemented in this example. Please use Ollama or implement OpenAI integration.")
elif LLM_MODEL == "llama2":
    llm = Ollama(model=LLM_MODEL, temperature=0.7)
else:
    raise ValueError(f"Unsupported LLM model: {LLM_MODEL}. Please use 'llama2' or configure OpenAI.")

# --- 4. Define RAG Prompt ---
# This prompt instructs the LLM to use the provided context.
rag_prompt_template = """You are a medical diagnostic assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't have enough information from the provided context, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
rag_prompt = PromptTemplate.from_template(rag_prompt_template)

# --- 5. Construct RAG Chain ---
# This chain retrieves relevant documents, formats them, and passes them to the LLM.
print("Constructing RAG chain...")
rag_chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- Function to get diagnostic support ---
def get_diagnostic_support(patient_symptoms: str) -> str:
    """Provides diagnostic support based on patient symptoms using RAG."""
    print(f"Processing query: {patient_symptoms}")
    response = rag_chain.invoke(patient_symptoms)
    return response

if __name__ == "__main__":
    # Example usage
    print("\n--- Example Diagnostic Query ---")
    symptoms = "I have a high fever, terrible muscle aches, and a persistent cough. My throat is also very sore."
    diagnosis = get_diagnostic_support(symptoms)
    print(f"\nPatient Symptoms: {symptoms}")
    print(f"Assistant Diagnosis: {diagnosis}")

    print("\n--- Another Example ---")
    symptoms_2 = "I've been feeling extremely thirsty, urinating a lot, and losing weight without trying. My vision seems a bit blurry too."
    diagnosis_2 = get_diagnostic_support(symptoms_2)
    print(f"\nPatient Symptoms: {symptoms_2}")
    print(f"Assistant Diagnosis: {diagnosis_2}")

    print("\n--- Query with insufficient information in context ---")
    symptoms_3 = "I have a rash on my arm. What could it be?"
    diagnosis_3 = get_diagnostic_support(symptoms_3)
    print(f"\nPatient Symptoms: {symptoms_3}")
    print(f"Assistant Diagnosis: {diagnosis_3}")

    # Clean up (optional: remove the Chroma DB directory if you want to regenerate it)
    # import shutil
    # if os.path.exists("./chroma_db"):
    #     shutil.rmtree("./chroma_db")
    #     print("Chroma DB directory removed.")
