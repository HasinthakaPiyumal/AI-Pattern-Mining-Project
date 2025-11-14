
import os
from typing import List
import uvicorn

# Langchain imports
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


################################################################################
# config.py content
################################################################################
class Config:
    # OpenAI API Key (replace with your actual key or set as environment variable)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    
    # Vector Database settings
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "medical_knowledge"
    
    # LLM settings
    LLM_MODEL_NAME = "gpt-3.5-turbo"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000

    # Embedding model settings
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Data ingestion settings
    MEDICAL_DATA_PATH = "data/medical_data.txt" # Path to your medical knowledge base file


################################################################################
# vector_database.py content
################################################################################
class VectorDatabase:
    def __init__(self):
        self.persist_directory = Config.CHROMA_DB_PATH
        self.collection_name = Config.COLLECTION_NAME
        self.embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.vector_store = Chroma(client=self.client, 
                                   collection_name=self.collection_name, 
                                   embedding_function=self.embedding_model)

    def get_vector_store(self):
        return self.vector_store

    def add_documents(self, documents):
        """Adds documents to the vector store."""
        if not documents:
            print("No documents to add.")
            return
        print(f"Adding {len(documents)} documents to Chroma DB...")
        # Ensure each document has a page_content attribute for Chroma
        processed_docs = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                processed_docs.append(doc)
            elif isinstance(doc, str):
                processed_docs.append(Document(page_content=doc))
            else:
                print(f"Warning: Skipping document of unknown type: {type(doc)}")

        if processed_docs:
            self.vector_store.add_documents(processed_docs)
            print("Documents added successfully.")
        else:
            print("No valid documents to add after processing.")

    def similarity_search(self, query: str, k: int = 4):
        """Performs similarity search in the vector store."""
        print(f"Performing similarity search for query: {query}")
        results = self.vector_store.similarity_search(query, k=k)
        print(f"Found {len(results)} relevant documents.")
        return results

    def clear_collection(self):
        """Clears the entire collection in the vector database."""
        print(f"Clearing collection: {self.collection_name}...")
        # ChromaDB client.delete_collection does not return the collection object, 
        # so we re-initialize the vector_store after deletion.
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Collection {self.collection_name} deleted.")
        except Exception as e:
            print(f"Could not delete collection {self.collection_name}, it might not exist: {e}")
        
        # Reinitialize the vector_store to ensure it points to a valid (potentially new) collection
        self.vector_store = Chroma(client=self.client, 
                                   collection_name=self.collection_name, 
                                   embedding_function=self.embedding_model)
        print(f"Collection {self.collection_name} reinitialized.")


################################################################################
# data_ingestion.py content
################################################################################
class DataIngestion:
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def load_medical_data(self, file_path: str):
        """Loads medical data from a text file."""
        if not os.path.exists(file_path):
            print(f"Error: Data file not found at {file_path}")
            return []
        print(f"Loading data from {file_path}...")
        loader = TextLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} raw documents.")
        return documents

    def process_and_ingest_data(self, file_path: str):
        """Loads, splits, and ingests medical data into the vector database."""
        documents = self.load_medical_data(file_path)
        if not documents:
            return

        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")

        self.vector_db.add_documents(chunks)
        print("Data ingestion complete.")

def create_dummy_medical_data(file_path: str):
    """Creates a dummy medical data file for testing purposes."""
    data_dir = os.path.dirname(file_path)
    if not data_dir: # Handle case where file_path is just a filename in current directory
        data_dir = "."

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dummy_content = [
        "COVID-19 symptoms include fever, cough, fatigue, and loss of taste or smell. Severe cases may lead to pneumonia and acute respiratory distress syndrome (ARDS). Vaccination is highly recommended to prevent severe disease.",
        "Diabetes Mellitus is a chronic metabolic disease characterized by high blood sugar levels. Type 1 diabetes is an autoimmune condition, while Type 2 diabetes is often associated with lifestyle factors. Management involves diet, exercise, and medication.",
        "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. Lifestyle modifications like reduced sodium intake, regular exercise, and medication can help manage hypertension.",
        "Common cold is a viral infection of the nose and throat. Symptoms include runny nose, sore throat, cough, and sneezing. It usually resolves within 7-10 days with symptomatic treatment.",
        "Asthma is a chronic respiratory condition where airways narrow and swell, producing extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out and shortness of breath.",
        "Migraine is a severe headache often accompanied by symptoms such as throbbing in the head, nausea, vomiting, and extreme sensitivity to light and sound. Triggers can include stress, certain foods, and hormonal changes.",
        "Influenza (Flu) is a contagious respiratory illness caused by influenza viruses. Symptoms are similar to the common cold but are often more severe and can lead to serious complications like pneumonia. Annual vaccination is recommended.",
        "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon on the lower right side of your abdomen. It typically causes pain in the lower right abdomen. Treatment usually involves surgery to remove the appendix.",
        "Gastroesophageal Reflux Disease (GERD) is a chronic digestive disease where stomach acid or bile irritates the food pipe lining. Symptoms include heartburn, regurgitation, and difficulty swallowing. Lifestyle changes and medication can help.",
        "Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. It occurs when the protective cartilage on the ends of your bones wears down over time, leading to pain, stiffness, and swelling, particularly in the joints.",
    ]
    with open(file_path, "w") as f:
        for line in dummy_content:
            f.write(line + "\n\n") # Add double newline to simulate separate paragraphs
    print(f"Dummy medical data created at {file_path}")


################################################################################
# llm_model.py content
################################################################################
class LLMModel:
    def __init__(self):
        if Config.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY" or not Config.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY is not set. Using a mock LLM for local testing.")
            self.llm = self._create_mock_llm()
        else:
            self.llm = ChatOpenAI(
                openai_api_key=Config.OPENAI_API_KEY,
                model_name=Config.LLM_MODEL_NAME,
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS
            )

    def _create_mock_llm(self):
        """Creates a simple mock LLM for testing when no API key is available."""
        class MockLLM:
            def invoke(self, prompt):
                print(f"Mock LLM received prompt:\n{prompt}\n")
                if "COVID-19" in prompt:
                    return Document(page_content="Mock response about COVID-19: Symptoms include fever and cough.")
                elif "Diabetes" in prompt:
                    return Document(page_content="Mock response about Diabetes: Chronic metabolic disease.")
                else:
                    return Document(page_content="Mock LLM response: This is a placeholder response.")
        return MockLLM()

    def get_llm(self):
        return self.llm

    def generate_text(self, prompt: str):
        """Generates text using the configured LLM."""
        print("Generating text with LLM...")
        try:
            response = self.llm.invoke(prompt)
            # For ChatOpenAI, invoke returns a AIMessage object, extract content
            if hasattr(response, 'content'):
                return response.content
            # For simple string responses from mock LLM, ensure it's a string
            return str(response) 
        except Exception as e:
            print(f"Error during LLM text generation: {e}")
            return "An error occurred while generating the response."


################################################################################
# rag_system.py content
################################################################################
class RAGSystem:
    def __init__(self, vector_db: VectorDatabase, llm_model: LLMModel):
        self.vector_db = vector_db
        self.llm = llm_model.get_llm()
        # Ensure the vector_store is correctly initialized before creating retriever
        self.retriever = self.vector_db.get_vector_store().as_retriever()

        self.system_prompt = (
            "You are a helpful medical assistant. Use the following retrieved medical knowledge " 
            "to answer the patient's questions accurately and comprehensively. " 
            "If the information is not in the provided context, state that you don't have " 
            "enough information to answer, and do not make up an answer."
            "Provide clear and concise medical information, avoiding jargon where possible " 
            "or explaining it simply. Always prioritize patient safety and suggest consulting " 
            "a qualified healthcare professional for diagnosis and treatment."
            "\n\nRetrieved medical knowledge:\n{context}"
        )
        self.human_prompt = "{question}"
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", self.human_prompt),
        ])

    def retrieve_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieves relevant documents from the vector database."""
        print(f"RAGSystem: Retrieving documents for query: {query}")
        return self.retriever.invoke(query)

    def condition_context(self, documents: List[Document]) -> str:
        """Formats the retrieved documents into a single context string for the LLM."""
        print("RAGSystem: Conditioning context from retrieved documents.")
        context_text = "\n\n".join([doc.page_content for doc in documents])
        return context_text

    def decide_action(self, query: str) -> str:
        """
        Adaptive decision-making: Decides whether to perform retrieval or 
        directly answer with LLM or abstain. (Placeholder for more advanced logic).
        For now, it always performs retrieval.
        """
        print(f"RAGSystem: Deciding action for query: {query}")
        # In a real system, this could involve:
        # 1. A classifier to determine if the query is medical-related.
        # 2. A confidence score from an initial LLM call without context.
        # 3. Checking for keywords that explicitly require external knowledge.
        # For this demonstration, we will always perform retrieval for medical queries.
        return "retrieve_and_generate"

    def generate_response(self, query: str) -> str:
        """
        Generates a comprehensive response using retrieved information and the LLM.
        """
        action = self.decide_action(query)

        if action == "retrieve_and_generate":
            retrieved_docs = self.retrieve_documents(query)
            if not retrieved_docs:
                return "I couldn't find relevant medical information to answer your question. Please consult a healthcare professional."
            
            context = self.condition_context(retrieved_docs)
            
            # Create the final prompt with context
            final_prompt = self.prompt_template.format(context=context, question=query)
            
            print("RAGSystem: Generating final response with LLM...")
            response = self.llm.invoke(final_prompt)
            # For ChatOpenAI, invoke returns a AIMessage object, extract content
            if hasattr(response, 'content'):
                return response.content
            return str(response) # Ensure response is a string
        elif action == "abstain":
            return "I am not equipped to answer this question. Please consult a qualified professional."
        else: # direct_generate or other actions
            # For now, if not retrieve_and_generate, we treat it as an error or unhandled state.
            # In a real system, more sophisticated logic would be here.
            return "An unexpected action was decided. Please try again or refine your query."


################################################################################
# main.py content
################################################################################
app = FastAPI(title="Medical Diagnostic and Treatment Recommendation System")

# Initialize core components
vector_db = VectorDatabase()
llm_model = LLMModel()
rag_system = RAGSystem(vector_db, llm_model)
data_ingestion = DataIngestion(vector_db)

class QueryRequest(BaseModel):
    patient_query: str

@app.on_event("startup")
async def startup_event():
    """
    On application startup, ensure dummy data exists and ingest it.
    In a production environment, this would be a more robust data pipeline.
    """
    print("Application startup: Initializing data...")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(Config.MEDICAL_DATA_PATH)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Create dummy data if file not found or is empty
    if not os.path.exists(Config.MEDICAL_DATA_PATH) or os.path.getsize(Config.MEDICAL_DATA_PATH) == 0:
        print("Medical data file not found or is empty. Creating dummy data...")
        create_dummy_medical_data(Config.MEDICAL_DATA_PATH)
    
    # Clear the collection first for consistent startup behavior
    vector_db.clear_collection()
    # Ingest data into the vector database
    data_ingestion.process_and_ingest_data(Config.MEDICAL_DATA_PATH)
    print("Application startup: Data initialization complete.")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Medical Diagnostic and Treatment Recommendation System!"}

@app.post("/diagnose_and_recommend")
async def diagnose_and_recommend(request: QueryRequest):
    """
    Endpoint to get a medical diagnosis and treatment recommendation based on a patient query.
    """
    try:
        print(f"Received query: {request.patient_query}")
        response = rag_system.generate_response(request.patient_query)
        return {"query": request.patient_query, "recommendation": response}
    except Exception as e:
        print(f"An error occurred during diagnosis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    # To run this application:
    # 1. Ensure you have uvicorn installed: pip install uvicorn
    # 2. Run from your terminal: python medical_rag_system.py or uvicorn medical_rag_system:app --reload --port 8000
    # 3. Access the API at http://127.0.0.1:8000/docs for Swagger UI

    # This block allows running directly with `python medical_rag_system.py`
    uvicorn.run(app, host="0.0.0.0", port=8000)
