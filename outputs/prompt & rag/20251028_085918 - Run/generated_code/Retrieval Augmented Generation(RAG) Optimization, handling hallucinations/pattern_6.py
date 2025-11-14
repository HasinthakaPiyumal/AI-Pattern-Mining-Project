
import os
from functools import lru_cache
from typing import List, Dict

# Mocking imports that would require installation or specific setup
try:
    from sentence_transformers import SentenceTransformer
    from chromadb import Client, Documents, EmbeddingFunction, Embeddings, HTTPClient, Settings
    from chromadb.utils import embedding_functions
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
    import gradio as gr
    from loguru import logger
except ImportError:
    logger.warning("Some libraries are not installed. Using mock implementations. Please install with `pip install sentence-transformers chromadb langchain-core gradio loguru` for full functionality.")

    # Mock classes/functions for demonstration without full installs
    class MockSentenceTransformer:
        def __init__(self, model_name):
            logger.info(f"Mocking SentenceTransformer for {model_name}")
            self.model_name = model_name
        def encode(self, sentences, convert_to_tensor=False):
            logger.info(f"Mocking embedding for: {sentences}")
            # Return a consistent mock embedding for demonstration
            return [[0.1] * 384 for _ in sentences] # Mock a 384-dim embedding

    class MockEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return [[0.1] * 384 for _ in input]

    class MockChromaCollection:
        def __init__(self, name="mock_collection"):
            logger.info(f"Mocking ChromaCollection: {name}")
            self.name = name
            self.documents = []
            self.metadatas = []
            self.ids = []
            self.embeddings = []
            self.ef = MockEmbeddingFunction()

        def add(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
            logger.info(f"Mocking add to ChromaDB: {len(documents)} documents")
            self.documents.extend(documents)
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)
            self.embeddings.extend(self.ef(documents))

        def query(self, query_texts: List[str], n_results: int = 2, where: Dict = None) -> Dict:
            logger.info(f"Mocking query to ChromaDB for: {query_texts}")
            # Simple mock: return first N documents
            if not self.documents:
                return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
            
            # Simulate finding relevant docs based on a keyword or just return some default ones
            results_docs = []
            results_metas = []
            
            # A very naive similarity simulation for demo
            # In a real scenario, this would use actual vector similarity
            for q_text in query_texts:
                found_docs = []
                found_metas = []
                for i, doc in enumerate(self.documents):
                    if q_text.lower() in doc.lower() or any(k.lower() in doc.lower() for k in q_text.lower().split()):
                        found_docs.append(doc)
                        found_metas.append(self.metadatas[i])
                        if len(found_docs) >= n_results:
                            break
                
                # If not enough specific matches, fill with general docs
                while len(found_docs) < n_results and len(found_docs) < len(self.documents):
                    for doc, meta in zip(self.documents, self.metadatas):
                        if doc not in found_docs:
                            found_docs.append(doc)
                            found_metas.append(meta)
                            if len(found_docs) >= n_results:
                                break
                
                results_docs.append(found_docs)
                results_metas.append(found_metas)

            return {
                "documents": results_docs,
                "metadatas": results_metas,
                "ids": [["mock_id"] * n_results for _ in query_texts]
            }

    class MockChromaClient:
        def __init__(self, path=None):
            logger.info(f"Mocking ChromaDB Client with path: {path}")
            self.path = path
            self.collections = {}
        def get_or_create_collection(self, name: str, embedding_function: EmbeddingFunction = None):
            if name not in self.collections:
                self.collections[name] = MockChromaCollection(name=name)
            return self.collections[name]

    Client = MockChromaClient
    SentenceTransformer = MockSentenceTransformer
    embedding_functions.SentenceTransformerEmbeddingFunction = MockEmbeddingFunction # Mock the embedding function for ChromaDB
    PromptTemplate = object # Mock LangChain classes
    RunnableLambda = object
    gr = object
    logger = print # Mock logger

# --- Configuration --- #
MEDICAL_CORPUS_DIR = "./medical_corpus"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3 # Number of relevant documents to retrieve

# --- Logging Setup --- #
logger.remove()
logger.add(os.sys.stderr, format="<green>{time}</green> <level>{level: <8}</level> <bold>{message}</bold>", level="INFO")
logger.add("medical_rag_system.log", rotation="1 week", level="DEBUG")

# --- 1. Data Ingestion and Knowledge Base (Medical Corpus) --- #
logger.info("Setting up medical corpus and vector database...")

# Simulate diverse medical data sources
mock_medical_documents = [
    {
        "id": "doc1",
        "content": "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy. With diabetes, your body either does not make enough insulin or cannot effectively use the insulin it does make.",
        "metadata": {"source": "medical_journal", "topic": "diabetes"}
    },
    {
        "id": "doc2",
        "content": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle changes, including diet and exercise, are often the first line of treatment.",
        "metadata": {"source": "health_blog", "topic": "hypertension"}
    },
    {
        "id": "doc3",
        "content": "Common symptoms of a flu virus include fever, cough, sore throat, muscle aches, and fatigue. Rest, hydration, and antiviral medications can help manage symptoms and reduce the duration of the illness.",
        "metadata": {"source": "cdc_guidelines", "topic": "influenza"}
    },
    {
        "id": "doc4",
        "content": "For patients experiencing symptoms of acute myocardial infarction (heart attack), immediate medical attention is crucial. Treatment often involves aspirin, nitroglycerin, oxygen, and thrombolytic agents or angioplasty to restore blood flow.",
        "metadata": {"source": "emergency_medicine_textbook", "topic": "cardiology"}
    },
    {
        "id": "doc5",
        "content": "Migraine headaches are characterized by severe throbbing pain or a pulsing sensation, usually on one side of the head, often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Triptans are a common class of drugs used to treat migraines.",
        "metadata": {"source": "neurology_handbook", "topic": "neurology"}
    }
]

# Initialize embedding model
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer: {e}. Using mock embedding model.")
    embedding_model = MockSentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_ef = MockEmbeddingFunction()

# Initialize ChromaDB client and collection
try:
    chroma_client = Client(settings=Settings(persist_directory=CHROMA_DB_PATH, is_persistent=True))
    medical_collection = chroma_client.get_or_create_collection(
        name="medical_knowledge_base",
        embedding_function=chroma_ef # Use the same embedding function for ChromaDB
    )

    # Add documents to ChromaDB if not already present
    if medical_collection.count() == 0:
        logger.info(f"Adding {len(mock_medical_documents)} documents to ChromaDB...")
        medical_collection.add(
            documents=[doc["content"] for doc in mock_medical_documents],
            metadatas=[doc["metadata"] for doc in mock_medical_documents],
            ids=[doc["id"] for doc in mock_medical_documents]
        )
        logger.info(f"Documents added. Total documents in DB: {medical_collection.count()}")
    else:
        logger.info(f"ChromaDB already contains {medical_collection.count()} documents. Skipping addition.")

except Exception as e:
    logger.error(f"Failed to initialize ChromaDB or add documents: {e}. Using mock ChromaDB client.")
    chroma_client = MockChromaClient(path=CHROMA_DB_PATH)
    medical_collection = chroma_client.get_or_create_collection(
        name="medical_knowledge_base",
        embedding_function=chroma_ef
    )
    # Manually add mock docs to the mock client if it's being used
    if medical_collection.count() == 0:
        medical_collection.add(
            documents=[doc["content"] for doc in mock_medical_documents],
            metadatas=[doc["metadata"] for doc in mock_medical_documents],
            ids=[doc["id"] for doc in mock_medical_documents]
        )
        logger.info(f"Mock ChromaDB populated with {medical_collection.count()} documents.")

logger.info("Medical knowledge base setup complete.")

# --- 2. Retrieval-Augmented Generation (RAG) Core --- #

# Mock LLM for demonstration purposes
# In a real application, you would load a local LLM via transformers or connect to an API.
class MockLLM:
    def __init__(self):
        logger.info("Initializing Mock LLM.")

    def invoke(self, prompt_text: str) -> str:
        logger.debug(f"Mock LLM received prompt: {prompt_text[:200]}...")
        # Simple keyword-based response simulation
        if "diagnosis for a patient with symptoms" in prompt_text.lower():
            if "fever, cough, sore throat" in prompt_text.lower():
                return "Based on the provided symptoms (fever, cough, sore throat) and medical context, the likely diagnosis is influenza (flu). Treatment typically involves rest, hydration, and possibly antiviral medications. Please consult a doctor for a definitive diagnosis and treatment plan."
            elif "high blood sugar" in prompt_text.lower():
                return "Given the symptom of high blood sugar and medical context, the patient likely has Diabetes Mellitus. Management includes insulin therapy or medication, diet, and exercise. A medical professional should confirm and prescribe treatment."
            elif "chest pain" in prompt_text.lower() or "heart attack" in prompt_text.lower():
                return "Symptoms like chest pain, especially described as acute myocardial infarction in the context, indicate a heart attack. Immediate emergency medical care is required, potentially involving aspirin, nitroglycerin, and angioplasty. This is a medical emergency."
            elif "severe throbbing pain in head" in prompt_text.lower():
                return "The symptoms of severe throbbing headache, nausea, and sensitivity to light are highly suggestive of a migraine. Triptans are often prescribed for treatment. Consulting a neurologist is recommended."
            elif "high blood pressure" in prompt_text.lower():
                return "High blood pressure (hypertension) requires careful management. Lifestyle changes such as diet and exercise are crucial, and medication may be necessary. Regular monitoring by a physician is essential."
            else:
                return "Based on the general symptoms and provided context, further investigation is needed for a precise diagnosis. Please consult a healthcare professional."
        elif "treatment for" in prompt_text.lower() and "diabetes" in prompt_text.lower():
            return "Treatment for diabetes involves careful management of blood sugar levels through insulin or other medications, dietary changes, and regular exercise. Continuous monitoring and a personalized plan from an endocrinologist are vital."
        elif "what is" in prompt_text.lower() and "hypertension" in prompt_text.lower():
            return "Hypertension is high blood pressure. It can lead to serious health issues if left untreated. Lifestyle modifications and medication are common treatments."
        else:
            return f"I am a medical recommendation system. For '{prompt_text[:50]}...', I recommend consulting a qualified healthcare professional for personalized advice and diagnosis, leveraging the information I have about your query."

mock_llm = MockLLM()

# --- Intelligent Retrieval Module --- #
def retrieve_context(query: str) -> List[str]:
    logger.info(f"Retrieving context for query: {query}")
    try:
        # ChromaDB query expects a list of query texts
        results = medical_collection.query(
            query_texts=[query],
            n_results=TOP_K_RETRIEVAL
        )
        retrieved_documents = results["documents"][0] if results["documents"] else []
        logger.debug(f"Retrieved {len(retrieved_documents)} documents.")
        return retrieved_documents
    except Exception as e:
        logger.error(f"Error during context retrieval: {e}")
        return []

# --- Context Conditioning Module --- #
# LangChain Prompt Template (mocked if LangChain not installed)
if not isinstance(PromptTemplate, object):
    medical_rag_prompt = PromptTemplate.from_template(
        """You are a highly accurate and trustworthy Medical Diagnostic and Treatment Recommendation System. 
        Use the following medical context to answer the patient's query. 
        If the context does not provide enough information, state that you need more information or recommend consulting a specialist.

        Medical Context:
        {context}

        Patient Query: {query}

        Provide a detailed diagnosis and treatment recommendation based ONLY on the provided context and your medical knowledge. 
        If you are unsure or the context is insufficient, state so clearly and recommend professional medical advice.
        """
    )

    # Simulate LangChain's Runnable for LLM
    class LangChainLLMRunnable:
        def __init__(self, llm_instance, prompt_template):
            self.llm = llm_instance
            self.prompt_template = prompt_template

        def invoke(self, inputs: Dict) -> str:
            formatted_prompt = self.prompt_template.format(**inputs)
            return self.llm.invoke(formatted_prompt)
    
    # For the actual runnable, we would typically chain components:
    # rag_chain = medical_rag_prompt | llm
    # For this mock, we'll create a single callable that mimics this.
    rag_chain = LangChainLLMRunnable(mock_llm, medical_rag_prompt)
else:
    # Mock LangChain components if not installed
    logger.warning("Using mock LangChain components for prompt and LLM chain.")
    class MockPromptTemplate:
        def from_template(self, template_str):
            self.template = template_str
            return self
        def format(self, **kwargs):
            formatted_string = self.template
            for key, value in kwargs.items():
                formatted_string = formatted_string.replace(f"{{{key}}}", str(value))
            return formatted_string
    
    PromptTemplate = MockPromptTemplate
    medical_rag_prompt = PromptTemplate().from_template(
        """You are a highly accurate and trustworthy Medical Diagnostic and Treatment Recommendation System. 
        Use the following medical context to answer the patient's query. 
        If the context does not provide enough information, state that you need more information or recommend consulting a specialist.

        Medical Context:
        {context}

        Patient Query: {query}

        Provide a detailed diagnosis and treatment recommendation based ONLY on the provided context and your medical knowledge. 
        If you are unsure or the context is insufficient, state so clearly and recommend professional medical advice.
        """
    )

    class MockLangChainLLMRunnable:
        def __init__(self, llm_instance, prompt_template_instance):
            self.llm = llm_instance
            self.prompt_template = prompt_template_instance

        def invoke(self, inputs: Dict) -> str:
            formatted_prompt = self.prompt_template.format(**inputs)
            return self.llm.invoke(formatted_prompt)
    
    rag_chain = MockLangChainLLMRunnable(mock_llm, medical_rag_prompt)

# --- Adaptive Decision-Making Module --- #
# A simple heuristic: if query contains very common, non-specific terms, don't retrieve.
# In a real system, this could be a small classifier or more complex logic.
def should_retrieve(query: str) -> bool:
    common_medical_terms = ["what is", "tell me about", "general health", "how to stay healthy"]
    for term in common_medical_terms:
        if term in query.lower():
            logger.info(f"Query '{query}' contains common terms, skipping extensive retrieval.")
            return False
    logger.info(f"Query '{query}' requires retrieval.")
    return True

# --- Main RAG Generation Function (with Caching) --- #
@lru_cache(maxsize=128) # Cache up to 128 recent queries
def generate_recommendation_cached(patient_query: str) -> str:
    logger.info(f"Generating recommendation for: {patient_query}")
    retrieved_context = []
    if should_retrieve(patient_query):
        retrieved_context = retrieve_context(patient_query)
    
    context_str = "\n---\n".join(retrieved_context) if retrieved_context else "No specific relevant medical context found from external sources. Relying on general medical knowledge."
    
    try:
        # LangChain Runnable takes a dictionary of inputs
        llm_response = rag_chain.invoke({"context": context_str, "query": patient_query})
        logger.info("Recommendation generated successfully.")
        return llm_response
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        return "An error occurred while generating the recommendation. Please try again or consult IT support."

# Expose a non-cached version for Gradio if needed, or just use the cached one.
def generate_recommendation(patient_query: str) -> str:
    return generate_recommendation_cached(patient_query)

# --- 4. User Interface (Gradio) --- #
logger.info("Setting up Gradio interface...")

if not isinstance(gr, object):
    with gr.Blocks(title="Medical RAG System") as demo:
        gr.Markdown(
            """# Medical Diagnostic and Treatment Recommendation System
            Enter patient symptoms or a medical query to get diagnostic insights and treatment recommendations.
            This system leverages Retrieval-Augmented Generation (RAG) for accurate and up-to-date information.
            **Disclaimer: This system is for informational purposes only and should not replace professional medical advice.**
            """
        )
        
        query_input = gr.Textbox(
            label="Patient Symptoms / Medical Query", 
            placeholder="e.g., Patient has high blood sugar, frequent urination, and increased thirst."
        )
        output_text = gr.Textbox(label="Diagnosis and Treatment Recommendation", interactive=False, lines=10)
        
        submit_button = gr.Button("Get Recommendation")
        submit_button.click(fn=generate_recommendation, inputs=query_input, outputs=output_text)
        
        gr.Examples(
            [
                "Patient complains of fever, persistent cough, and sore throat.",
                "What are the latest treatments for type 2 diabetes?",
                "Describe the symptoms and immediate steps for a heart attack.",
                "I have severe throbbing pain on one side of my head, with nausea and light sensitivity. What could it be?",
                "Tell me about hypertension and its management."
            ],
            inputs=query_input
        )

    logger.info("Gradio interface ready. Launching...")
    try:
        demo.launch(share=False) # Set share=True to get a public link (careful with sensitive data)
    except Exception as e:
        logger.error(f"Failed to launch Gradio app: {e}")
        print("Gradio app could not be launched. Please ensure Gradio is installed and port is available.")
        print("You can still test the RAG function directly via Python calls.")
        # Fallback to CLI test if Gradio fails
        while True:
            cli_query = input("Enter a medical query (or 'exit' to quit): ")
            if cli_query.lower() == 'exit':
                break
            if cli_query:
                print("\n--- Recommendation ---")
                print(generate_recommendation(cli_query))
                print("---------------------\n")
else:
    logger.error("Gradio is not installed or mocked. Cannot launch UI. Please install Gradio to use the web interface.")
    print("\n--- CLI Fallback for Medical RAG System ---")
    print("Gradio UI is not available. You can test the RAG system via the command line.")
    while True:
        cli_query = input("Enter a medical query (or 'exit' to quit): ")
        if cli_query.lower() == 'exit':
            break
        if cli_query:
            print("\n--- Recommendation ---")
            print(generate_recommendation(cli_query))
            print("---------------------\n")

logger.info("Medical RAG System stopped.")
