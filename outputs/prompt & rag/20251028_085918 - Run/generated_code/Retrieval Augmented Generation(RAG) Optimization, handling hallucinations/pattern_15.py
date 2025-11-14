import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp # Placeholder, can be replaced with other LLMs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import functools

class MedicalRAGAssistant:
    """
    A Medical Information and Diagnostic Assistant utilizing Retrieval-Augmented Generation (RAG).
    This system integrates external medical knowledge to provide accurate and reliable answers,
    mitigating common LLM issues like factual inaccuracies and hallucinations.
    """

    def __init__(self, model_path=None, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        load_dotenv() # Load environment variables, e.g., API keys if using external LLMs

        # 1. Initialize Embedding Model for document vectorization
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # 2. Initialize LLM (Language Model)
        # If model_path is provided, it attempts to load a local LlamaCpp model (.gguf file).
        # Otherwise, a MockLLM is used for demonstration purposes, allowing immediate execution.
        # For production, consider using robust LLM providers (e.g., OpenAI, Google Gemini) or fine-tuned local models.
        if model_path:
             # LlamaCpp requires a locally downloaded .gguf model file.
             # n_gpu_layers=-1 attempts to offload all layers to GPU if available.
             self.llm = LlamaCpp(model_path=model_path, temperature=0.1, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        else:
             # Mock LLM for demonstration if no model_path is provided
             class MockLLM:
                 def invoke(self, prompt):
                     # Simulate responses based on common medical queries
                     if "treatment for diabetes" in prompt.lower():
                         return "Based on the provided context, common treatments for diabetes include lifestyle modifications, oral medications like metformin, and insulin therapy. Specific treatment plans depend on the type and severity of diabetes, as well as individual patient factors."
                     if "hypertension" in prompt.lower() and "what is" in prompt.lower():
                         return "Hypertension, or high blood pressure, is a medical condition where the blood pressure in the arteries is persistently elevated. It can lead to serious health problems like heart disease and stroke."
                     if "symptoms of a heart attack" in prompt.lower():
                         return "The symptoms of a heart attack commonly include chest pain or discomfort, shortness of breath, pain in the left arm, lightheadedness, and sweating. Immediate medical attention is crucial."
                     if "how does metformin work" in prompt.lower():
                         return "Metformin primarily works by decreasing glucose production in the liver and improving insulin sensitivity in the body's cells."
                     return "This is a mock response from the LLM based on your query and the provided context. For complex queries, a real LLM would provide a more nuanced answer."
             self.llm = MockLLM()

        self.vector_store = None
        self.retriever = None
        self.rag_chain = None

        # 3. Define the Prompt Template for the RAG system
        # This template instructs the LLM to use the retrieved context for answering.
        self.template = """You are a highly accurate and trustworthy medical assistant.
        Use the following retrieved context to answer the question.
        If you don't know the answer based on the provided context, truthfully say you don't know.

        Context: {context}

        Question: {question}

        Answer:"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def _format_docs(self, docs) -> str:
        """Formats retrieved documents into a single string for LLM context."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Basic Caching mechanism using a decorator
    def _cache(func):
        cache = {}
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a simple hashable key from args and kwargs
            key = str(args) + str(sorted(kwargs.items()))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        return wrapper

    @_cache
    def load_and_index_medical_data(self, texts: list[str], medical_doc_ids: list[str] = None):
        """
        Simulates loading and indexing medical data into a FAISS vector store.
        In a real-world application, 'texts' would be sourced from medical databases
        (e.g., PubMed abstracts, clinical guidelines, drug formularies).
        """
        if not texts:
            print("No texts provided for indexing.")
            return

        documents = []
        for i, text in enumerate(texts):
            metadata = {"source": f"medical_doc_{medical_doc_ids[i]}" if medical_doc_ids else f"medical_doc_{i+1}"}
            documents.append(Document(page_content=text, metadata=metadata))

        # Split documents into smaller chunks for effective retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Create FAISS vector store from document chunks and embeddings
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        # Configure retriever to fetch top 5 most relevant documents
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print(f"Indexed {len(splits)} chunks into FAISS vector store.")

    def build_rag_chain(self):
        """Constructs the RAG chain using LangChain Expression Language (LCEL)."""
        if not self.retriever:
            raise ValueError("Vector store and retriever must be initialized before building the RAG chain.")

        # LCEL chain for RAG:
        # 1. Parallel execution: retrieve context and pass through the original question.
        # 2. Format context and combine with question in the prompt.
        # 3. Invoke the LLM with the formulated prompt.
        # 4. Parse the LLM's string output.
        self.rag_chain = (
            RunnableParallel(
                {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("RAG chain built successfully.")

    def _adaptive_decision_making(self, query: str) -> str:
        """
        Conceptual method for adaptive decision-making: when to retrieve, when to generate, when to abstain.
        This simple heuristic checks for medical keywords to decide if retrieval is necessary.
        In a more advanced system, this could involve a semantic router, classifier, or confidence scores.
        """
        if self.retriever:
            medical_keywords = ["diagnosis", "treatment", "symptom", "disease", "medication", "condition", "cause", "therapy", "prognosis"]
            if any(keyword in query.lower() for keyword in medical_keywords):
                return "retrieve_and_generate" # Use RAG for medical queries
            else:
                return "generate_only" # For general questions, just use the LLM without retrieval
        return "generate_only" # Fallback if no retriever is initialized

    def ask(self, query: str):
        """
        Asks the RAG assistant a question, incorporating adaptive decision-making.
        """
        decision = self._adaptive_decision_making(query)
        print(f"[System Decision]: For query '{query}', the system decided to: {decision.replace('_', ' ').upper()}")

        if decision == "retrieve_and_generate":
            if not self.rag_chain:
                self.build_rag_chain() # Build chain if not already built
            print("  >> Using RAG chain for response...")
            return self.rag_chain.invoke(query)
        elif decision == "generate_only":
            print("  >> Generating response directly from LLM (no retrieval)...")
            # For pure generation, a different prompt might be optimal
            return self.llm.invoke(f"Answer the following question: {query}")
        else: # Covers an 'abstain' scenario, though not explicitly returned by current heuristic
            return "I cannot provide an answer for this query at the moment, or it falls outside my medical domain expertise."

if __name__ == "__main__":
    # --- Example Usage ---
    # Prepare some simulated medical data.
    # In a real application, this data would come from actual medical databases.
    medical_texts_data = [
        "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy. With diabetes, your body either doesn't make enough insulin or can't effectively use the insulin it does make.",
        "Type 1 diabetes is an autoimmune reaction where the body attacks itself by mistake. This stops your body from making insulin. Type 2 diabetes is when your body doesn't use insulin well and can't keep blood sugar at normal levels. It develops over many years and is usually diagnosed in adults.",
        "Common treatments for Type 2 diabetes include lifestyle changes (diet and exercise), oral medications such as Metformin, and sometimes insulin injections. Regular monitoring of blood glucose levels is crucial.",
        "Hypertension, also known as high blood pressure, is a serious medical condition that significantly increases the risks of heart, brain, kidney and other diseases. A blood pressure reading above 140/90 mmHg is generally considered hypertension. Symptoms often include headaches, shortness of breath, nosebleeds, and dizziness, though many people have no symptoms.",
        "Symptoms of a heart attack include chest pain, shortness of breath, pain radiating to the left arm, jaw, or back, lightheadedness, and cold sweats. Immediate medical attention is required. Aspirin can be given if a heart attack is suspected, under medical supervision.",
        "Metformin is a first-line medication for type 2 diabetes. It works by decreasing glucose production by the liver and improving insulin sensitivity in peripheral tissues. It does not cause weight gain and can even lead to modest weight loss."
    ]
    medical_document_ids = ["DM_001", "DM_002", "DM_003", "HT_001", "HA_001", "DRUG_001"]

    # Initialize the RAG Assistant.
    # To use a local LlamaCpp model, provide the path: e.g., model_path="./llama-2-7b-chat.gguf"
    assistant = MedicalRAGAssistant(model_path=None) # Using MockLLM for immediate execution
    
    # Load and index the simulated medical data.
    assistant.load_and_index_medical_data(medical_texts_data, medical_document_ids)
    
    print("\n--- Testing Queries ---")

    # Query 1: Medical question requiring retrieval
    print("\nQuery: What are common treatments for diabetes?")
    response1 = assistant.ask("What are common treatments for diabetes?")
    print(f"Assistant: {response1}")

    # Query 2: Medical question requiring retrieval
    print("\nQuery: What is hypertension and what are its symptoms?")
    response2 = assistant.ask("What is hypertension and what are its symptoms?")
    print(f"Assistant: {response2}")

    # Query 3: Medical question requiring retrieval
    print("\nQuery: What are the symptoms of a heart attack and what should be done?")
    response3 = assistant.ask("What are the symptoms of a heart attack and what should be done?")
    print(f"Assistant: {response3}")

    # Query 4: General question (should trigger 'generate_only' if the adaptive decision works)
    print("\nQuery: Tell me a fun fact about space.")
    response4 = assistant.ask("Tell me a fun fact about space.")
    print(f"Assistant: {response4}")

    # Query 5: Specific drug mechanism
    print("\nQuery: How does Metformin work to treat diabetes?")
    response5 = assistant.ask("How does Metformin work to treat diabetes?")
    print(f"Assistant: {response5}")

    # Query 6: Another medical query to show RAG in action
    print("\nQuery: What is Type 1 diabetes?")
    response6 = assistant.ask("What is Type 1 diabetes?")
    print(f"Assistant: {response6}")
