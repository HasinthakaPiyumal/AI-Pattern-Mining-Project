from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document # Explicitly import Document
from operator import itemgetter

# 1. Medical Knowledge Base (Data Layer)
class MedicalKnowledgeBase:
    def __init__(self):
        self.documents = []

    def add_document(self, title: str, content: str):
        """
        Adds a medical document to the knowledge base.
        Each document is stored as a dictionary with 'title' and 'page_content'.
        'page_content' is used by LangChain document loaders.
        """
        self.documents.append({"metadata": {"title": title}, "page_content": content})
        print(f"Added document: {title}")

    def get_all_documents(self):
        """
        Retrieves all documents from the knowledge base.
        """
        return self.documents

# Mock LLM class to avoid dependency on actual LLM API keys for demonstration
class MockLLM:
    def invoke(self, messages):
        user_query = messages[-1].content if isinstance(messages[-1], HumanMessage) else ""
        retrieved_context = "No context found." # Default if not explicitly passed

        # Try to extract context from previous messages if available
        # In LCEL, the context is usually passed as part of the system message.
        for msg in reversed(messages):
            if isinstance(msg, str) and "Context:" in msg:
                parts = msg.split("Context:", 1)
                if len(parts) > 1:
                    retrieved_context = parts[1].strip()
                break
            elif isinstance(msg, HumanMessage) and "Context:" in msg.content:
                 parts = msg.content.split("Context:", 1)
                 if len(parts) > 1:
                    retrieved_context = parts[1].strip()
                 break
            elif isinstance(msg, dict) and 'context' in msg: # For direct dictionary input in RunnableLambda
                retrieved_context = msg['context']
                break
            elif hasattr(msg, 'content') and "Context:" in msg.content: # For ChatPromptTemplate output
                parts = msg.content.split("Context:", 1)
                if len(parts) > 1:
                    retrieved_context = parts[1].strip()
                break
        
        # The RAG chain directly passes context into the system prompt
        # We need to extract it from the constructed prompt if it's there
        if messages and isinstance(messages[0], tuple) and messages[0][0] == "system":
            system_prompt_content = messages[0][1]
            if "Context:" in system_prompt_content:
                parts = system_prompt_content.split("Context:", 1)
                if len(parts) > 1:
                    retrieved_context = parts[1].split("\n\n")[0].strip() # Get context before the question placeholder


        # Simple mock response logic
        if "diabetes" in user_query.lower() and "type 2" in user_query.lower() and "metformin" in retrieved_context.lower():
            return AIMessage(content=f"Based on the provided context about Type 2 Diabetes, Metformin is a common medication used in its treatment. Context excerpt: {retrieved_context[:100]}...")
        elif "hypertension" in user_query.lower() and "medication" in user_query.lower() and "ace inhibitors" in retrieved_context.lower():
            return AIMessage(content=f"For hypertension, ACE inhibitors are a class of medications often prescribed, as mentioned in the context. Context excerpt: {retrieved_context[:100]}...")
        elif "asthma" in user_query.lower() and "treatment" in user_query.lower() and ("inhalers" in retrieved_context.lower() or "corticosteroids" in retrieved_context.lower()):
            return AIMessage(content=f"Based on the context regarding asthma, treatments often include quick-relief inhalers and long-term control medications like inhaled corticosteroids. Context excerpt: {retrieved_context[:100]}...")
        elif retrieved_context != "No context found." and len(retrieved_context) > 50:
             return AIMessage(content=f"I found relevant information: {retrieved_context[:150]}... Based on your query, here's a synthesized response. (Mock LLM response)")
        else:
            return AIMessage(content=f"I cannot provide a specific medical answer without more context or external retrieval. Query: '{user_query}' (Mock LLM response)")


# 2. Retrieval-Augmented Generation (RAG) Core
class ClinicalRAGSystem:
    def __init__(self, medical_knowledge_base):
        self.medical_kb = medical_knowledge_base
        
        print("Initializing embeddings...")
        # Using a local sentence-transformer model for embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print("Initializing text splitter...")
        # Text splitter for breaking documents into smaller chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        print("Loading and processing documents...")
        # Get documents from the medical knowledge base
        docs_raw = self.medical_kb.get_all_documents()
        # LangChain expects Document objects, so we adapt the dicts
        langchain_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs_raw]

        # Split documents into chunks
        docs = self.text_splitter.split_documents(langchain_docs)
        
        print(f"Created {len(docs)} document chunks.")

        print("Creating vector store (Chroma in-memory)...")
        # Create a vector store from the document chunks
        self.vectorstore = Chroma.from_documents(documents=docs, embedding=self.embeddings)
        
        print("Initializing retriever...")
        # Initialize the retriever from the vector store
        self.retriever = self.vectorstore.as_retriever()

        print("Initializing mock LLM...")
        # Initialize the mock LLM
        self.llm = MockLLM()

        print("Setting up RAG prompt template...")
        # Define the RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful medical assistant. Use only the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context: {context}"),
            ("human", "{question}")
        ])

        print("Setting up RAG chain...")
        # Define the RAG chain using LangChain Expression Language (LCEL)
        self.rag_chain = (
            RunnablePassthrough.assign(context=itemgetter("question") | self.retriever | RunnableLambda(self._format_docs))
            | self.rag_prompt
            | self.llm
        )

        self.cache = {}
        print("ClinicalRAGSystem initialized successfully.")

    def _format_docs(self, docs):
        """
        Helper function to format retrieved documents into a single string for the prompt.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, user_query: str) -> str:
        """
        Queries the RAG system for medical insights.
        Implements simple caching and adaptive retrieval (currently always retrieves).
        """
        print(f"Received query: {user_query}")

        # Check cache first
        if user_query in self.cache:
            print("Returning cached response.")
            return self.cache[user_query]

        # Invoke the RAG chain
        try:
            response = self.rag_chain.invoke({"question": user_query})
            response_content = response.content # Extract content from AIMessage
            self.cache[user_query] = response_content
            print("Generated new response and cached it.")
            return response_content
        except Exception as e:
            print(f"Error during RAG chain invocation: {e}")
            return f"An error occurred while processing your request: {e}"

# 3. API Layer (User Interface/Interaction)

# Define a Pydantic model for incoming query requests
class QueryRequest(BaseModel):
    query: str

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Insight RAG Assistant",
    description="An AI assistant for healthcare professionals providing accurate, up-to-date, and contextually relevant medical information."
)

# Initialize medical knowledge base and RAG system globally
medical_kb = MedicalKnowledgeBase()

# Populate the knowledge base with some sample medical data
medical_kb.add_document("Diabetes Mellitus Type 2 Guidelines", "Type 2 diabetes is a chronic condition characterized by high blood sugar levels. It often develops due to a combination of insulin resistance and impaired insulin secretion. Management typically involves lifestyle modifications (diet, exercise), oral medications like metformin, sulfonylureas, GLP-1 receptor agonists, SGLT2 inhibitors, and sometimes insulin therapy. Regular monitoring of blood glucose, HbA1c, and screening for complications are essential. Early diagnosis and intervention can prevent or delay severe complications such as cardiovascular disease, nephropathy, and neuropathy.")
medical_kb.add_document("Hypertension Management Strategies", "Hypertension (high blood pressure) is a major risk factor for cardiovascular disease. Treatment aims to reduce blood pressure to target levels, typically below 130/80 mmHg for most adults. Lifestyle modifications, including a low-sodium diet (e.g., DASH diet), regular physical activity, moderation of alcohol intake, and weight management, are foundational. Pharmacological interventions include diuretics (e.g., thiazides), ACE inhibitors, Angiotensin Receptor Blockers (ARBs), beta-blockers, and calcium channel blockers. The choice of medication depends on patient comorbidities and individual response. Regular blood pressure monitoring is crucial.")
medical_kb.add_document("Common Cold vs. Flu", "The common cold and flu (influenza) are both respiratory illnesses, but they are caused by different viruses. The flu is generally worse than a cold, and symptoms are more intense. Colds usually involve a runny or stuffy nose, sore throat, and cough. Flu symptoms often include fever, body aches, extreme tiredness, and dry cough. While colds rarely lead to serious health problems, the flu can result in complications like pneumonia, bronchitis, and sinus infections. Vaccination is effective against the flu but not the common cold.")
medical_kb.add_document("Asthma Treatment Options", "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, shortness of breath, chest tightness, and coughing. Treatment typically involves a combination of quick-relief (rescue) medications and long-term control medications. Quick-relief inhalers (e.g., short-acting beta-agonists like albuterol) are used for immediate symptom relief. Long-term control medications, such as inhaled corticosteroids, long-acting beta-agonists (LABAs), leukotriene modifiers, and biologics, are taken daily to prevent symptoms and reduce airway inflammation. A personalized asthma action plan is crucial for effective management.")
medical_kb.add_document("COVID-19 Symptoms and Prevention", "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Common symptoms include fever, cough, fatigue, loss of taste or smell, sore throat, and headache. Severe cases can lead to pneumonia and acute respiratory distress syndrome. Prevention strategies include vaccination, wearing masks in crowded indoor settings, maintaining physical distancing, frequent handwashing, and avoiding close contact with sick individuals. Regular testing and isolation for positive cases are also vital to control the spread.")


clinical_rag_assistant = ClinicalRAGSystem(medical_kb)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Clinical Insight RAG Assistant API. Use the /query endpoint to get medical information."}

@app.post("/query", response_model=dict, tags=["RAG Assistant"])
async def get_clinical_insight(request: QueryRequest):
    """
    Submit a medical query and receive a comprehensive, contextually relevant insight.
    The system leverages Retrieval-Augmented Generation to provide accurate information
    from its medical knowledge base.
    """
    print(f"API received query: {request.query}")
    response = clinical_rag_assistant.query(request.query)
    return {"query": request.query, "response": response}

# To run this application:
# 1. Save this file as `app.py`.
# 2. Install necessary packages: `pip install fastapi uvicorn pydantic langchain-community langchain-text-splitters langchain-core sentence-transformers chromadb`
#    Note: For `sentence-transformers`, you might need `pip install torch` separately if not already installed.
# 3. Run from your terminal: `uvicorn app:app --reload`
# 4. Access the API documentation at http://127.0.0.1:8000/docs