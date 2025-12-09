import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

class KnowledgeBaseManager:
    def __init__(self, data_path: str, persist_directory: str):
        self.data_path = data_path
        self.persist_directory = persist_directory
        self.vector_store = None

    def initialize_vector_store(self):
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
            print("Loaded existing vector store.")
        else:
            print("Creating new vector store...")
            loader = DirectoryLoader(self.data_path, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_store = Chroma.from_documents(texts, embeddings, persist_directory=self.persist_directory)
            self.vector_store.persist()
            print("New vector store created and persisted.")

    def retrieve_documents(self, query: str, top_k: int = 4) -> list[Document]:
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store first.")
        return self.vector_store.similarity_search(query, k=top_k)

class LLMResponder:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    def generate_answer(self, query: str, context_documents: list[Document]) -> str:
        context = "\n---\n".join([doc.page_content for doc in context_documents])
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful medical assistant. Use the following context to answer the user's question accurately and concisely. If you don't know the answer, state that you don't have enough information."),
                ("human", "Context: {context}\n\nQuestion: {query}"),
            ]
        )
        
        chain = prompt_template | self.llm
        response = chain.invoke({"context": context, "query": query})
        return response.content

def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in a .env file or as an environment variable.")
        return

    DATA_PATH = "data"
    PERSIST_DIRECTORY = "chroma_db"

    # Ensure data directory exists for example files
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # Create dummy data files if they don't exist
    if not os.path.exists(os.path.join(DATA_PATH, "clinical_guideline_diabetes.txt")):
        with open(os.path.join(DATA_PATH, "clinical_guideline_diabetes.txt"), "w") as f:
            f.write("Clinical Guideline for Type 2 Diabetes Management:\n- Initial treatment often involves lifestyle modifications (diet, exercise).\n- Metformin is typically the first-line pharmacotherapy.\n- Regular monitoring of blood glucose, HbA1c, and renal function is crucial.\n- Insulin therapy may be considered if oral agents are insufficient.")
    
    if not os.path.exists(os.path.join(DATA_PATH, "drug_formulary_hypertension.txt")):
        with open(os.path.join(DATA_PATH, "drug_formulary_hypertension.txt"), "w") as f:
            f.write("Drug Formulary Excerpt: Hypertension Medications\n- ACE Inhibitors (e.g., Lisinopril): First-line for many patients, especially with proteinuria.\n- ARBs (e.g., Losartan): Alternative to ACEIs for patients with cough.\n- Thiazide Diuretics (e.g., Hydrochlorothiazide): Effective first-line, often combined.\n- Calcium Channel Blockers (e.g., Amlodipine): Useful for isolated systolic hypertension.\n- Beta-Blockers (e.g., Metoprolol): Often used for hypertension with co-existing conditions like CAD or heart failure.")

    kb_manager = KnowledgeBaseManager(DATA_PATH, PERSIST_DIRECTORY)
    kb_manager.initialize_vector_store()

    llm_responder = LLMResponder(openai_api_key)

    print("\nClinical Knowledge Assistant Ready! Type your medical queries.\nType 'quit' to exit or 'reload' to re-index the knowledge base.\n")

    while True:
        query = input("Your query: ").strip()

        if query.lower() == "quit":
            break
        elif query.lower() == "reload":
            print("Re-indexing knowledge base...")
            kb_manager.initialize_vector_store()
            print("Knowledge base re-indexed.")
            continue

        if not query:
            continue

        try:
            retrieved_docs = kb_manager.retrieve_documents(query)
            answer = llm_responder.generate_answer(query, retrieved_docs)

            print(f"\nAssistant: {answer}")
            print("\n--- Retrieved Evidence --- ")
            for i, doc in enumerate(retrieved_docs):
                filename = os.path.basename(doc.metadata.get("source", "Unknown Source"))
                print(f"Source {i+1}: {filename}")
                print(f"Snippet: {doc.page_content[:200]}...")
                print("--------------------------")
            print("--------------------------\n")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()