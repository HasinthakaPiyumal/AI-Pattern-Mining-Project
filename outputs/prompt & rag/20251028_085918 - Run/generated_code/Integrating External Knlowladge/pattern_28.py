# utils.py
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# data_ingestion.py
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

def ingest_medical_data(data_sources, embedding_model):
    chroma_client = Client(Settings(persist_directory="./chroma_db"))
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model.model_name_or_path)

    try:
        chroma_client.delete_collection(name="medical_knowledge")
    except:
        pass

    collection = chroma_client.get_or_create_collection(
        name="medical_knowledge",
        embedding_function=embedding_function
    )

    documents = []
    metadatas = []
    ids = []
    doc_id_counter = 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for source_name, content in data_sources.items():
        if isinstance(content, str):
            docs = text_splitter.split_documents([Document(page_content=content, metadata={"source": source_name})])
            for doc in docs:
                documents.append(doc.page_content)
                metadatas.append(doc.metadata)
                ids.append(f"doc_{doc_id_counter}")
                doc_id_counter += 1
        elif isinstance(content, list):
            for item in content:
                if item.startswith("http"):
                    try:
                        response = requests.get(item)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            text = ' '.join(p.get_text() for p in soup.find_all('p'))
                            docs = text_splitter.split_documents([Document(page_content=text, metadata={"source": item})])
                            for doc in docs:
                                documents.append(doc.page_content)
                                metadatas.append(doc.metadata)
                                ids.append(f"doc_{doc_id_counter}")
                                doc_id_counter += 1
                    except Exception as e:
                        print(f"Error fetching {item}: {e}")
                else:
                    docs = text_splitter.split_documents([Document(page_content=item, metadata={"source": source_name})])
                    for doc in docs:
                        documents.append(doc.page_content)
                        metadatas.append(doc.metadata)
                        ids.append(f"doc_{doc_id_counter}")
                        doc_id_counter += 1

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Ingested {len(documents)} document chunks into ChromaDB.")
    else:
        print("No documents to ingest.")
    return collection

# knowledge_graph.py
import networkx as nx
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'. This may take a moment...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_medical_fact(self, subject, relation, obj):
        self.graph.add_edge(subject, obj, relation=relation)

    def extract_and_add_facts_from_text(self, text):
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "DRUG", "MEDICAL_CONDITION", "ORGANIZATION"]]

        if len(entities) >= 2:
            self.add_medical_fact(entities[0], "related_to", entities[1])
            print(f"Added simple fact from text: {entities[0]} related_to {entities[1]}")
        elif len(entities) == 1 and "disease" in text.lower() and "treatment" in text.lower():
             self.add_medical_fact(entities[0], "has_potential_treatment", "General Treatment Strategy")
             print(f"Added simple fact from text: {entities[0]} has_potential_treatment General Treatment Strategy")

    def query_related_info(self, entity):
        if entity in self.graph:
            related_nodes = list(self.graph.neighbors(entity))
            relations = [(entity, self.graph[entity][neighbor]['relation'], neighbor) for neighbor in related_nodes]
            return relations
        return []

# rag_system.py
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

class MockLLM:
    def invoke(self, prompt):
        return f"Mocked LLM response to: {prompt}"

class RAGSystem:
    def __init__(self, embedding_model):
        chroma_client = Client(Settings(persist_directory="./chroma_db"))
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model.model_name_or_path)
        self.vector_store = chroma_client.get_or_create_collection(
            name="medical_knowledge",
            embedding_function=embedding_function
        )
        self.retriever = self.vector_store.as_retriever()
        self.llm = MockLLM()

        self.prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:
        {context}

        Question: {question}
        """)

        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def retrieve_and_generate(self, question):
        return self.rag_chain.invoke(question)

# web_agent.py
import requests
from bs4 import BeautifulSoup

class WebAgent:
    def __init__(self, trusted_domains=None):
        self.trusted_domains = trusted_domains if trusted_domains else [
            "pubmed.ncbi.nlm.nih.gov",
            "www.nejm.org",
            "www.thelancet.com",
            "www.who.int",
            "www.cdc.gov"
        ]
        self.mock_llm_for_synthesis = MockLLM()

    def _is_trusted_url(self, url):
        import urllib.parse
        parsed_url = urllib.parse.urlparse(url)
        return any(domain in parsed_url.netloc for domain in self.trusted_domains)

    def search_and_synthesize(self, query, num_results=3):
        print(f"WebAgent: Searching for '{query}' on trusted medical sources...")
        mock_search_results = [
            f"https://pubmed.ncbi.nlm.nih.gov/mock_article_1_{query.replace(' ', '_')}",
            f"https://www.nejm.org/mock_article_2_{query.replace(' ', '_')}",
            f"https://www.who.int/news/item/mock_summary_3_{query.replace(' ', '_')}"
        ]
        
        relevant_content = []
        for url in mock_search_results[:num_results]:
            if self._is_trusted_url(url):
                print(f"WebAgent: Accessing trusted URL: {url}")
                try:
                    mock_html_content = f"""
                    <html>
                    <body>
                        <h1>Mock Article Title for {query}</h1>
                        <p>This is a simulated paragraph about {query} from {url}. It provides some key information and findings.</p>
                        <p>More details specific to medical research and {query} are discussed here.</p>
                    </body>
                    </html>
                    """
                    soup = BeautifulSoup(mock_html_content, 'html.parser')
                    text = ' '.join(p.get_text() for p in soup.find_all('p'))
                    relevant_content.append(f"Content from {url}:\n{text}")
                except Exception as e:
                    print(f"WebAgent: Error processing {url}: {e}")
            else:
                print(f"WebAgent: Skipping untrusted URL: {url}")
        
        if relevant_content:
            combined_content = "\n\n".join(relevant_content)
            synthesis_prompt = f"Synthesize the following medical information regarding '{query}':\n\n{combined_content}\n\nProvide a concise summary."
            print("WebAgent: Synthesizing information...")
            return self.mock_llm_for_synthesis.invoke(synthesis_prompt)
        else:
            return "WebAgent: Could not find relevant information from trusted sources."

# main.py
def main():
    print("Initializing Clinical Knowledge Assistant...")

    embedding_model = load_embedding_model()
    print("Embedding model loaded.")

    mock_medical_data = {
        "article_1": "A recent study on COVID-19 vaccines showed high efficacy against severe disease. Common side effects included fever and fatigue.",
        "article_2": "Metformin is a first-line treatment for type 2 diabetes, primarily reducing glucose production by the liver. Patients should monitor kidney function.",
        "guideline_cardiology": "New guidelines suggest statins for primary prevention of cardiovascular disease in high-risk individuals.",
        "pubmed_url_example": [
            "https://pubmed.ncbi.nlm.nih.gov/34567890/"
        ]
    }
    vector_db_collection = ingest_medical_data(mock_medical_data, embedding_model)
    print("Medical data ingested into vector database.")

    rag_system = RAGSystem(embedding_model)
    print("RAG System initialized.")

    kg = MedicalKnowledgeGraph()
    kg.add_medical_fact("COVID-19", "has_treatment", "Vaccine")
    kg.add_medical_fact("Metformin", "treats", "Type 2 Diabetes")
    kg.add_medical_fact("Statins", "prevents", "Cardiovascular Disease")
    kg.extract_and_add_facts_from_text("Insulin therapy is crucial for Type 1 Diabetes management.")
    print("Knowledge Graph initialized and populated.")

    web_agent = WebAgent()
    print("Web Agent initialized.")

    print("\nReady to assist. Type 'exit' to quit.")

    while True:
        query = input("\nMedical professional query: ")
        if query.lower() == 'exit':
            break

        print("\n--- RAG System Response ---")
        rag_response = rag_system.retrieve_and_generate(query)
        print(f"RAG: {rag_response}")

        print("\n--- Knowledge Graph Response ---")
        doc = nlp(query)
        entities_in_query = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "DRUG", "MEDICAL_CONDITION"]]
        kg_results = []
        if entities_in_query:
            for entity in entities_in_query:
                related_info = kg.query_related_info(entity)
                if related_info:
                    kg_results.extend(related_info)
        
        if kg_results:
            print("KG: Related information found:")
            for s, p, o in kg_results:
                print(f"- {s} {p} {o}")
        else:
            print("KG: No direct related information found in the knowledge graph for this query.")
        
        print("\n--- Web Agent Response (if needed) ---")
        if "latest" in query.lower() or "new findings" in query.lower() or len(kg_results) == 0 and "treatment" in query.lower():
            web_synthesis = web_agent.search_and_synthesize(query)
            print(f"WebAgent: {web_synthesis}")
        else:
            print("WebAgent: Not triggered for this query (based on simple heuristic).")

if __name__ == '__main__':
    main()