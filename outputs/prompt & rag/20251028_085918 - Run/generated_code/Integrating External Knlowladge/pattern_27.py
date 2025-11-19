import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import chromadb
import networkx as nx
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")

load_dotenv()

class NewsScraper:
    def scrape_article(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("title").get_text() if soup.find("title") else "No Title"
            paragraphs = soup.find_all("p")
            article_text = " ".join([p.get_text() for p in paragraphs])
            return {"url": url, "title": title, "text": article_text}
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

class VectorDBManager:
    def __init__(self, collection_name="news_articles"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_document(self, doc_id, text, metadata=None):
        embedding = self.model.encode(text).tolist()
        self.collection.add(embeddings=[embedding], documents=[text], metadatas=[metadata], ids=[doc_id])

    def query_documents(self, query_text, n_results=5):
        query_embedding = self.model.encode(query_text).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results

class KnowledgeGraphManager:
    def __init__(self):
        self.graph = nx.Graph()

    def add_entity(self, entity_name, entity_type="general"):
        self.graph.add_node(entity_name, type=entity_type)

    def add_relation(self, entity1, relation, entity2):
        if not self.graph.has_node(entity1):
            self.add_entity(entity1)
        if not self.graph.has_node(entity2):
            self.add_entity(entity2)
        self.graph.add_edge(entity1, entity2, relation=relation)

    def get_relations(self, entity):
        if not self.graph.has_node(entity):
            return []
        relations = []
        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            if edge_data and "relation" in edge_data:
                relations.append((entity, edge_data["relation"], neighbor))
        return relations

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, prompt_template, **kwargs):
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = chain.invoke(kwargs)
        return response.content

class RAGSystem:
    def __init__(self, vector_db_manager, llm_service):
        self.vector_db_manager = vector_db_manager
        self.llm_service = llm_service

    def retrieve_and_generate(self, query):
        retrieved_docs = self.vector_db_manager.query_documents(query)
        context = " ".join(retrieved_docs["documents"][0]) if retrieved_docs["documents"] else "No relevant context found."

        prompt_template = """
        Based on the following context, answer the query:

        Context: {context}

        Query: {query}

        Answer:
        """
        return self.llm_service.generate_response(prompt_template, context=context, query=query)

class FactChecker:
    def __init__(self, llm_service, kg_manager):
        self.llm_service = llm_service
        self.kg_manager = kg_manager

    def check_fact(self, claim, context):
        # Placeholder for sophisticated entity linking
        entities_in_claim = re.findall(r"\b[A-Z][a-z]+\b", claim) # Simple heuristic for proper nouns
        kg_evidence = []
        for entity in entities_in_claim:
            kg_evidence.extend(self.kg_manager.get_relations(entity))

        kg_evidence_str = "\n".join([f"{e1} {r} {e2}" for e1, r, e2 in kg_evidence])

        prompt_template = """
        Evaluate the following claim based on the provided context and knowledge graph evidence. 
        State whether the claim is TRUE, FALSE, or UNVERIFIED, and provide a brief explanation.

        Claim: {claim}
        Context: {context}
        Knowledge Graph Evidence:
        {kg_evidence_str}

        Verdict:
        """
        return self.llm_service.generate_response(prompt_template, claim=claim, context=context, kg_evidence_str=kg_evidence_str)

class Summarizer:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def summarize_article(self, article_text, length="brief"):
        prompt_template = """
        Summarize the following article {length}. Focus on key information and main points.

        Article: {article_text}

        Summary:
        """
        return self.llm_service.generate_response(prompt_template, article_text=article_text, length=length)

def main():
    st.set_page_config(layout="wide")
    st.title("AI-Powered News Aggregator & Fact-Checker")

    scraper = NewsScraper()
    preprocessor = DataPreprocessor()
    vector_db_manager = VectorDBManager()
    kg_manager = KnowledgeGraphManager()
    llm_service = LLMService()
    rag_system = RAGSystem(vector_db_manager, llm_service)
    fact_checker = FactChecker(llm_service, kg_manager)
    summarizer = Summarizer(llm_service)

    # Initialize KG with some sample data
    kg_manager.add_entity("Joe Biden", "Person")
    kg_manager.add_entity("United States", "Country")
    kg_manager.add_relation("Joe Biden", "PresidentOf", "United States")
    kg_manager.add_entity("Paris", "City")
    kg_manager.add_entity("France", "Country")
    kg_manager.add_relation("Paris", "CapitalOf", "France")

    st.sidebar.header("News Article Ingestion")
    article_url = st.sidebar.text_input("Enter News Article URL:", "https://www.bbc.com/news/world-us-canada-68971844")
    if st.sidebar.button("Scrape & Process Article"):
        if article_url:
            with st.spinner("Scraping and processing article..."):
                article_data = scraper.scrape_article(article_url)
                if article_data:
                    processed_text = preprocessor.preprocess_text(article_data["text"])
                    doc_id = str(hash(article_data["url"])) # Simple ID generation
                    vector_db_manager.add_document(doc_id, processed_text, {"title": article_data["title"], "url": article_data["url"]})
                    st.session_state["current_article"] = article_data
                    st.session_state["current_processed_text"] = processed_text
                    st.success("Article scraped and added to knowledge base.")
                else:
                    st.error("Failed to scrape article.")
        else:
            st.sidebar.warning("Please enter an article URL.")

    if "current_article" in st.session_state:
        st.header("Current Article")
        st.subheader(st.session_state["current_article"]["title"])
        st.write(f"URL: {st.session_state["current_article"]["url"]}")
        st.expander("Full Text").write(st.session_state["current_article"]["text"])

        st.subheader("Article Summary")
        summary_length = st.radio("Select summary length:", ["brief", "detailed"])
        if st.button("Generate Summary"):
            with st.spinner(f"Generating {summary_length} summary..."):
                summary = summarizer.summarize_article(st.session_state["current_article"]["text"], length=summary_length)
                st.write(summary)
                st.session_state["current_summary"] = summary

        st.subheader("Fact-Check a Claim")
        claim_to_check = st.text_input("Enter a claim to fact-check:", "Joe Biden is the President of France.")
        if st.button("Fact-Check"):
            if claim_to_check:
                with st.spinner("Checking claim..."):
                    # Use the original article text as context for fact-checking
                    fact_check_result = fact_checker.check_fact(claim_to_check, st.session_state["current_article"]["text"])
                    st.info(fact_check_result)
            else:
                st.warning("Please enter a claim to check.")

        st.subheader("Ask the RAG System")
        rag_query = st.text_input("Ask a question about the news content:", "What is the main topic of the article?")
        if st.button("Get RAG Answer"):
            if rag_query:
                with st.spinner("Retrieving and generating answer..."):
                    rag_answer = rag_system.retrieve_and_generate(rag_query)
                    st.write(rag_answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()