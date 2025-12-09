import os
import datetime
from typing import List, Dict, Any
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model for SpaCy...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.7)

# Customer Memory Module (Chroma Vector Database)
class CustomerMemory:
    def __init__(self):
        self.vectorstore = Chroma(
            collection_name="customer_interactions",
            embedding_function=lambda texts: embedding_model.encode(texts).tolist()
        )
        self.customer_profiles: Dict[str, Dict[str, Any]] = {}
        self.customer_interactions: Dict[str, List[Dict[str, Any]]] = {}

    def add_interaction(self, customer_id: str, text: str, timestamp: datetime.datetime):
        doc = Document(page_content=text, metadata={"customer_id": customer_id, "timestamp": timestamp.isoformat()})
        self.vectorstore.add_documents([doc])
        if customer_id not in self.customer_interactions:
            self.customer_interactions[customer_id] = []
        self.customer_interactions[customer_id].append({"text": text, "timestamp": timestamp})

    def get_customer_history(self, customer_id: str) -> List[Dict[str, Any]]:
        # Retrieve from in-memory dictionary for full historical context
        return self.customer_interactions.get(customer_id, [])

    def get_relevant_docs_from_vectorstore(self, customer_id: str, query: str, k: int = 5) -> List[Document]:
        # Need to implement filtering by customer_id for Chroma
        # For now, it will retrieve top k and then filter. A real system would use metadata filters in Chroma.
        results = self.vectorstore.similarity_search(query, k=10) # Retrieve more to filter down
        filtered_results = [doc for doc in results if doc.metadata.get("customer_id") == customer_id]
        return filtered_results[:k]

    def update_customer_profile(self, customer_id: str, profile_data: Dict[str, Any]):
        if customer_id not in self.customer_profiles:
            self.customer_profiles[customer_id] = {}
        self.customer_profiles[customer_id].update(profile_data)

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        return self.customer_profiles.get(customer_id, {})

customer_memory = CustomerMemory()

# Data Ingestion & Preprocessing (Simulated)
def simulate_data_ingestion():
    customer_memory.add_interaction("cust_001", "Customer complained about slow internet speed last week. Troubleshooting steps: rebooted router, checked cables. Issue resolved temporarily.", datetime.datetime(2023, 10, 20, 10, 0, 0))
    customer_memory.add_interaction("cust_001", "Follow-up call: Internet slow again. Scheduled technician visit.", datetime.datetime(2023, 10, 22, 14, 30, 0))
    customer_memory.add_interaction("cust_001", "Technician visited. Replaced faulty modem. Internet speed now stable.", datetime.datetime(2023, 10, 25, 9, 0, 0))
    customer_memory.add_interaction("cust_001", "Customer asks about upgrading their internet plan to fiber optic.", datetime.datetime(2023, 11, 1, 11, 15, 0))
    customer_memory.update_customer_profile("cust_001", {"plan": "Standard DSL", "modem_model": "XYZ-2000"})

    customer_memory.add_interaction("cust_002", "Issue with billing. Customer claims double charge for October. Investigated and refunded one charge.", datetime.datetime(2023, 10, 15, 16, 0, 0))
    customer_memory.add_interaction("cust_002", "New query: Customer wants to know current balance.", datetime.datetime(2023, 11, 3, 9, 45, 0))

simulate_data_ingestion()

# Prioritization and Selection Module
def prioritize_and_select_history(
    customer_history: List[Dict[str, Any]], current_query: str, top_n: int = 3
) -> List[str]:
    docs_with_scores = []
    current_query_doc = nlp(current_query)

    for entry in customer_history:
        text = entry["text"]
        timestamp = entry["timestamp"]
        doc = nlp(text)

        # Recency score
        time_diff = (datetime.datetime.now() - timestamp).total_seconds()
        recency_score = 1.0 / (1.0 + time_diff / (3600 * 24 * 7)) # Halves every week

        # Keyword Relevance (simple overlap)
        query_keywords = {token.lemma_.lower() for token in current_query_doc if not token.is_stop and not token.is_punct}
        history_keywords = {token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct}
        keyword_overlap = len(query_keywords.intersection(history_keywords))
        relevance_score = keyword_overlap / max(len(query_keywords), 1)

        # Combine scores (simple weighted average)
        total_score = (recency_score * 0.4) + (relevance_score * 0.6)
        docs_with_scores.append((text, total_score))

    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in docs_with_scores[:top_n]]

# Summarization and Compression Module
def summarize_text(text: str) -> str:
    if not text.strip():
        return ""
    try:
        summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Error summarizing: {e}"

def extract_entities(text: str) -> Dict[str, List[str]]:
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

# API Models
class QueryRequest(BaseModel):
    customer_id: str
    query: str

class SupportResponse(BaseModel):
    suggestion: str
    context_used: Dict[str, Any]

app = FastAPI()

@app.post("/support/query", response_model=SupportResponse)
async def handle_support_query(request: QueryRequest):
    customer_id = request.customer_id
    current_query = request.query

    # 1. Retrieve raw customer history
    full_history = customer_memory.get_customer_history(customer_id)

    # 2. Prioritization and Selection
    prioritized_history_texts = prioritize_and_select_history(full_history, current_query, top_n=5)
    combined_prioritized_text = "\n".join(prioritized_history_texts)

    # 3. Summarization and Compression
    summarized_history = summarize_text(combined_prioritized_text)
    extracted_entities = extract_entities(combined_prioritized_text + " " + current_query)
    customer_profile = customer_memory.get_customer_profile(customer_id)

    # 4. Retrieval from Vector Store for additional context (e.g., resolutions)
    relevant_docs = customer_memory.get_relevant_docs_from_vectorstore(customer_id, current_query, k=2)
    relevant_knowledge = "\n".join([doc.page_content for doc in relevant_docs])

    # 5. Prompt Engineering & LLM Interaction
    system_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "You are an intelligent customer support assistant. "
                "Your goal is to provide accurate and helpful suggestions to the customer support agent "
                "based on the current customer query and their historical context. "
                "Focus on the most relevant information."
            ),
            HumanMessage(content=(
                "Customer ID: {customer_id}\n"
                "Customer Profile: {customer_profile}\n"
                "Historical Summary: {history_summary}\n"
                "Key Entities from History/Query: {entities}\n"
                "Relevant Knowledge/Resolutions: {relevant_knowledge}\n\n"
                "Current Customer Query: {current_query}\n\n"
                "Please provide a concise and actionable suggestion for the support agent:"
            ))
        ]
    )

    prompt = system_prompt_template.format_messages(
        customer_id=customer_id,
        customer_profile=customer_profile,
        history_summary=summarized_history,
        entities=extracted_entities,
        relevant_knowledge=relevant_knowledge,
        current_query=current_query
    )
    
    try:
        ai_suggestion = llm.invoke(prompt).content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    context_used = {
        "customer_profile": customer_profile,
        "summarized_history": summarized_history,
        "extracted_entities": extracted_entities,
        "relevant_knowledge": relevant_knowledge,
        "prioritized_history_texts": prioritized_history_texts
    }

    return SupportResponse(suggestion=ai_suggestion, context_used=context_used)

# To run the app:
# 1. pip install fastapi uvicorn transformers sentence-transformers spacy langchain-openai "chromadb>=0.4.14" pydantic
# 2. python -m spacy download en_core_web_sm
# 3. uvicorn customer_support_assistant:app --reload --port 8000
# 4. Set OPENAI_API_KEY environment variable. If not set, it will use a placeholder and likely fail OpenAI calls.))
