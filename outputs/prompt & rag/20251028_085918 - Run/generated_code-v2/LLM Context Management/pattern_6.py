import pandas as pd
from datetime import datetime, timedelta
import json
import os

from pydantic import BaseModel
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings # Using OpenAIEmbeddings for consistency with ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import StrOutputParser

# --- Patient DB Schema --- 
class PatientProfile(BaseModel):
    patient_id: str
    name: str
    date_of_birth: str
    gender: str
    chronic_conditions: List[str]
    medications: List[Dict[str, str]]
    allergies: List[str]
    last_updated: str

# --- Data Ingestion --- 
def load_simulated_ehr_data(file_path="simulated_ehr.json"):
    if not os.path.exists(file_path):
        # Create dummy data if file does not exist
        dummy_data = [
            {
                "patient_id": "P001",
                "record_type": "consultation_note",
                "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
                "content": "Patient P001 presented with elevated blood pressure. Discussed lifestyle modifications and initiated Lisinopril 10mg daily."
            },
            {
                "patient_id": "P001",
                "record_type": "lab_result",
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
                "content": "Lab results for P001: Blood Pressure 145/95 mmHg, Cholesterol LDL 130 mg/dL."
            },
            {
                "patient_id": "P001",
                "record_type": "medication_change",
                "timestamp": (datetime.now() - timedelta(days=4)).isoformat(),
                "content": "Patient P001 medication adjusted: Lisinopril increased to 20mg daily due to insufficient BP control."
            },
            {
                "patient_id": "P002",
                "record_type": "consultation_note",
                "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
                "content": "Patient P002 with type 2 diabetes. HbA1c at 8.2%. Advised diet control and metformin 500mg BID."
            }
        ]
        with open(file_path, "w") as f:
            json.dump(dummy_data, f, indent=4)
    with open(file_path, "r") as f:
        return json.load(f)

def load_simulated_lifestyle_data(file_path="simulated_lifestyle.json"):
    if not os.path.exists(file_path):
        dummy_data = [
            {
                "patient_id": "P001",
                "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                "activity": "Daily 30 min walk, reports eating fast food 3 times a week."
            },
            {
                "patient_id": "P002",
                "timestamp": (datetime.now() - timedelta(days=7)).isoformat(),
                "activity": "Sedentary lifestyle, occasional sugary drinks."
            }
        ]
        with open(file_path, "w") as f:
            json.dump(dummy_data, f, indent=4)
    with open(file_path, "r") as f:
        return json.load(f)

def load_simulated_conversations(file_path="simulated_conversations.json"):
    if not os.path.exists(file_path):
        dummy_data = [
            {
                "patient_id": "P001",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "conversation": "Doctor: How are you feeling on the new medication? Patient: A bit better, but still dizzy sometimes. Doctor: We might need to adjust the dosage again."
            },
            {
                "patient_id": "P002",
                "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                "conversation": "Doctor: Have you been monitoring your blood sugar? Patient: Yes, it's been high in the mornings. Doctor: Let's review your diet."
            }
        ]
        with open(file_path, "w") as f:
            json.dump(dummy_data, f, indent=4)
    with open(file_path, "r") as f:
        return json.load(f)

def prepare_patient_data(patient_id: str):
    ehr_data = load_simulated_ehr_data()
    lifestyle_data = load_simulated_lifestyle_data()
    conversations_data = load_simulated_conversations()

    patient_docs = []

    for record in ehr_data:
        if record["patient_id"] == patient_id:
            patient_docs.append({
                "page_content": f"EHR Record ({record['record_type']}): {record['content']}",
                "metadata": {"patient_id": patient_id, "timestamp": datetime.fromisoformat(record["timestamp"]), "source": "ehr", "type": record["record_type"]}
            })

    for record in lifestyle_data:
        if record["patient_id"] == patient_id:
            patient_docs.append({
                "page_content": f"Lifestyle Data: {record['activity']}",
                "metadata": {"patient_id": patient_id, "timestamp": datetime.fromisoformat(record["timestamp"]), "source": "lifestyle"}
            })

    for record in conversations_data:
        if record["patient_id"] == patient_id:
            patient_docs.append({
                "page_content": f"Conversation Transcript: {record['conversation']}",
                "metadata": {"patient_id": patient_id, "timestamp": datetime.fromisoformat(record["timestamp"]), "source": "conversation"}
            })
    return patient_docs

# --- Context Management Layer ---
class ContextManager:
    def __init__(self, embedding_model_name="text-embedding-ada-002", summarizer_model="facebook/bart-large-cnn"):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = OpenAIEmbeddings(model=embedding_model_name) # Ensure OPENAI_API_KEY is set in environment
        self.vectorstore = Chroma(embedding_function=self.embeddings, persist_directory="./chroma_db")
        self.summarizer = pipeline("summarization", model=summarizer_model)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def add_to_memory(self, patient_id: str, documents: List[Dict[str, Any]]):
        texts = [doc["page_content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Filter documents for the specific patient before adding to Chroma
        patient_texts = []
        patient_metadatas = []
        for i in range(len(texts)):
            if metadatas[i].get("patient_id") == patient_id:
                patient_texts.append(texts[i])
                patient_metadatas.append(metadatas[i])

        if patient_texts:
            self.vectorstore.add_texts(texts=patient_texts, metadatas=patient_metadatas)
            self.vectorstore.persist()

    def retrieve_relevant_context(self, patient_id: str, query: str, k=5, recency_weight=0.7, importance_keywords=None):
        # Retrieve initial documents based on semantic similarity
        retrieved_docs = self.vectorstore.similarity_search_with_score(query, k=10) # Retrieve more to filter

        # Filter by patient_id
        patient_specific_docs = [doc for doc, score in retrieved_docs if doc.metadata.get("patient_id") == patient_id]

        # Apply recency and importance prioritization
        prioritized_docs = []
        for doc in patient_specific_docs:
            score = 0
            # Recency factor
            if "timestamp" in doc.metadata:
                time_diff = datetime.now() - doc.metadata["timestamp"]
                recency_score = 1.0 / (1.0 + time_diff.days) # More recent -> higher score
                score += recency_weight * recency_score

            # Importance factor (simple keyword matching for demonstration)
            if importance_keywords:
                for keyword in importance_keywords:
                    if keyword.lower() in doc.page_content.lower():
                        score += (1 - recency_weight) * 0.5 # Add a fixed importance boost
            
            prioritized_docs.append((doc, score))

        # Sort by combined score and take top k
        prioritized_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in prioritized_docs[:k]]

    def summarize_text(self, text: str, max_length=150, min_length=50):
        if not text.strip():
            return ""
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Error during summarization: {e}. Returning original text.")
            return text

# --- LLM Integration & Recommendation Layer ---
class LLMRecommender:
    def __init__(self, context_manager: ContextManager, llm_model_name="gpt-4-0125-preview"):
        self.context_manager = context_manager
        self.llm = ChatOpenAI(model=llm_model_name, temperature=0.2) # Ensure OPENAI_API_KEY is set in environment
        self.rag_chain = self._construct_rag_chain()

    def _construct_rag_chain(self):
        template = """You are a highly experienced medical AI assistant specialized in chronic disease management. Your task is to provide personalized treatment recommendations to doctors based on the patient's comprehensive medical history.

        Use the following context to generate your recommendation. If the context does not contain enough information, state that clearly.

        Context:
        {context}

        Question: {question}
        Personalized treatment recommendation for the doctor:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # This will be replaced by a proper retrieval from the context manager
        def get_context(inputs):
            patient_id = inputs["patient_id"]
            query = inputs["question"]
            retrieved_docs = self.context_manager.retrieve_relevant_context(patient_id, query, k=5)
            # Summarize long documents before passing to LLM
            summarized_context = []
            for doc in retrieved_docs:
                content = doc.page_content
                if len(content) > 500: # Arbitrary length for summarization trigger
                    content = self.context_manager.summarize_text(content)
                summarized_context.append(content)
            return "\n\n".join(summarized_context)

        rag_chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(get_context)
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def get_recommendations(self, patient_id: str, current_query: str):
        response = self.rag_chain.invoke({"patient_id": patient_id, "question": current_query})
        return response

# --- Main Application Logic ---
if __name__ == "__main__":
    # Set your OpenAI API key as an environment variable
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    if os.getenv("OPENAI_API_KEY") is None:
        print("Warning: OPENAI_API_KEY environment variable not set. Please set it for LLM functionality.")

    # Initialize components
    context_manager = ContextManager()
    llm_recommender = LLMRecommender(context_manager)

    # Simulate data ingestion for a patient
    patient_id = "P001"
    print(f"\n--- Ingesting data for patient {patient_id} ---")
    patient_data = prepare_patient_data(patient_id)
    context_manager.add_to_memory(patient_id, patient_data)
    print(f"Added {len(patient_data)} documents to memory for patient {patient_id}.")

    # Example of retrieving context
    print(f"\n--- Retrieving context for patient {patient_id} ---")
    sample_query = "recent blood pressure readings and current medication for hypertension"
    retrieved_context = context_manager.retrieve_relevant_context(patient_id, sample_query, k=2, importance_keywords=["blood pressure", "lisinopril", "hypertension"])
    print("Retrieved Context Snippets (first 200 chars):")
    for doc in retrieved_context:
        print(f"- {doc.metadata['source']} (Score: {doc.metadata.get('score', 'N/A')}): {doc.page_content[:200]}...")

    # Get personalized treatment recommendation
    print(f"\n--- Generating recommendation for patient {patient_id} ---")
    recommendation_query = "Given the patient's history of hypertension, recent blood pressure and medication adjustments, what is the recommended next step for treatment? Consider lifestyle and current symptoms."
    if os.getenv("OPENAI_API_KEY"):
        recommendation = llm_recommender.get_recommendations(patient_id, recommendation_query)
        print("\nPersonalized Treatment Recommendation:")
        print(recommendation)
    else:
        print("Skipping LLM recommendation as OPENAI_API_KEY is not set.")

    # Simulate data ingestion for another patient
    patient_id_2 = "P002"
    print(f"\n--- Ingesting data for patient {patient_id_2} ---")
    patient_data_2 = prepare_patient_data(patient_id_2)
    context_manager.add_to_memory(patient_id_2, patient_data_2)
    print(f"Added {len(patient_data_2)} documents to memory for patient {patient_id_2}.")

    print(f"\n--- Generating recommendation for patient {patient_id_2} ---")
    recommendation_query_2 = "Based on the patient's type 2 diabetes history and recent HbA1c, what are the dietary and medication recommendations?"
    if os.getenv("OPENAI_API_KEY"):
        recommendation_2 = llm_recommender.get_recommendations(patient_id_2, recommendation_query_2)
        print("\nPersonalized Treatment Recommendation for P002:")
        print(recommendation_2)
    else:
        print("Skipping LLM recommendation as OPENAI_API_KEY is not set.")
