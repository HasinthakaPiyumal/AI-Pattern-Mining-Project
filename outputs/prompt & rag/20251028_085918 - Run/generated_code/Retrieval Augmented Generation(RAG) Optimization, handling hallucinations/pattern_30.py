
import streamlit as st
import functools
from typing import List, Dict, Any
import random

try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    st.warning("Sentence-Transformers not found. Using mock embeddings. Please install 'sentence-transformers' for better functionality.")
    class MockEmbeddingModel:
        def encode(self, texts, **kwargs):
            return [[random.random() for _ in range(384)] for _ in texts]
    embedding_model = MockEmbeddingModel()


# Mock Electronic Health Records (EHR)
mock_ehr_data = {
    "patient_A123": {
        "name": "Alice Smith",
        "age": 45,
        "conditions": ["Hypertension", "Type 2 Diabetes"],
        "medications": ["Lisinopril", "Metformin"],
        "allergies": ["Penicillin"],
        "last_visit": "2023-10-26",
        "notes": "Patient A123 shows stable blood pressure, HBA1C slightly elevated."
    },
    "patient_B456": {
        "name": "Bob Johnson",
        "age": 62,
        "conditions": ["Coronary Artery Disease"],
        "medications": ["Aspirin", "Atorvastatin"],
        "allergies": [],
        "last_visit": "2023-11-15",
        "notes": "Follow-up for CAD, stable angina, advise lifestyle modifications."
    }
}

# Mock Medical Research Knowledge Base (Vector Database Simulation)
class MockVectorDB:
    def __init__(self):
        self.documents = [] # list of (text, embedding)
        self.texts = []

    def add_document(self, text: str):
        embedding = embedding_model.encode([text])[0]
        self.documents.append((text, embedding))
        self.texts.append(text)

    def search(self, query_embedding, top_k: int = 3) -> List[str]:
        if not self.documents:
            return []
        
        similarities = []
        for i, (text, doc_embedding) in enumerate(self.documents):
            similarity = sum(query_embedding[j] * doc_embedding[j] for j in range(len(query_embedding)))
            similarities.append((similarity, text))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in similarities[:top_k]]

mock_medical_research_db = MockVectorDB()
mock_medical_research_db.add_document("Recent study on Type 2 Diabetes management highlights benefits of SGLT2 inhibitors in reducing cardiovascular events.")
mock_medical_research_db.add_document("Guidelines for hypertension treatment emphasize lifestyle changes and individualized medication regimens.")
mock_medical_research_db.add_document("Mechanism of action of ACE inhibitors: blocks conversion of angiotensin I to angiotensin II, leading to vasodilation.")
mock_medical_research_db.add_document("Symptoms of myocardial infarction include chest pain, shortness of breath, and arm pain.")
mock_medical_research_db.add_document("Metformin is a common oral medication for type 2 diabetes, primarily reducing glucose production by the liver.")


# Query Classifier
def classify_query_intent(query: str) -> str:
    lower_query = query.lower()
    if "patient" in lower_query or "ehr" in lower_query or any(pid in lower_query for pid in mock_ehr_data.keys()):
        return "patient_specific"
    return "general_medical"

# Retrievers
class MedicalResearchRetriever:
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = embedding_model.encode([query])[0]
        return mock_medical_research_db.search(query_embedding, top_k=top_k)

class EHRRetriever:
    def retrieve(self, patient_id: str) -> Dict[str, Any]:
        return mock_ehr_data.get(patient_id, {})

# Mock LLM and Self-Reflection Module
class MockLLM:
    @functools.lru_cache(maxsize=128)
    def generate_response(self, query: str, context: List[str]) -> Dict[str, Any]:
        full_context = " ".join(context) if context else "No specific context provided."
        
        # Simple rule-based generation and confidence for demonstration
        if "patient_A123" in query and "hypertension" in query.lower() and context:
            answer = f"Based on patient A123's record and general medical knowledge, their hypertension is managed with Lisinopril. Context: {full_context[:100]}..."
            confidence = 0.95
        elif "metformin" in query.lower() and context:
            answer = f"Metformin is a medication for Type 2 Diabetes that primarily reduces hepatic glucose production. Context: {full_context[:100]}..."
            confidence = 0.90
        elif "type 2 diabetes" in query.lower() and context:
            answer = f"Type 2 Diabetes management often involves SGLT2 inhibitors and lifestyle changes, as seen in recent studies. Context: {full_context[:100]}..."
            confidence = 0.88
        elif "patient_A123" in query and not context:
            answer = "To provide accurate information for patient A123, specific medical context from their EHR is needed. I can retrieve general details if requested, but a precise medical query related to their records would yield better results."
            confidence = 0.60
        elif "patient_B456" in query and "cardiac" in query.lower() and context:
            answer = f"Patient B456 has Coronary Artery Disease and is on Aspirin and Atorvastatin. Advised lifestyle modifications for stable angina. Context: {full_context[:100]}..."
            confidence = 0.92
        elif "patient" in query and not any(pid in query for pid in mock_ehr_data.keys()):
             answer = "Please specify a valid patient ID for EHR-specific queries."
             confidence = 0.75
        else:
            answer = f"I am processing your general medical query. Here's what I found from the available context: {full_context}"
            confidence = 0.70
            if not context:
                answer = "I couldn't find very specific context for your query. Here is a general statement. Please try rephrasing or providing more details."
                confidence = 0.55
        
        return {"answer": answer, "confidence": confidence}


# RAG Orchestrator
class AMIA_RAG_Orchestrator:
    def __init__(self):
        self.medical_retriever = MedicalResearchRetriever()
        self.ehr_retriever = EHRRetriever()
        self.llm = MockLLM()
        self.conversation_history = []

    def _get_patient_id_from_query(self, query: str) -> str:
        for pid in mock_ehr_data.keys():
            if pid.lower() in query.lower():
                return pid
        return None

    def process_query(self, query: str, iteration: int = 0) -> Dict[str, Any]:
        if iteration > 2: # Limit iterative refinement to avoid loops
            return {"answer": "Could not confidently answer after multiple refinements. Please rephrase your query or seek human expertise.", "confidence": 0.40, "retrieved_context": []}

        intent = classify_query_intent(query)
        retrieved_context = []
        
        if intent == "patient_specific":
            patient_id = self._get_patient_id_from_query(query)
            if patient_id:
                ehr_info = self.ehr_retriever.retrieve(patient_id)
                if ehr_info:
                    retrieved_context.append(f"EHR for {patient_id}: {ehr_info}")
                    # Also try to get general medical context related to patient's conditions
                    for condition in ehr_info.get("conditions", []):
                        retrieved_context.extend(self.medical_retriever.retrieve(condition, top_k=1))
                else:
                    retrieved_context.append(f"No EHR found for patient ID: {patient_id}.")
            else:
                retrieved_context.append("Could not extract a patient ID from the query for EHR retrieval.")

        if not retrieved_context or intent == "general_medical":
            medical_research_results = self.medical_retriever.retrieve(query)
            retrieved_context.extend(medical_research_results)

        llm_output = self.llm.generate_response(query, tuple(retrieved_context)) # tuple for caching hashability
        answer = llm_output["answer"]
        confidence = llm_output["confidence"]

        if confidence < 0.75: # Threshold for triggering refinement/abstention
            if iteration == 0: # First attempt, try to refine
                st.info("Low confidence detected. Attempting to refine retrieval...")
                # For simplicity, refinement here is a re-run with potential more context. 
                # A real system might generate a follow-up question or re-rank results.
                return self.process_query(query, iteration + 1)
            else:
                answer += " (Confidence too low; consider human review or provide more details.)"
        
        self.conversation_history.append({"query": query, "answer": answer, "confidence": confidence, "context": retrieved_context})
        return {"answer": answer, "confidence": confidence, "retrieved_context": retrieved_context}

# Streamlit UI
st.title("Adaptive Medical Information Assistant (AMIA)")
st.write("An AI-powered assistant for healthcare professionals leveraging Adaptive RAG.")

if 'amia_orchestrator' not in st.session_state:
    st.session_state.amia_orchestrator = AMIA_RAG_Orchestrator()

user_query = st.text_input("Enter your medical query:", "What are the latest treatments for Type 2 Diabetes?")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Processing your query..."):
            response = st.session_state.amia_orchestrator.process_query(user_query)
            
            st.subheader("AMIA's Response:")
            st.write(response["answer"])
            st.metric(label="Confidence Score", value=f"{response['confidence']:.2f}")
            
            with st.expander("Retrieved Context"):
                if response["retrieved_context"]:
                    for i, context_item in enumerate(response["retrieved_context"]):
                        st.write(f"- {context_item}")
                else:
                    st.write("No specific context retrieved.")
    else:
        st.warning("Please enter a query.")

st.sidebar.subheader("Conversation History")
for i, entry in enumerate(reversed(st.session_state.amia_orchestrator.conversation_history)):
    st.sidebar.markdown(f"**Q{len(st.session_state.amia_orchestrator.conversation_history) - i}:** {entry['query'][:50]}...")
    st.sidebar.text(f"Conf: {entry['confidence']:.2f}")
