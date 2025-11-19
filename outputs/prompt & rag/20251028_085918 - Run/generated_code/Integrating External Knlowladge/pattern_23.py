import streamlit as st
import os
import requests
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

# Placeholder for LangChain/OpenAI components
# In a real application, you would initialize your LLM and retriever here
class MockOpenAI:
    def __init__(self):
        pass
    def generate(self, prompt):
        return f"LLM Response to: {prompt}"

class MockRetriever:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def retrieve(self, query, top_k=3):
        # In a real scenario, this would involve vector search
        results = [doc for doc in self.knowledge_base if query.lower() in doc.lower()]
        return results[:top_k]

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
PINE_CONE_API_KEY = os.getenv("PINE_CONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINE_CONE_ENVIRONMENT = os.getenv("PINE_CONE_ENVIRONMENT", "YOUR_PINECONE_ENVIRONMENT")

# --- Embedding Model (Sentence-Transformers) ---
# Using a pre-trained model for embeddings
@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

tokenizer, embedding_model = load_embedding_model()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with np.no_grad():
        model_output = embedding_model(**inputs)
    sentence_embeddings = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return sentence_embeddings

# --- Dummy Knowledge Base and External API Integrations ---
dummy_medical_documents = [
    "Latest research on Type 2 Diabetes treatment: SGLT2 inhibitors and GLP-1 receptor agonists show promising results.",
    "Guidelines for managing hypertension: Lifestyle modifications are crucial, followed by ACE inhibitors or ARBs.",
    "Symptoms and diagnosis of appendicitis: Acute abdominal pain, nausea, vomiting, and fever. Ultrasound or CT scan for confirmation.",
    "Drug information for Metformin: Used for Type 2 Diabetes, common side effects include gastrointestinal issues.",
    "Clinical trial results for a new Alzheimer's drug indicate a slowdown in cognitive decline in early-stage patients.",
    "Updated recommendations for COVID-19 vaccination in immunocompromised individuals."
]

class DummyVectorDB:
    def __init__(self, documents, embedding_func):
        self.documents = documents
        self.embeddings = [embedding_func(doc) for doc in documents]

    def query(self, query_text, top_k=3):
        query_embedding = get_embedding(query_text)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]

dummy_vector_db = DummyVectorDB(dummy_medical_documents, get_embedding)

def fetch_from_pubmed(query):
    # Simulate API call to PubMed
    if "diabetes" in query.lower():
        return "[PubMed] Recent article: 'Novel therapies for diabetic retinopathy.'"
    elif "hypertension" in query.lower():
        return "[PubMed] Review: 'Advances in blood pressure management.'"
    return "[PubMed] No specific recent articles found for your query."

def fetch_from_clinicaltrials(query):
    # Simulate API call to ClinicalTrials.gov
    if "alzheimer" in query.lower():
        return "[ClinicalTrials.gov] Ongoing trial: 'Phase 3 study on amyloid-beta targeting drug.'"
    return "[ClinicalTrials.gov] No relevant clinical trials found."

def fetch_from_fda_drug_database(query):
    # Simulate API call to FDA Drug Database
    if "metformin" in query.lower():
        return "[FDA] Metformin (Glucophage): Indicated for Type 2 Diabetes. Common adverse reactions include diarrhea, nausea, vomiting."
    return "[FDA] Drug information not found."

# --- Modular Knowledge Consolidation & Processing Pipelines ---
nlp = spacy.load("en_core_web_sm") # For basic entity linking

def entity_linking(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "DATE", "NORP"] or ent.text.lower() in ["diabetes", "hypertension", "appendicitis", "metformin", "alzheimer"]]
    return list(set(entities))

def evidence_chaining(retrieved_info):
    # Simple logic: combine and clean retrieved information
    combined_info = "\n".join(retrieved_info)
    return combined_info

def information_filtering(retrieved_info, query):
    # Simple filtering: prioritize items directly mentioning query terms
    filtered_info = [item for item in retrieved_info if query.lower() in item.lower()]
    if not filtered_info and retrieved_info:
        return [retrieved_info[0]] # Fallback to top result if no direct match
    return filtered_info

# --- Browser-Assisted LLM (Simplified) ---
def browser_assisted_search(query, allowed_domains=None):
    if allowed_domains is None:
        allowed_domains = ["example.com", "medicaljournal.com"]
    
    st.warning(f"Simulating web search for: '{query}' on trusted domains. This is a placeholder for a real browser agent.")
    # In a real app, this would use Selenium/Playwright with LLM agent logic.
    # For this demo, we just simulate a request to a dummy medical site.
    for domain in allowed_domains:
        try:
            response = requests.get(f"https://{domain}/search?q={query.replace(' ', '+')}", timeout=2)
            if response.status_code == 200:
                return f"[Browser Search on {domain}] Found relevant snippet for '{query}'. (Status: 200 OK)"
        except requests.exceptions.RequestException:
            pass
    return "[Browser Search] No specific information found via simulated web search."

# --- Streamlit UI and Main Logic ---
st.set_page_config(page_title="Medical Information Assistant", layout="wide")
st.title("👨‍⚕️ Medical Information Assistant")
st.markdown("This assistant helps doctors get up-to-date and accurate medical information by leveraging LLMs and external knowledge sources.")

# Initialize LLM and Mock Retriever (LangChain components)
llm = MockOpenAI()
mock_langchain_retriever = MockRetriever(dummy_medical_documents)

query = st.text_area("Enter your medical query here:", height=100)

if st.button("Get Medical Information"):
    if query:
        st.subheader("Processing your query...")

        # 1. Query Preprocessing & Entity Linking
        st.write("\n🔍 **Step 1: Entity Linking & Preprocessing**")
        extracted_entities = entity_linking(query)
        st.write(f"Extracted entities: {', '.join(extracted_entities) if extracted_entities else 'None'}")
        processed_query = query # For now, simple pass-through

        # 2. Knowledge Retrieval (Vector DB + External APIs)
        st.write("\n📚 **Step 2: Knowledge Retrieval**")
        retrieved_docs_vd = dummy_vector_db.query(processed_query, top_k=5)
        st.write("From Vector Database (Semantic Search):")
        for doc, score in retrieved_docs_vd:
            st.write(f"- {doc} (Score: {score:.2f})")

        api_results = []
        api_results.append(fetch_from_pubmed(processed_query))
        api_results.append(fetch_from_clinicaltrials(processed_query))
        api_results.append(fetch_from_fda_drug_database(processed_query))
        st.write("\nFrom External Medical APIs:")
        for res in api_results:
            st.write(f"- {res}")
        
        all_retrieved_info = [doc for doc, _ in retrieved_docs_vd] + api_results

        # 3. Modular Knowledge Consolidation
        st.write("\n⚙️ **Step 3: Knowledge Consolidation**")
        filtered_info = information_filtering(all_retrieved_info, processed_query)
        st.write("Filtered relevant information:")
        for info in filtered_info:
            st.write(f"- {info}")
        
        evidence_chained_info = evidence_chaining(filtered_info)
        st.write("\nConsolidated Evidence:")
        st.write(evidence_chained_info)

        # 4. LLM Augmentation & Response Generation
        st.write("\n🧠 **Step 4: LLM Augmentation & Response Generation**")
        if not evidence_chained_info or len(evidence_chained_info.strip()) < 50: # Arbitrary length check
            st.info("Initial knowledge retrieval was limited. Attempting browser-assisted search...")
            browser_info = browser_assisted_search(processed_query)
            st.write(browser_info)
            final_context = f"Query: {processed_query}\nContext from Knowledge Base: {evidence_chained_info}\nContext from Web Search: {browser_info}"
        else:
            final_context = f"Query: {processed_query}\nContext from Knowledge Base: {evidence_chained_info}"

        llm_prompt = f"Based on the following context, provide a comprehensive and accurate medical response for a doctor:\n\n{final_context}\n\nMedical Assistant:"
        
        llm_response = llm.generate(llm_prompt)
        
        st.subheader("\n✅ Medical Assistant's Response:")
        st.success(llm_response)

    else:
        st.warning("Please enter a medical query.")

st.markdown("\n---")
st.info("Disclaimer: This is a demonstration of an AI Medical Assistant architecture. It is not intended for actual medical diagnosis or treatment. Always consult with a qualified medical professional.")
