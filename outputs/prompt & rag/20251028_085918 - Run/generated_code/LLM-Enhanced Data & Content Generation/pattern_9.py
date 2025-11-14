import streamlit as st
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

# --- 1. Data Ingestion and Knowledge Base (KB) Construction ---

# Load English tokenizer, tagger, parser and NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Downloading spaCy model 'en_core_web_sm'. This may take a moment.")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Simulate a medical Knowledge Graph using NetworkX
def create_medical_kg():
    kg = nx.DiGraph()

    # Entities
    diseases = ["Hypertension", "Diabetes Mellitus Type 2", "Asthma", "Pneumonia", "Myocardial Infarction"]
    symptoms = ["Headache", "Chest Pain", "Shortness of Breath", "Cough", "Fatigue", "Increased Thirst", "Frequent Urination"]
    drugs = ["Lisinopril", "Metformin", "Albuterol", "Amoxicillin", "Aspirin"]
    tests = ["Blood Pressure Measurement", "Blood Glucose Test", "Spirometry", "Chest X-ray", "ECG"]
    treatments = ["Lifestyle Modification", "Insulin Therapy", "Bronchodilator", "Antibiotics", "Angioplasty"]

    kg.add_nodes_from(diseases, type="disease")
    kg.add_nodes_from(symptoms, type="symptom")
    kg.add_nodes_from(drugs, type="drug")
    kg.add_nodes_from(tests, type="test")
    kg.add_nodes_from(treatments, type="treatment")

    # Relationships (simplified)
    # Symptom-Disease
    kg.add_edge("Headache", "Hypertension", relation="symptom_of")
    kg.add_edge("Chest Pain", "Myocardial Infarction", relation="symptom_of")
    kg.add_edge("Shortness of Breath", "Asthma", relation="symptom_of")
    kg.add_edge("Shortness of Breath", "Pneumonia", relation="symptom_of")
    kg.add_edge("Cough", "Pneumonia", relation="symptom_of")
    kg.add_edge("Fatigue", "Diabetes Mellitus Type 2", relation="symptom_of")
    kg.add_edge("Increased Thirst", "Diabetes Mellitus Type 2", relation="symptom_of")
    kg.add_edge("Frequent Urination", "Diabetes Mellitus Type 2", relation="symptom_of")

    # Disease-Drug
    kg.add_edge("Hypertension", "Lisinopril", relation="treatable_by")
    kg.add_edge("Diabetes Mellitus Type 2", "Metformin", relation="treatable_by")
    kg.add_edge("Asthma", "Albuterol", relation="treatable_by")
    kg.add_edge("Pneumonia", "Amoxicillin", relation="treatable_by")
    kg.add_edge("Myocardial Infarction", "Aspirin", relation="treatable_by")

    # Disease-Test
    kg.add_edge("Hypertension", "Blood Pressure Measurement", relation="diagnosed_by")
    kg.add_edge("Diabetes Mellitus Type 2", "Blood Glucose Test", relation="diagnosed_by")
    kg.add_edge("Asthma", "Spirometry", relation="diagnosed_by")
    kg.add_edge("Pneumonia", "Chest X-ray", relation="diagnosed_by")
    kg.add_edge("Myocardial Infarction", "ECG", relation="diagnosed_by")

    # Disease-Treatment (non-drug)
    kg.add_edge("Hypertension", "Lifestyle Modification", relation="managed_by")
    kg.add_edge("Diabetes Mellitus Type 2", "Insulin Therapy", relation="managed_by")
    kg.add_edge("Asthma", "Bronchodilator", relation="managed_by") # Albuterol is a bronchodilator
    kg.add_edge("Myocardial Infarction", "Angioplasty", relation="managed_by")

    return kg

medical_kg = create_medical_kg()

# Simulate a Vector Database for unstructured medical text
# In a real application, this would be a persistent vector DB (Chroma, Pinecone)
# We use a simple in-memory list for demonstration

# Initialize a sentence transformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def create_vector_db_mock():
    documents = [
        {"text": "Hypertension (high blood pressure) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.", "source": "Mayo Clinic"},
        {"text": "Diabetes mellitus type 2 is a long-term medical condition in which your body does not use insulin properly, leading to high blood sugar levels.", "source": "NIH"},
        {"text": "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out and shortness of breath.", "source": "Mayo Clinic"},
        {"text": "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms can include cough with phlegm or pus, fever, chills, and difficulty breathing.", "source": "CDC"},
        {"text": "A myocardial infarction, commonly known as a heart attack, occurs when blood flow to the heart muscle is severely reduced or stopped.", "source": "WHO"},
        {"text": "Lisinopril is an ACE inhibitor used to treat high blood pressure (hypertension) and heart failure.", "source": "Drug Info"},
        {"text": "Metformin is a medication used to treat type 2 diabetes, particularly in people who are overweight.", "source": "Drug Info"},
        {"text": "Albuterol is a bronchodilator that relaxes muscles in the airways and increases air flow to the lungs. Used to treat or prevent bronchospasm in patients with asthma or other reversible obstructive airway disease.", "source": "Drug Info"},
        {"text": "Amoxicillin is an antibiotic used to treat a number of bacterial infections. It is in the penicillin class of medication.", "source": "Drug Info"},
        {"text": "Aspirin is used to reduce fever and to relieve mild to moderate pain from conditions such as muscle aches, toothaches, common cold, and headaches. It may also be used to reduce pain and swelling in conditions like arthritis. As an antiplatelet, it's used to prevent blood clots.", "source": "Drug Info"},
        {"text": "Lifestyle modifications for hypertension include dietary changes, regular exercise, weight management, and stress reduction.", "source": "Clinical Guidelines"},
        {"text": "Insulin therapy is a treatment for diabetes that replaces the insulin your body isn't making or isn't using properly.", "source": "Endocrine Society"}
    ]
    # Generate embeddings for documents
    embeddings = embedding_model.encode([doc["text"] for doc in documents], convert_to_tensor=False)
    return {"documents": documents, "embeddings": embeddings}

vector_db_mock = create_vector_db_mock()

# --- 2. Unified Retrieval and Reasoning Model (Core LLM Component) ---

def ner_extraction(text):
    doc = nlp(text)
    entities = {"symptoms": [], "diseases": []}
    for ent in doc.ents:
        # Simple heuristic for medical entities - can be improved with custom NER models
        if "symptom" in ent.label_.lower() or "medical condition" in ent.label_.lower() or "disease" in ent.label_.lower():
            entities["diseases"].append(ent.text)
        else:
            entities["symptoms"].append(ent.text) # Generalize other entities as symptoms for simplicity
    return entities


def kg_retrieval(kg, query_entities):
    retrieved_info = []
    for entity_type, entities in query_entities.items():
        for entity in entities:
            if entity in kg.nodes:
                # Find directly related nodes (1-hop)
                for neighbor, data in kg[entity].items():
                    retrieved_info.append(f"Relationship: {entity} {data['relation']} {neighbor}")
                # Find nodes that have a relationship to this entity
                for source, target, data in kg.edges(data=True):
                    if target == entity:
                        retrieved_info.append(f"Relationship: {source} {data['relation']} {target}")

    # More complex multi-hop reasoning could be implemented here using pathfinding algorithms
    return list(set(retrieved_info))

def vector_db_retrieval(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    similarities = cosine_similarity(query_embedding, vector_db_mock["embeddings"])[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_docs = []
    for i in top_indices:
        doc = vector_db_mock["documents"][i]
        retrieved_docs.append(f"Document: {doc['text']} (Source: {doc['source']}, Similarity: {similarities[i]:.2f})")
    return retrieved_docs

def unified_reasoning_llm_mock(patient_data, kg_info, vector_db_info, user_query):
    # This function simulates an LLM's reasoning capabilities.
    # In a real system, this would involve a complex prompt to a powerful LLM
    # with few-shot examples or fine-tuning.

    context = f"Patient Data: {patient_data}\n" \
              f"Knowledge Graph Info: {'; '.join(kg_info)}\n" \
              f"Vector Database Info: {'; '.join(vector_db_info)}\n" \
              f"User Query: {user_query}"

    differential_diagnosis = []
    treatment_plan = []
    justification = "Based on the provided patient data, retrieved medical knowledge, and your query, here is a simulated diagnosis and treatment recommendation."

    # Simple rule-based reasoning based on mock data and context
    if "Hypertension" in str(kg_info) or "high blood pressure" in user_query.lower() or "Headache" in patient_data:
        differential_diagnosis.append("Hypertension")
        treatment_plan.append("Prescribe Lisinopril, advise Lifestyle Modification")
        justification += "\n - Hypertension inferred from related symptoms/drugs in KG."

    if "Diabetes Mellitus Type 2" in str(kg_info) or "high blood sugar" in user_query.lower() or "increased thirst" in patient_data.lower():
        differential_diagnosis.append("Diabetes Mellitus Type 2")
        treatment_plan.append("Prescribe Metformin, advise Insulin Therapy and dietary changes.")
        justification += "\n - Diabetes inferred from symptoms and associated treatments."

    if "Asthma" in str(kg_info) or "shortness of breath" in patient_data.lower() and "wheezing" in patient_data.lower():
        differential_diagnosis.append("Asthma")
        treatment_plan.append("Prescribe Albuterol (bronchodilator).")
        justification += "\n - Asthma indicated by respiratory symptoms and bronchodilator treatment."

    if "Myocardial Infarction" in str(kg_info) or "chest pain" in patient_data.lower():
        differential_diagnosis.append("Myocardial Infarction (Heart Attack)")
        treatment_plan.append("Immediate medical attention, consider Angioplasty and Aspirin for clot prevention.")
        justification += "\n - Myocardial Infarction strongly suggested by chest pain and emergency treatments."

    if not differential_diagnosis:
        differential_diagnosis.append("Further investigation needed. No clear diagnosis based on current information.")
        treatment_plan.append("Recommend general health check-up.")
        justification += "\n - Limited specific information to form a definitive diagnosis."

    return {
        "differential_diagnosis": list(set(differential_diagnosis)),
        "treatment_plan": list(set(treatment_plan)),
        "justification": justification,
        "context_used": context
    }

def rlhf_feedback_mock(feedback_data):
    # This function simulates the RLHF loop. In a real system, this would:
    # 1. Store feedback for later fine-tuning.
    # 2. Potentially trigger re-training or model updates.
    st.success(f"Feedback received: {feedback_data}. This feedback would be used to improve the LLM over time.")
    # For demonstration, we just acknowledge receipt
    pass

# --- 3. User Interface (UI) and Interaction with Streamlit ---

st.title("🧠 Medical Diagnosis and Treatment Recommendation System")
st.markdown("This system assists healthcare professionals by integrating patient data with medical knowledge for differential diagnoses and treatment plans.")

st.header("Patient Information")
patient_symptoms = st.text_area("Enter patient's symptoms (e.g., 'severe chest pain, shortness of breath, headache'):",
                                 "patient presents with headache and mild fatigue")
patient_history = st.text_area("Enter patient's medical history or other relevant data:",
                                 "45-year-old male, no known allergies")

user_query = st.text_input("Ask a specific medical question or guidance:",
                           "What are the possible diagnoses and treatments?")

if st.button("Get Recommendations"):
    if not patient_symptoms and not user_query:
        st.warning("Please enter patient symptoms or a specific query.")
    else:
        st.subheader("Processing Request...")

        # 1. Preprocessing and Entity Extraction
        all_text = patient_symptoms + " " + patient_history + " " + user_query
        entities = ner_extraction(all_text)
        st.write(f"Extracted Entities: {entities}")

        # 2. Contextualized Retrieval
        st.subheader("Retrieving Knowledge...")
        kg_retrieved_info = kg_retrieval(medical_kg, entities)
        st.write(f"Knowledge Graph Retrieval (relevant relationships):\n - {';\n - '.join(kg_retrieved_info) if kg_retrieved_info else 'No direct KG relationships found.'}")

        vector_db_retrieved_docs = vector_db_retrieval(all_text)
        st.write(f"Vector Database Retrieval (similar documents):\n - {';\n - '.join(vector_db_retrieved_docs)}")

        # 3. Unified Reasoning
        st.subheader("Generating Diagnosis and Treatment Plan...")
        patient_full_data = f"Symptoms: {patient_symptoms}. History: {patient_history}"
        llm_response = unified_reasoning_llm_mock(patient_full_data, kg_retrieved_info, vector_db_retrieved_docs, user_query)

        st.success("Recommendations Generated!")
        st.subheader("Differential Diagnosis:")
        for diagnosis in llm_response["differential_diagnosis"]:
            st.write(f"- {diagnosis}")

        st.subheader("Recommended Treatment Plan:")
        for treatment in llm_response["treatment_plan"]:
            st.write(f"- {treatment}")

        st.subheader("Justification:")
        st.info(llm_response["justification"])

        with st.expander("View Full Context Used for Reasoning"):
            st.code(llm_response["context_used"], language="text")

        # 4. RLHF Feedback Mechanism
        st.subheader("Provide Feedback (for RLHF)")
        feedback_option = st.radio(
            "Was this recommendation helpful and accurate?",
            ('Yes, it was helpful', 'No, it was inaccurate', 'Partially accurate', 'Irrelevant')
        )
        feedback_text = st.text_area("Optional: Provide more detailed feedback:")
        if st.button("Submit Feedback"):
            rlhf_feedback_mock({"overall": feedback_option, "details": feedback_text,
                                 "patient_data": patient_full_data, "query": user_query,
                                 "system_output": llm_response})

st.markdown("""
---
**Disclaimer:** This is a demonstration system for AI project code generation. It uses mocked LLM reasoning and a simplified knowledge base for illustrative purposes. **It is not intended for actual medical diagnosis or treatment. Always consult with qualified healthcare professionals.**
""")

# --- Instructions to run this application: ---
# 1. Ensure you have the required libraries installed:
#    pip install streamlit networkx sentence-transformers scikit-learn spacy
#    python -m spacy download en_core_web_sm
# 2. Save the code as `medical_diagnosis_app.py`.
# 3. Run from your terminal: `streamlit run medical_diagnosis_app.py`
