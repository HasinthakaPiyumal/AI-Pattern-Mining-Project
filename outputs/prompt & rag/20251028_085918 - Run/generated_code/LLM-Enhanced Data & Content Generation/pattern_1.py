
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# For embedding and vector store
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# For LangChain RAG pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Data Layer Simulation ---
# Simulated Medical Knowledge Base
MEDICAL_KNOWLEDGE_BASE = [
    "Malaria: Caused by Plasmodium parasite, transmitted by mosquitoes. Symptoms include fever, chills, headache, muscle pain. Treatment: Antimalarial drugs like Artemisinin-based Combination Therapies (ACTs).",
    "Influenza (Flu): Viral infection affecting respiratory system. Symptoms: fever, cough, sore throat, body aches. Treatment: Antivirals (Oseltamivir), rest, fluids.",
    "Common Cold: Mild viral infection of nose and throat. Symptoms: runny nose, sneezing, mild sore throat. Treatment: Rest, fluids, over-the-counter medications.",
    "Diabetes Mellitus Type 2: Chronic condition affecting how body processes blood sugar. Symptoms: increased thirst, frequent urination, fatigue, blurred vision. Treatment: Diet, exercise, oral medications (Metformin), insulin.",
    "Hypertension (High Blood Pressure): Chronic medical condition in which blood pressure in arteries is elevated. Often no symptoms. Risk factors: obesity, high sodium diet, stress. Treatment: Lifestyle changes, ACE inhibitors, diuretics.",
    "Asthma: Chronic inflammatory disease of the airways. Symptoms: wheezing, coughing, shortness of breath, chest tightness. Triggers: allergens, exercise. Treatment: Inhalers (bronchodilators, corticosteroids).",
    "Pneumonia: Infection that inflames air sacs in one or both lungs. Symptoms: cough with phlegm, fever, chills, difficulty breathing. Causes: bacteria, viruses, fungi. Treatment: Antibiotics (bacterial), antivirals (viral), rest.",
    "Migraine: Severe headache accompanied by throbbing pain, sensitivity to light/sound, nausea. Triggers: stress, certain foods. Treatment: Pain relievers, triptans.",
    "Appendicitis: Inflammation of the appendix. Symptoms: sharp pain in lower right abdomen, nausea, vomiting, fever. Treatment: Appendectomy (surgical removal).",
    "Gastroenteritis (Stomach Flu): Inflammation of stomach and intestines. Symptoms: diarrhea, vomiting, abdominal cramps, fever. Causes: viruses, bacteria. Treatment: Rest, hydration, bland diet."
]

class PatientData(BaseModel):
    symptoms: str
    medical_history: str = "None"
    vital_signs: Dict[str, str] = {} # e.g., {"temperature": "38.5 C", "blood_pressure": "140/90 mmHg"}

# --- 2. Retrieval Component ---
# Initialize embedding model
# Using 'all-MiniLM-L6-v2' for a lightweight and effective embedding model
embedding_model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)

# Custom embedding function for ChromaDB
class SentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, texts: embedding_functions.Documents) -> embedding_functions.Embeddings:
        return embedding_model.encode(texts).tolist()

sentence_transformer_ef = SentenceTransformerEmbeddingFunction()

# Initialize ChromaDB in-memory client
chroma_client = chromadb.Client()
collection_name = "medical_knowledge"

def initialize_vector_db():
    try:
        chroma_client.delete_collection(name=collection_name) # Ensure a fresh collection for demo
    except Exception:
        pass # Ignore if collection doesn't exist

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )

    # Split documents for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(MEDICAL_KNOWLEDGE_BASE)

    # Add documents to ChromaDB
    collection.add(
        documents=[doc.page_content for doc in docs],
        ids=[f"doc_{i}" for i in range(len(docs))]
    )
    print(f"Initialized ChromaDB with {len(docs)} documents.")
    return collection

# Initialize the vector database globally for the app
medical_collection = initialize_vector_db()

def retrieve_medical_knowledge(query: str, k: int = 3) -> List[str]:
    results = medical_collection.query(
        query_texts=[query],
        n_results=k
    )
    return results["documents"][0] if results["documents"] else []

# --- 3. Reasoning Component (LLM Integration - Simulated) ---
# For a real application, replace this with an actual LLM client (e.g., OpenAI, HuggingFace Llama etc.)
# Here, we use a simple mock to demonstrate the RAG chain
class MockLLM:
    def invoke(self, prompt: str) -> str:
        # Simulate LLM response based on keywords and structure
        if "differential diagnosis" in prompt.lower() and "treatment recommendations" in prompt.lower():
            # Extract relevant info from the prompt
            patient_info_start = prompt.find("Patient Information:")
            patient_info_end = prompt.find("Medical Knowledge:")
            patient_info = prompt[patient_info_start:patient_info_end].strip() if patient_info_start != -1 and patient_info_end != -1 else ""

            medical_knowledge_start = prompt.find("Medical Knowledge:")
            medical_knowledge_content = prompt[medical_knowledge_start:].strip() if medical_knowledge_start != -1 else ""

            # Simple keyword matching for a mock diagnosis
            diagnosis_lines = []
            treatment_lines = []
            
            if "fever" in patient_info.lower() and "chills" in patient_info.lower() and "headache" in patient_info.lower():
                if "mosquito" in medical_knowledge_content.lower():
                    diagnosis_lines.append("Possible Malaria.")
                    treatment_lines.append("Consider Antimalarial drugs like ACTs.")
                elif "cough" in patient_info.lower():
                    diagnosis_lines.append("Possible Influenza or Pneumonia.")
                    treatment_lines.append("Suggest rest, fluids, and potentially antivirals/antibiotics based on further tests.")
            elif "abdominal pain" in patient_info.lower() and "nausea" in patient_info.lower():
                if "appendix" in medical_knowledge_content.lower():
                    diagnosis_lines.append("Possible Appendicitis.")
                    treatment_lines.append("Emergency surgical consultation (appendectomy).")
            elif "thirst" in patient_info.lower() and "urination" in patient_info.lower():
                 if "diabetes" in medical_knowledge_content.lower():
                    diagnosis_lines.append("Possible Diabetes Mellitus Type 2.")
                    treatment_lines.append("Recommend lifestyle changes, diet, and medication consultation.")
            
            if not diagnosis_lines:
                diagnosis_lines.append("Further investigation required. Based on provided information and retrieved knowledge, potential conditions include:")
                for doc in medical_knowledge_content.split('\n'):
                    if doc.strip() and not doc.startswith("Medical Knowledge:"):
                        diagnosis_lines.append(f"- {doc.split(':')[0].strip()}")
                
            if not treatment_lines:
                 treatment_lines.append("General recommendations: Consult a medical professional for accurate diagnosis and personalized treatment plan.")


            return (f"Differential Diagnosis:\n{chr(10).join(diagnosis_lines)}\n\n"
                    f"Treatment Recommendations:\n{chr(10).join(treatment_lines)}")
        return "I need more specific instructions for diagnosis."

mock_llm = MockLLM()

# Define the prompt template for the LLM
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a highly skilled AI diagnostic assistant for healthcare professionals. Your task is to provide a differential diagnosis and treatment recommendations based on patient data and retrieved medical knowledge. Be thorough, consider all relevant information, and always emphasize that this is an AI assistant and a human doctor's final decision is paramount."),
        ("human", """
            Patient Information:
            Symptoms: {symptoms}
            Medical History: {medical_history}
            Vital Signs: {vital_signs}

            Medical Knowledge:
            {context}

            Based on the patient's information and the provided medical knowledge, provide a differential diagnosis and suggest appropriate treatment recommendations.
        """)
    ]
)

# --- Construct the RAG Chain ---
def format_docs(docs: List[str]) -> str:
    return "\n\n".join(docs)

rag_chain = (
    RunnablePassthrough.assign(context=lambda x: retrieve_medical_knowledge(f"{x['symptoms']} {x['medical_history']}", k=5))
    | prompt_template
    | mock_llm # In a real app, this would be an actual LLM instance (e.g., OpenAI, HuggingFace)
    | StrOutputParser()
)


# --- 4. Application Layer (FastAPI) ---
app = FastAPI(
    title="AI Diagnostic Assistant",
    description="A diagnostic AI assistant for healthcare professionals leveraging unified retrieval and reasoning."
)

@app.post("/diagnose")
async def get_diagnosis(patient_data: PatientData):
    """
    Provides a differential diagnosis and treatment recommendations based on patient data.
    """
    try:
        # Prepare input for the RAG chain
        input_for_llm = {
            "symptoms": patient_data.symptoms,
            "medical_history": patient_data.medical_history,
            "vital_signs": str(patient_data.vital_signs) # Convert dict to string for prompt
        }

        # Invoke the RAG chain
        diagnosis_output = rag_chain.invoke(input_for_llm)

        return {"diagnosis": diagnosis_output}
    except Exception as e:
        return {"error": str(e)}

# To run this application:
# 1. Save the code as main.py
# 2. Install dependencies: pip install fastapi uvicorn sentence-transformers chromadb langchain_core pydantic
# 3. Run: uvicorn main:app --reload
# 4. Access the API documentation at http://127.0.0.1:8000/docs
# 5. Make a POST request to /diagnose with PatientData.
