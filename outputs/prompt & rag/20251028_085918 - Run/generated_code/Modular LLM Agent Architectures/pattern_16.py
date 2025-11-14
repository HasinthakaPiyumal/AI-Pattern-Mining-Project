import streamlit as st
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration and Environment Setup ---
# In a real application, use python-dotenv for API keys
# For this demonstration, set your OpenAI API key directly or via Streamlit secrets
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# --- 1. Medical Knowledge Retrieval Module (`modules/medical_knowledge.py`) ---
class MedicalKnowledgeRetriever:
    def __init__(self):
        # Dummy medical texts for demonstration. In a real app, this would be a comprehensive database.
        self.medical_texts = [
            "Symptoms of common cold include runny nose, sore throat, cough, congestion, and body aches.",
            "Influenza (Flu) symptoms are similar to a cold but often more severe, including high fever, body aches, extreme fatigue.",
            "Diabetes Mellitus is a metabolic disease that causes high blood sugar. Symptoms include frequent urination, increased thirst, and unexplained weight loss.",
            "Hypertension, or high blood pressure, often has no symptoms. Regular check-ups are essential.",
            "Migraine headaches are characterized by severe throbbing pain or a pulsing sensation, usually on one side of the head.",
            "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult."
        ]
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_texts(self.medical_texts, self.embeddings)

    def retrieve_knowledge(self, query: str, k: int = 2) -> List[str]:
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

# --- 2. Symptom Analysis Module (`modules/symptom_analyzer.py`) ---
class Symptom(BaseModel):
    name: str = Field(description="The name of the symptom.")
    severity: str = Field(description="The severity of the symptom (e.g., 'mild', 'moderate', 'severe').")
    duration: str = Field(description="The duration of the symptom (e.g., '2 days', 'since yesterday').")

class SymptomAnalysisOutput(BaseModel):
    symptoms: List[Symptom] = Field(description="A list of structured symptoms identified from the text.")
    key_findings: str = Field(description="A summary of key findings and potential initial interpretations.")

class SymptomAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
        self.parser = JsonOutputParser(pydantic_object=SymptomAnalysisOutput)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical assistant that extracts and structures symptoms from patient descriptions. Always respond in JSON format according to the SymptomAnalysisOutput schema."),
            ("human", "Analyze the following patient's symptoms: {symptoms_text}\n{format_instructions}")
        ]).partial(format_instructions=self.parser.get_format_instructions())
        self.chain = self.prompt | self.llm | self.parser

    def analyze_symptoms(self, symptoms_text: str) -> SymptomAnalysisOutput:
        try:
            return self.chain.invoke({"symptoms_text": symptoms_text})
        except Exception as e:
            st.error(f"Error analyzing symptoms: {e}")
            return SymptomAnalysisOutput(symptoms=[], key_findings="Could not analyze symptoms.")

# --- 3. Patient History Management Module (`modules/patient_history.py`) ---
class PatientHistoryManager:
    def __init__(self):
        self.patients: Dict[str, Dict[str, Any]] = {}

    def add_patient(self, patient_id: str, history: Dict[str, Any]):
        self.patients[patient_id] = history

    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        return self.patients.get(patient_id, {})

    def update_patient(self, patient_id: str, new_data: Dict[str, Any]):
        if patient_id in self.patients:
            self.patients[patient_id].update(new_data)
        else:
            self.add_patient(patient_id, new_data)

# --- 4. Treatment Protocol Recommendations Module (`modules/treatment_recommender.py`) ---
class TreatmentRecommender:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical recommendation system. Based on the provided symptoms and medical knowledge, suggest potential diagnoses and initial treatment recommendations. Always emphasize that this is for informational purposes and not a substitute for professional medical advice."),
            ("human", "Patient symptoms: {symptoms}\nRelevant medical knowledge: {knowledge}\nBased on this, what are potential diagnoses and initial treatment recommendations?")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def recommend_treatment(self, symptoms: List[Symptom], knowledge: List[str]) -> str:
        symptoms_str = ", ".join([f"{s.name} (severity: {s.severity}, duration: {s.duration})" for s in symptoms])
        knowledge_str = "\n".join(knowledge)
        return self.chain.invoke({"symptoms": symptoms_str, "knowledge": knowledge_str})

# --- 5. Core Orchestrator (`app.py`) ---
st.set_page_config(page_title="AI Diagnostic Assistant", layout="wide")
st.title("🩺 AI Diagnostic Assistant")
st.markdown("This assistant helps in preliminary diagnosis and treatment recommendations based on patient symptoms.")

# Initialize modules
knowledge_retriever = MedicalKnowledgeRetriever()
symptom_analyzer = SymptomAnalyzer()
patient_manager = PatientHistoryManager()
treatment_recommender = TreatmentRecommender()

# --- Patient Input Section ---
st.header("Patient Information")
patient_id_input = st.text_input("Enter Patient ID (e.g., P001)", value="P001")

if patient_id_input not in st.session_state:
    st.session_state[patient_id_input] = {"history": {}}

current_patient_history = st.session_state[patient_id_input]["history"]

# Display/Edit patient history (simplified)
st.subheader(f"Patient History for {patient_id_input}")
if current_patient_history:
    st.json(current_patient_history)
else:
    st.write("No history found for this patient. You can add notes below.")

patient_notes = st.text_area("Add/Update Patient Notes", value=current_patient_history.get("notes", ""), height=100)
if st.button("Save Patient Notes"):
    patient_manager.update_patient(patient_id_input, {"notes": patient_notes})
    st.session_state[patient_id_input]["history"] = patient_manager.get_patient(patient_id_input)
    st.success("Patient notes saved!")

# --- Symptom Input Section ---
st.header("Enter Patient Symptoms")
symptoms_text = st.text_area(
    "Describe the patient's symptoms (e.g., 'I have a mild sore throat for 2 days and a cough.', 'Sudden severe headache and nausea since morning.')",
    height=150
)

if st.button("Analyze Symptoms and Get Recommendations"):
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        st.error("Please set your OPENAI_API_KEY at the top of the script or via Streamlit secrets.")
    elif symptoms_text:
        st.subheader("Analysis Results")
        with st.spinner("Analyzing symptoms..."):
            # 1. Symptom Analysis
            analyzed_symptoms_output = symptom_analyzer.analyze_symptoms(symptoms_text)
            st.write("**Structured Symptoms:**")
            if analyzed_symptoms_output.symptoms:
                for symptom in analyzed_symptoms_output.symptoms:
                    st.write(f"- **{symptom.name.capitalize()}**: Severity: {symptom.severity}, Duration: {symptom.duration}")
            else:
                st.write("No specific symptoms identified.")
            st.write(f"**Key Findings:** {analyzed_symptoms_output.key_findings}")

            # 2. Medical Knowledge Retrieval
            symptom_keywords = " ".join([s.name for s in analyzed_symptoms_output.symptoms])
            retrieval_query = f"{symptom_keywords} {analyzed_symptoms_output.key_findings}"
            st.write("\n**Retrieving Medical Knowledge...**")
            relevant_knowledge = knowledge_retriever.retrieve_knowledge(retrieval_query)
            if relevant_knowledge:
                st.write("**Relevant Medical Knowledge:**")
                for knowledge in relevant_knowledge:
                    st.markdown(f"- {knowledge}")
            else:
                st.write("No specific relevant knowledge found.")

            # 3. Treatment Protocol Recommendations
            st.write("\n**Generating Recommendations...**")
            recommendations = treatment_recommender.recommend_treatment(analyzed_symptoms_output.symptoms, relevant_knowledge)
            st.markdown("**Potential Diagnoses & Treatment Recommendations:**")
            st.write(recommendations)

            st.warning("Disclaimer: This AI assistant provides preliminary information and recommendations for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider for any medical concerns.")
    else:
        st.warning("Please enter some symptoms to get started.")
