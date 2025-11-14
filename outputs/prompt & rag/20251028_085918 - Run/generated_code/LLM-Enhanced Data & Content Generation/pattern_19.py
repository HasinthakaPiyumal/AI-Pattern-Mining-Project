from pydantic import BaseModel
from typing import List, Dict, Optional

# --- 1. Data Models (using pydantic for structured output) ---
class PatientRecord(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    medical_history: List[str]
    allergies: List[str]
    current_medications: List[str]

class MedicalFact(BaseModel):
    id: str
    title: str
    content: str
    keywords: List[str]

class DiagnosisOutput(BaseModel):
    diagnostic_hypotheses: List[str]
    differential_diagnoses: List[str]
    treatment_recommendations: List[str]
    prognosis: str
    confidence_score: float = 0.0 # Simulated confidence

# --- 2. Simulated Knowledge Bases ---
# In a real application, these would be Chroma/FAISS and PostgreSQL/MongoDB
medical_literature_db: List[MedicalFact] = [
    MedicalFact(id="mf001", title="Diabetes Mellitus Type 2", content="Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). It's characterized by insulin resistance or insufficient insulin production.", keywords=["diabetes", "glucose", "insulin resistance"]),
    MedicalFact(id="mf002", title="Hypertension Guidelines", content="Hypertension, or high blood pressure, is a common condition. Lifestyle changes and medication are key for management. Normal blood pressure is typically below 120/80 mmHg.", keywords=["hypertension", "blood pressure", "cardiovascular"]),
    MedicalFact(id="mf003", title="Symptoms of Influenza", content="Influenza (flu) is a contagious respiratory illness caused by flu viruses. Symptoms include fever, cough, sore throat, muscle aches, and fatigue.", keywords=["influenza", "flu", "respiratory", "fever"]),
    MedicalFact(id="mf004", title="Asthma Management", content="Asthma is a chronic lung disease that inflames and narrows the airways. It causes recurring periods of wheezing, chest tightness, shortness of breath, and coughing.", keywords=["asthma", "lung", "wheezing", "inhaler"]),
    MedicalFact(id="mf005", title="Pneumonia Overview", content="Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing.", keywords=["pneumonia", "lung infection", "cough", "fever"]),
    MedicalFact(id="mf006", title="Common Cold vs. Flu", content="While both the common cold and the flu are respiratory illnesses, they are caused by different viruses. The flu is generally worse than the common cold and can lead to serious complications.", keywords=["cold", "flu", "respiratory", "virus"]),
]

patient_records_db: Dict[str, PatientRecord] = {
    "patient_a1b2c3d4": PatientRecord(
        patient_id="patient_a1b2c3d4",
        name="Alice Smith",
        age=45,
        gender="Female",
        medical_history=["Type 2 Diabetes (diagnosed 5 years ago)", "Seasonal Allergies"],
        allergies=["Penicillin"],
        current_medications=["Metformin", "Loratadine"]
    ),
    "patient_e5f6g7h8": PatientRecord(
        patient_id="patient_e5f6g7h8",
        name="Bob Johnson",
        age=62,
        gender="Male",
        medical_history=["Hypertension (diagnosed 10 years ago)", "Mild Asthma"],
        allergies=["N/A"],
        current_medications=["Lisinopril", "Albuterol inhaler"]
    ),
}

# --- 3. Retrieval Module ---
class MedicalFactRetriever:
    def retrieve_facts(self, query: str, top_k: int = 3) -> List[MedicalFact]:
        # Simulate keyword-based retrieval from a vector DB for simplicity
        # In a real system, this would involve embedding the query and searching vector_db
        relevant_facts = []
        query_keywords = query.lower().split()
        for fact in medical_literature_db:
            if any(qk in kw.lower() for qk in query_keywords for kw in fact.keywords) or \
               any(qk in fact.content.lower() for qk in query_keywords):
                relevant_facts.append(fact)
        
        # Simple scoring based on keyword overlap (can be replaced by semantic similarity)
        scored_facts = []
        for fact in relevant_facts:
            score = sum(1 for qk in query_keywords if any(qk in kw.lower() for kw in fact.keywords) or qk in fact.content.lower())
            scored_facts.append((score, fact))
        
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored_facts[:top_k]]

class PatientHistoryRetriever:
    def retrieve_history(self, patient_id: str) -> Optional[PatientRecord]:
        return patient_records_db.get(patient_id)

# --- 4. Unified Retrieval and Reasoning Engine (LLM Simulation) ---
class UnifiedDiagnosticEngine:
    def __init__(self):
        self.fact_retriever = MedicalFactRetriever()
        self.history_retriever = PatientHistoryRetriever()
        # In a real application, an LLM would be initialized here (e.g., via langchain/llama_index)

    def diagnose(self, patient_id: str, symptoms: List[str]) -> DiagnosisOutput:
        print(f"\n--- Starting Diagnosis for Patient: {patient_id} ---")
        
        # 1. Retrieve patient history
        patient_history = self.history_retriever.retrieve_history(patient_id)
        if not patient_history:
            print(f"Error: Patient {patient_id} not found.")
            return DiagnosisOutput(
                diagnostic_hypotheses=["Patient record not found"],
                differential_diagnoses=[],
                treatment_recommendations=[],
                prognosis="Unknown",
                confidence_score=0.0
            )
        print(f"Retrieved Patient History: {patient_history.name}, Age: {patient_history.age}, Med History: {patient_history.medical_history}")

        # 2. Initial medical fact retrieval based on symptoms
        symptoms_query = " ".join(symptoms) + " medical condition symptoms"
        medical_facts = self.fact_retriever.retrieve_facts(symptoms_query, top_k=5)
        print(f"Retrieved Initial Medical Facts ({len(medical_facts)}): {[fact.title for fact in medical_facts]}")

        # 3. Simulate LLM-based unified reasoning
        # This is where a real LLM would take all information and reason.
        # For demonstration, we'll use a rule-based simulation heavily influenced by keywords.
        
        all_input_text = f"Patient ID: {patient_history.patient_id}\n"
        all_input_text += f"Name: {patient_history.name}\n"
        all_input_text += f"Age: {patient_history.age}, Gender: {patient_history.gender}\n"
        all_input_text += f"Symptoms: {', '.join(symptoms)}\n"
        all_input_text += f"Medical History: {', '.join(patient_history.medical_history)}\n"
        all_input_text += f"Allergies: {', '.join(patient_history.allergies)}\n"
        all_input_text += f"Current Medications: {', '.join(patient_history.current_medications)}\n\n"
        all_input_text += "Relevant Medical Literature:\n"
        for i, fact in enumerate(medical_facts):
            all_input_text += f"  {i+1}. {fact.title}: {fact.content[:150]}...\n"

        print("\n--- Simulating LLM Reasoning Process ---")
        # Here, a real LLM would generate the diagnosis and recommendations.
        # We'll simulate this by looking for keywords and combining information.
        
        hypotheses = []
        differentials = []
        treatments = []
        prognosis = "Good with proper management."
        confidence = 0.75

        if "fever" in symptoms_query.lower() and "cough" in symptoms_query.lower() and "fatigue" in symptoms_query.lower():
            hypotheses.append("Influenza (Flu)")
            treatments.append("Rest, hydration, antiviral medication if severe (e.g., Oseltamivir).")
            differentials.append("Common Cold")
            differentials.append("Pneumonia")
            confidence += 0.1

        if "wheezing" in symptoms_query.lower() and "shortness of breath" in symptoms_query.lower() and "asthma" in patient_history.medical_history[0].lower():
            hypotheses.append("Asthma Exacerbation")
            treatments.append("Administer bronchodilator (e.g., Albuterol), consider oral corticosteroids.")
            differentials.append("Bronchitis")
            confidence += 0.15
            if "albuterol inhaler" in [med.lower() for med in patient_history.current_medications]:
                 treatments.append("Ensure proper inhaler technique and adherence.")

        if "high blood pressure" in symptoms_query.lower() or "hypertension" in patient_history.medical_history[0].lower():
            hypotheses.append("Uncontrolled Hypertension")
            treatments.append("Review current medication (e.g., Lisinopril), recommend lifestyle changes (diet, exercise).")
            differentials.append("Essential Hypertension")
            confidence += 0.1

        if "elevated blood sugar" in symptoms_query.lower() or "diabetes" in patient_history.medical_history[0].lower():
            hypotheses.append("Suboptimal Diabetes Management")
            treatments.append("Adjust Metformin dosage, recommend dietary consultation and blood glucose monitoring.")
            differentials.append("Diabetic Ketoacidosis (if severe symptoms)")
            confidence += 0.1

        if not hypotheses:
            hypotheses.append("Further investigation needed. Symptoms are non-specific.")
            treatments.append("Recommend detailed physical examination and lab tests.")
            prognosis = "Requires more data."
            confidence = 0.5
            
        # Refine recommendations based on allergies and current medications
        final_treatments = []
        for treatment in treatments:
            is_safe = True
            for allergy in patient_history.allergies:
                if allergy.lower() in treatment.lower():
                    is_safe = False
                    final_treatments.append(f"WARNING: Avoid {allergy}-related components in {treatment}.")
                    break
            # Simple check for drug interaction (highly simplified)
            for current_med in patient_history.current_medications:
                if "lisinopril" in current_med.lower() and "ibuprofen" in treatment.lower(): # Example interaction
                    final_treatments.append(f"CAUTION: Potential interaction between {current_med} and Ibuprofen. Consult pharmacist.")
            if is_safe:
                final_treatments.append(treatment)
        
        print("--- LLM Reasoning Complete ---")

        return DiagnosisOutput(
            diagnostic_hypotheses=list(set(hypotheses)), # Remove duplicates
            differential_diagnoses=list(set(differentials)),
            treatment_recommendations=list(set(final_treatments)),
            prognosis=prognosis,
            confidence_score=min(1.0, confidence) # Cap confidence at 1.0
        )

# --- Main Demonstration Logic ---
if __name__ == "__main__":
    diagnostic_engine = UnifiedDiagnosticEngine()

    # Scenario 1: Alice with flu-like symptoms and existing diabetes
    print("\n=======================================================")
    print("SCENARIO 1: Alice Smith with flu-like symptoms and diabetes")
    print("=======================================================")
    alice_symptoms = ["fever", "cough", "fatigue", "sore throat", "muscle aches"]
    alice_diagnosis = diagnostic_engine.diagnose("patient_a1b2c3d4", alice_symptoms)
    print("\nFinal Diagnosis for Alice:")
    print(alice_diagnosis.model_dump_json(indent=2))

    # Scenario 2: Bob with worsening asthma and hypertension
    print("\n=======================================================")
    print("SCENARIO 2: Bob Johnson with worsening asthma and hypertension")
    print("=======================================================")
    bob_symptoms = ["wheezing", "shortness of breath", "chest tightness", "high blood pressure reading"]
    bob_diagnosis = diagnostic_engine.diagnose("patient_e5f6g7h8", bob_symptoms)
    print("\nFinal Diagnosis for Bob:")
    print(bob_diagnosis.model_dump_json(indent=2))

    # Scenario 3: New patient with non-specific symptoms
    print("\n=======================================================")
    print("SCENARIO 3: New Patient with non-specific symptoms")
    print("=======================================================")
    new_patient_id = "patient_x9y0z1w2"
    patient_records_db[new_patient_id] = PatientRecord(
        patient_id=new_patient_id,
        name="Carol White",
        age=30,
        gender="Female",
        medical_history=[],
        allergies=[],
        current_medications=[]
    )
    carol_symptoms = ["headache", "mild fatigue"]
    carol_diagnosis = diagnostic_engine.diagnose(new_patient_id, carol_symptoms)
    print("\nFinal Diagnosis for Carol:")
    print(carol_diagnosis.model_dump_json(indent=2))

    # Scenario 4: Patient not found
    print("\n=======================================================")
    print("SCENARIO 4: Patient not found in records")
    print("=======================================================")
    unknown_patient_symptoms = ["severe abdominal pain"]
    unknown_patient_diagnosis = diagnostic_engine.diagnose("non_existent_id", unknown_patient_symptoms)
    print("\nFinal Diagnosis for Unknown Patient:")
    print(unknown_patient_diagnosis.model_dump_json(indent=2))
