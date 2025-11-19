import os
import json
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
import gradio as gr

class MockLLM:
    def invoke(self, prompt: str) -> str:
        if "generate an initial treatment plan" in prompt:
            return json.dumps({
                "medication_plan": [{"drug": "Metformin", "dosage": "500mg BID", "reason": "Standard first-line for Type 2 Diabetes"}],
                "diet_recommendations": "Adopt a balanced, low-carb diet focusing on whole foods.",
                "exercise_recommendations": "Engage in 30 minutes of moderate-intensity aerobic exercise most days of the week.",
                "monitoring_schedule": "Monitor blood glucose daily, HbA1c every 3 months, and renal function annually.",
                "rationale": "Initial plan tailored for Type 2 Diabetes management based on current guidelines."
            })
        elif "revise the treatment plan" in prompt:
            if "Metformin contraindicated due to renal impairment" in prompt:
                return json.dumps({
                    "medication_plan": [{"drug": "Glipizide", "dosage": "5mg QD", "reason": "Alternative to Metformin given renal impairment."}
                    ],
                    "diet_recommendations": "Continue a balanced, low-carb diet. Focus on consistent carbohydrate intake.",
                    "exercise_recommendations": "Maintain 30 minutes of moderate-intensity aerobic exercise daily.",
                    "monitoring_schedule": "Monitor blood glucose daily, HbA1c every 3 months, monitor renal function (eGFR) closely due to new medication, and watch for hypoglycemia symptoms.",
                    "rationale": "Revised plan to replace Metformin with Glipizide due to patient's renal impairment. Emphasizing hypoglycemia awareness."
                })
            elif "Sulfonylurea (e.g., Glipizide) contraindicated due to sulfa allergy" in prompt:
                 return json.dumps({
                    "medication_plan": [{"drug": "Sitagliptin", "dosage": "100mg QD", "reason": "Dipeptidyl peptidase-4 (DPP-4) inhibitor, suitable alternative to sulfonylureas and Metformin."}
                    ],
                    "diet_recommendations": "Adhere to a low-carb, balanced diet.",
                    "exercise_recommendations": "Regular physical activity for overall metabolic health.",
                    "monitoring_schedule": "Daily blood glucose monitoring, regular HbA1c, and annual check-ups.",
                    "rationale": "Revised plan to use Sitagliptin, considering both renal impairment and sulfa allergy, avoiding contraindicated drug classes."
                })
            else:
                return json.dumps({
                    "medication_plan": [{"drug": "Default Drug", "dosage": "Default Dose"}],
                    "diet_recommendations": "Generic revised diet recommendations.",
                    "exercise_recommendations": "Generic revised exercise recommendations.",
                    "monitoring_schedule": "Generic revised monitoring schedule.",
                    "rationale": "Plan revised based on general feedback."
                })
        return json.dumps({"error": "LLM did not understand the prompt."})

class MockSentenceTransformer:
    def encode(self, texts: List[str], convert_to_tensor: bool = False) -> List[List[float]]:
        return [[float(hash(text) % 1000) / 1000] for text in texts]

class MockChromaClient:
    def __init__(self):
        self.collection_data = {}

    def get_or_create_collection(self, name: str):
        if name not in self.collection_data:
            self.collection_data[name] = {"documents": [], "embeddings": [], "metadatas": [], "ids": []}
        return MockCollection(self.collection_data[name])

class MockCollection:
    def __init__(self, data_store):
        self.data_store = data_store
        self.embedding_model = MockSentenceTransformer()

    def add(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        for i, doc in enumerate(documents):
            self.data_store["documents"].append(doc)
            self.data_store["embeddings"].append(self.embedding_model.encode([doc])[0])
            self.data_store["metadatas"].append(metadatas[i] if metadatas else {})
            self.data_store["ids"].append(ids[i] if ids else str(len(self.data_store["ids"])))

    def query(self, query_texts: List[str], n_results: int) -> Dict:
        if not self.data_store["documents"]:
            return {"documents": [], "metadatas": [], "ids": []}
        query_embedding = self.embedding_model.encode(query_texts)[0]
        results = {
            "documents": self.data_store["documents"][:n_results],
            "metadatas": self.data_store["metadatas"][:n_results],
            "ids": self.data_store["ids"][:n_results]
        }
        return results

class PatientData(BaseModel):
    patient_id: str
    age: int = Field(..., gt=0, lt=120)
    gender: str
    diagnosis: str
    weight_kg: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    comorbidities: List[str] = Field(default_factory=list)
    lab_results: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)

class TreatmentPlan(BaseModel):
    medication_plan: List[Dict[str, Any]]
    diet_recommendations: str
    exercise_recommendations: str
    monitoring_schedule: str
    rationale: str

class MedicalConstraint(BaseModel):
    constraint_type: str
    condition: str
    feedback: str

class KnowledgeBase:
    def __init__(self):
        self.chroma_client = MockChromaClient()
        self.collection = self.chroma_client.get_or_create_collection("medical_knowledge")
        self._load_mock_data()

    def _load_mock_data(self):
        docs = [
            "Metformin is contraindicated in patients with severe renal impairment (eGFR < 30 mL/min/1.73 m2).",
            "For Type 2 Diabetes, initial treatment often includes Metformin unless contraindicated.",
            "A low-carb diet is beneficial for managing blood glucose in diabetes.",
            "Regular aerobic exercise (e.g., 30 minutes brisk walking daily) improves insulin sensitivity.",
            "Glipizide can be used for Type 2 Diabetes but carries a risk of hypoglycemia.",
            "Patients with sulfa allergies should avoid sulfonylureas like Glipizide.",
            "Hypertension management often involves ACE inhibitors or ARBs as first-line.",
            "Sodium restriction is crucial for hypertension management.",
            "Regular blood pressure monitoring is essential for hypertensive patients."
        ]
        metadatas = [{"source": "guideline"}, {"source": "guideline"}, {"source": "commonsense"}, {"source": "commonsense"},
                     {"source": "guideline"}, {"source": "guideline"}, {"source": "guideline"}, {"source": "commonsense"},
                     {"source": "guideline"}]
        self.collection.add(documents=docs, metadatas=metadatas)

    def retrieve_context(self, query: str, n_results: int = 3) -> List[str]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results.get("documents", [])

class ConstraintEngine:
    def __init__(self):
        self.constraints: List[MedicalConstraint] = [
            MedicalConstraint(constraint_type="medication_contraindication", condition="renal_impairment", feedback="Metformin contraindicated due to renal impairment."),
            MedicalConstraint(constraint_type="allergy_contraindication", condition="sulfa_allergy_sulfonylurea", feedback="Sulfonylurea (e.g., Glipizide) contraindicated due to sulfa allergy.")
        ]

    def validate_plan(self, patient_data: PatientData, plan: TreatmentPlan) -> Optional[str]:
        feedback = []

        if "Metformin" in [m["drug"] for m in plan.medication_plan] and patient_data.lab_results.get("eGFR", 100) < 30:
            feedback.append(next(c.feedback for c in self.constraints if c.condition == "renal_impairment"))

        sulfonylureas = ["Glipizide", "Glimepiride", "Glyburide"]
        for med in plan.medication_plan:
            if med["drug"] in sulfonylureas and "sulfa" in [a.lower() for a in patient_data.allergies]:
                 feedback.append(next(c.feedback for c in self.constraints if c.condition == "sulfa_allergy_sulfonylurea"))

        if feedback:
            return "\n".join(feedback)
        return None

class TreatmentPlanGenerator:
    def __init__(self, llm_model, knowledge_base: KnowledgeBase, constraint_engine: ConstraintEngine):
        self.llm = llm_model
        self.kb = knowledge_base
        self.constraint_engine = constraint_engine

    def generate_plan(self, patient_data: PatientData, max_iterations: int = 3) -> Dict:
        current_plan_json = None
        feedback_history = []

        for i in range(max_iterations):
            prompt = self._construct_prompt(patient_data, current_plan_json, feedback_history, is_initial=(current_plan_json is None))
            llm_response_str = self.llm.invoke(prompt)

            try:
                llm_response_dict = json.loads(llm_response_str)
                current_plan = TreatmentPlan(**llm_response_dict)
                current_plan_json = llm_response_dict
            except (json.JSONDecodeError, ValidationError) as e:
                feedback_history.append(f"LLM output parsing error or invalid format: {e}. Please try again.")
                continue

            validation_feedback = self.constraint_engine.validate_plan(patient_data, current_plan)

            if not validation_feedback:
                return current_plan_json

            feedback_history.append(f"Validation Feedback (Iteration {i+1}): {validation_feedback}")

        if current_plan_json:
            current_plan_json["status"] = "Warning: Plan may still have issues after multiple revisions."
        else:
            current_plan_json = {"status": "Error: Could not generate a valid plan."}
        return current_plan_json

    def _construct_prompt(self, patient_data: PatientData, previous_plan: Optional[Dict], feedback_history: List[str], is_initial: bool) -> str:
        patient_summary = f"""
        Patient ID: {patient_data.patient_id}
        Age: {patient_data.age}
        Gender: {patient_data.gender}
        Diagnosis: {patient_data.diagnosis}
        Weight: {patient_data.weight_kg} kg, Height: {patient_data.height_cm} cm
        Current Medications: {', '.join(patient_data.current_medications) if patient_data.current_medications else 'None'}
        Allergies: {', '.join(patient_data.allergies) if patient_data.allergies else 'None'}
        Comorbidities: {', '.join(patient_data.comorbidities) if patient_data.comorbidities else 'None'}
        Lab Results: {json.dumps(patient_data.lab_results)}
        Preferences: {json.dumps(patient_data.preferences)}
        """

        query = f"Treatment for {patient_data.diagnosis} considering {patient_data.comorbidities} and {patient_data.allergies}"
        context = self.kb.retrieve_context(query)
        context_str = "\n".join(context)

        base_prompt = f"""
        You are an AI assistant helping a healthcare professional generate personalized treatment plans.
        The patient's summary is provided below.
        Consider the following medical knowledge and guidelines:
        {context_str}

        Patient Summary:
        {patient_summary}

        Your task is to generate a comprehensive treatment plan in JSON format.
        The plan must include:
        - "medication_plan": A list of dictionaries, each with "drug" and "dosage".
        - "diet_recommendations": A string describing dietary advice.
        - "exercise_recommendations": A string describing exercise advice.
        - "monitoring_schedule": A string detailing necessary monitoring.
        - "rationale": A brief explanation for the plan.
        """

        if is_initial:
            return f"{base_prompt}\n\nPlease generate an initial treatment plan."
        else:
            feedback_str = "\n".join(feedback_history)
            previous_plan_str = json.dumps(previous_plan, indent=2)
            return f"""
            {base_prompt}

            Previous Plan:
            {previous_plan_str}

            Based on the following feedback, please revise the treatment plan to address the issues:
            {feedback_str}

            Revised Plan:
            """

def run_generator(
    patient_id: str,
    age: int,
    gender: str,
    diagnosis: str,
    weight_kg: float,
    height_cm: float,
    current_medications_str: str,
    allergies_str: str,
    comorbidities_str: str,
    lab_results_json: str,
    preferences_json: str
):
    try:
        current_medications = [m.strip() for m in current_medications_str.split(',') if m.strip()]
        allergies = [a.strip() for a in allergies_str.split(',') if a.strip()]
        comorbidities = [c.strip() for c in comorbidities_str.split(',') if c.strip()]
        lab_results = json.loads(lab_results_json) if lab_results_json.strip() else {}
        preferences = json.loads(preferences_json) if preferences_json.strip() else {}

        patient_data = PatientData(
            patient_id=patient_id,
            age=age,
            gender=gender,
            diagnosis=diagnosis,
            weight_kg=weight_kg,
            height_cm=height_cm,
            current_medications=current_medications,
            allergies=allergies,
            comorbidities=comorbidities,
            lab_results=lab_results,
            preferences=preferences
        )

        llm = MockLLM()
        kb = KnowledgeBase()
        constraint_engine = ConstraintEngine()
        generator = TreatmentPlanGenerator(llm, kb, constraint_engine)

        plan = generator.generate_plan(patient_data)
        return json.dumps(plan, indent=2)

    except ValidationError as e:
        return f"Input validation error: {e}"
    except json.JSONDecodeError as e:
        return f"JSON parsing error in lab results or preferences: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

iface = gr.Interface(
    fn=run_generator,
    inputs=[
        gr.Textbox(label="Patient ID", value="P001"),
        gr.Number(label="Age", value=55),
        gr.Textbox(label="Gender", value="Male"),
        gr.Textbox(label="Diagnosis", value="Type 2 Diabetes"),
        gr.Number(label="Weight (kg)", value=80),
        gr.Number(label="Height (cm)", value=175),
        gr.Textbox(label="Current Medications (comma-separated)", value=""),
        gr.Textbox(label="Allergies (comma-separated)", value=""),
        gr.Textbox(label="Comorbidities (comma-separated)", value=""),
        gr.Textbox(label="Lab Results (JSON)", value='{"eGFR": 90, "HbA1c": 8.2}'),
        gr.Textbox(label="Preferences (JSON)", value='{"dietary_restrictions": ["vegetarian"]}')
    ],
    outputs="json",
    title="AI-Powered Personalized Treatment Plan Generator",
    description="Generate adaptive treatment plans for chronic disease management. The system uses LLMs, a knowledge base, and constraint validation for personalized recommendations."
)

if __name__ == "__main__":
    iface.launch()