from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from fastapi import FastAPI, HTTPException
import uvicorn
import asyncio
from datetime import datetime

# --- Configuration and Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TreatmentPlanGenerator")

# --- Pydantic Data Models ---
class PatientDemographics(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    dob: str
    ethnicity: Optional[str] = None

class Diagnosis(BaseModel):
    condition: str
    icd10_code: Optional[str] = None
    onset_date: Optional[str] = None

class Medication(BaseModel):
    medication_name: str
    dosage: str
    frequency: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class PastTreatment(BaseModel):
    treatment_name: str
    outcome: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class MedicalRecordStructured(BaseModel):
    demographics: PatientDemographics
    diagnoses: List[Diagnosis] = []
    medications: List[Medication] = []
    past_treatments: List[PastTreatment] = []
    allergies: List[str] = []

class AnalyzedLabResult(BaseModel):
    marker: str
    value: float
    unit: str
    is_abnormal: bool
    clinical_significance: str

class MedicationInteractionWarning(BaseModel):
    drug1: str
    drug2: str
    interaction_type: str
    severity: str
    description: str

class RecommendedTreatment(BaseModel):
    medication: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    monitoring_protocol: Optional[str] = None
    lifestyle_change: Optional[str] = None
    justification: str

class LifestyleRecommendation(BaseModel):
    category: str
    description: str
    rationale: str
    examples: List[str] = []

class TreatmentPlanOutput(BaseModel):
    patient_id: str
    generated_date: str
    parsed_medical_records: MedicalRecordStructured
    analyzed_lab_results: List[AnalyzedLabResult]
    medication_interaction_warnings: List[MedicationInteractionWarning]
    recommended_treatments: List[RecommendedTreatment]
    lifestyle_recommendations: List[LifestyleRecommendation]
    overall_summary: str

# --- Tool Implementations (Stubs) ---

class MedicalRecordParser:
    def parse_medical_record(self, raw_clinical_notes: str) -> MedicalRecordStructured:
        logger.info("Parsing medical record...")
        # Simulate NLP extraction based on keywords for demonstration
        patient_id = "PAT" + str(hash(raw_clinical_notes) % 10000)
        name = "Unknown Patient"
        age = 0
        gender = "Unknown"
        dob = "YYYY-MM-DD"

        if "Jane Doe" in raw_clinical_notes:
            name = "Jane Doe"
        if "55 y.o." in raw_clinical_notes:
            age = 55
        if "female" in raw_clinical_notes:
            gender = "Female"

        demographics = PatientDemographics(patient_id=patient_id, name=name, age=age, gender=gender, dob=dob)

        diagnoses = []
        if "Type 2 Diabetes" in raw_clinical_notes:
            diagnoses.append(Diagnosis(condition="Type 2 Diabetes", icd10_code="E11.9", onset_date="2010-03-01"))
        if "hypertension" in raw_clinical_notes:
            diagnoses.append(Diagnosis(condition="Hypertension", icd10_code="I10", onset_date="Unknown"))

        medications = []
        if "Metformin 500mg BID" in raw_clinical_notes:
            medications.append(Medication(medication_name="Metformin", dosage="500mg", frequency="Twice daily"))

        past_treatments = []
        if "dietary counseling" in raw_clinical_notes:
            past_treatments.append(PastTreatment(treatment_name="Dietary counseling", outcome="Improved HbA1c", start_date="2010-04-01"))

        allergies = []
        if "Penicillin" in raw_clinical_notes:
            allergies.append("Penicillin")

        return MedicalRecordStructured(
            demographics=demographics,
            diagnoses=diagnoses,
            medications=medications,
            past_treatments=past_treatments,
            allergies=allergies
        )

class LabResultAnalyzer:
    def analyze_lab_results(self, raw_lab_data: Dict[str, Any]) -> List[AnalyzedLabResult]:
        logger.info("Analyzing lab results...")
        results = []
        # Simulate lab analysis with some dummy rules
        glucose_fasting_val = raw_lab_data.get("glucose_fasting", 0)
        if glucose_fasting_val > 125:
            results.append(AnalyzedLabResult(marker="Fasting Glucose", value=glucose_fasting_val, unit="mg/dL", is_abnormal=True, clinical_significance="High, indicative of diabetes"))
        else:
            results.append(AnalyzedLabResult(marker="Fasting Glucose", value=glucose_fasting_val if glucose_fasting_val else 90, unit="mg/dL", is_abnormal=False, clinical_significance="Normal"))

        hba1c_val = raw_lab_data.get("hba1c", 0)
        if hba1c_val > 6.4:
            results.append(AnalyzedLabResult(marker="HbA1c", value=hba1c_val, unit="%", is_abnormal=True, clinical_significance="High, poor glycemic control"))
        else:
            results.append(AnalyzedLabResult(marker="HbA1c", value=hba1c_val if hba1c_val else 5.8, unit="%", is_abnormal=False, clinical_significance="Controlled"))

        creatinine_val = raw_lab_data.get("creatinine", 0)
        if creatinine_val > 1.2:
            results.append(AnalyzedLabResult(marker="Creatinine", value=creatinine_val, unit="mg/dL", is_abnormal=True, clinical_significance="High, potential kidney impairment"))
        else:
            results.append(AnalyzedLabResult(marker="Creatinine", value=creatinine_val if creatinine_val else 0.9, unit="mg/dL", is_abnormal=False, clinical_significance="Normal"))

        cholesterol_total_val = raw_lab_data.get("cholesterol_total", 0)
        if cholesterol_total_val > 200:
            results.append(AnalyzedLabResult(marker="Total Cholesterol", value=cholesterol_total_val, unit="mg/dL", is_abnormal=True, clinical_significance="High, increased cardiovascular risk"))
        else:
            results.append(AnalyzedLabResult(marker="Total Cholesterol", value=cholesterol_total_val if cholesterol_total_val else 180, unit="mg/dL", is_abnormal=False, clinical_significance="Normal"))

        return results

class MedicationInteractionChecker:
    _INTERACTIONS = {
        ("Metformin", "Iodinated contrast media"): ("Drug-drug", "High", "Increased risk of lactic acidosis"),
        ("Warfarin", "NSAIDs"): ("Drug-drug", "Moderate", "Increased risk of bleeding"),
        ("Metformin", "Kidney Disease"): ("Drug-condition", "High", "Contraindicated in severe renal impairment"),
        ("Metformin", "Creatinine High"): ("Drug-lab", "High", "Increased risk of lactic acidosis with impaired renal function"),
        ("Allergy: Penicillin", "Amoxicillin"): ("Drug-allergy", "High", "Risk of severe allergic reaction"),
    }

    def check_interactions(self, medications: List[Medication], diagnoses: List[Diagnosis], analyzed_lab_results: List[AnalyzedLabResult], allergies: List[str]) -> List[MedicationInteractionWarning]:
        logger.info("Checking medication interactions...")
        warnings = []
        med_names = {m.medication_name for m in medications}
        conditions = {d.condition for d in diagnoses}
        abnormal_lab_flags = {f"{lr.marker} {('High' if lr.value > 0 else 'Low')}" for lr in analyzed_lab_results if lr.is_abnormal}
        patient_allergies = {f"Allergy: {a}" for a in allergies}

        for (item1, item2), (int_type, severity, desc) in self._INTERACTIONS.items():
            if int_type == "Drug-drug":
                if item1 in med_names and item2 in med_names:
                    warnings.append(MedicationInteractionWarning(
                        drug1=item1, drug2=item2,
                        interaction_type=int_type, severity=severity, description=desc
                    ))
            elif int_type == "Drug-condition":
                if item1 in med_names and item2 in conditions:
                    warnings.append(MedicationInteractionWarning(
                        drug1=item1, drug2=item2, # item2 here is a condition
                        interaction_type=int_type, severity=severity, description=desc
                    ))
            elif int_type == "Drug-lab":
                if item1 in med_names and item2 in abnormal_lab_flags:
                    warnings.append(MedicationInteractionWarning(
                        drug1=item1, drug2=item2, # item2 here is a lab flag
                        interaction_type=int_type, severity=severity, description=desc
                    ))
            elif int_type == "Drug-allergy":
                # For drug-allergy, item1 represents the allergy, item2 the drug
                if item1 in patient_allergies and item2 in med_names:
                    warnings.append(MedicationInteractionWarning(
                        drug1=item2, drug2=item1, # Swapping for clarity in output
                        interaction_type=int_type, severity=severity, description=desc
                    ))
        return warnings

class TreatmentGuidelineMatcher:
    def match_guidelines(self, patient_profile: MedicalRecordStructured, lab_results: List[AnalyzedLabResult], interaction_warnings: List[MedicationInteractionWarning]) -> List[RecommendedTreatment]:
        logger.info("Matching treatment guidelines...")
        recommendations = []
        diagnoses = {d.condition for d in patient_profile.diagnoses}
        current_meds = {m.medication_name for m in patient_profile.medications}
        abnormal_labs = {lr.marker for lr in lab_results if lr.is_abnormal}

        # Guideline for Type 2 Diabetes
        if "Type 2 Diabetes" in diagnoses:
            hba1c_value = next((lr.value for lr in lab_results if lr.marker == "HbA1c"), None)
            if hba1c_value is not None and hba1c_value > 7.0:
                if "Metformin" in current_meds:
                    recommendations.append(RecommendedTreatment(
                        medication="Consider adding GLP-1 receptor agonist or SGLT2 inhibitor",
                        justification="HbA1c remains elevated despite Metformin; guideline recommends escalating therapy."
                    ))
                else:
                    recommendations.append(RecommendedTreatment(
                        medication="Metformin", dosage="Start with 500mg daily, titrate up", frequency="Daily/Twice daily",
                        monitoring_protocol="Monitor renal function, GI side effects", justification="First-line therapy for Type 2 Diabetes with elevated HbA1c"
                    ))
            elif "Metformin" not in current_meds and hba1c_value is not None and hba1c_value > 6.4:
                recommendations.append(RecommendedTreatment(
                    medication="Metformin", dosage="500mg-1000mg", frequency="Twice daily",
                    monitoring_protocol="Monitor renal function annually", justification="First-line therapy for Type 2 Diabetes"
                ))

            # Consider lifestyle for all diabetes patients
            recommendations.append(RecommendedTreatment(
                lifestyle_change="Intensive diet and exercise program",
                justification="Cornerstone of Type 2 Diabetes management."
            ))

        # Guideline for Hypertension
        if "Hypertension" in diagnoses:
            # Simplified for demo: if no anti-hypertensive meds, suggest start
            if not any(med in current_meds for med in ["Lisinopril", "Amlodipine", "Hydrochlorothiazide"]):
                recommendations.append(RecommendedTreatment(
                    medication="Initiate ACE inhibitor or ARB",
                    justification="First-line pharmacotherapy for most patients with hypertension."
                ))
            recommendations.append(RecommendedTreatment(
                lifestyle_change="Sodium restriction and regular aerobic exercise",
                justification="Essential non-pharmacological interventions for hypertension."
            ))

        # General warning for high severity interactions
        if any(w.severity == "High" for w in interaction_warnings):
            recommendations.append(RecommendedTreatment(
                lifestyle_change="Urgent review of current medication regimen with a clinician.",
                justification="High severity drug interactions detected that require immediate attention."
            ))

        return recommendations

class LifestyleDietaryRecommender:
    def recommend_lifestyle_changes(self, patient_profile: MedicalRecordStructured, recommended_treatments: List[RecommendedTreatment]) -> List[LifestyleRecommendation]:
        logger.info("Recommending lifestyle and dietary changes...")
        lifestyle_recs = []
        diagnoses = {d.condition for d in patient_profile.diagnoses}

        if "Type 2 Diabetes" in diagnoses:
            lifestyle_recs.append(LifestyleRecommendation(
                category="Diet",
                description="Adopt a balanced diet low in refined sugars and saturated fats, rich in whole grains, lean proteins, and vegetables. Focus on consistent carbohydrate intake.",
                rationale="Essential for blood sugar management and overall cardiovascular health in Type 2 Diabetes. Consistent carbohydrate intake helps prevent glucose spikes.",
                examples=["Mediterranean diet principles", "Portion control using the plate method", "Avoid sugary drinks and highly processed foods"]
            ))
            lifestyle_recs.append(LifestyleRecommendation(
                category="Exercise",
                description="Engage in at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week, plus 2-3 sessions of strength training.",
                rationale="Improves insulin sensitivity, aids weight management, reduces cardiovascular risk, and helps manage blood pressure.",
                examples=["Brisk walking (30 min, 5 times/week)", "Swimming, Cycling", "Weightlifting, Resistance band exercises"]
            ))
            lifestyle_recs.append(LifestyleRecommendation(
                category="Stress Management",
                description="Incorporate stress-reducing practices into daily routine, as stress can impact blood glucose levels and overall well-being.",
                rationale="Chronic stress can elevate cortisol, potentially leading to increased blood sugar. Effective stress management improves quality of life.",
                examples=["Mindfulness meditation", "Yoga", "Deep breathing exercises", "Hobbies and social engagement"]
            ))
            lifestyle_recs.append(LifestyleRecommendation(
                category="Monitoring",
                description="Regularly monitor blood glucose levels as advised by your doctor. Keep a log of your readings.",
                rationale="Self-monitoring is crucial for understanding how diet, exercise, and medication affect your blood sugar.",
                examples=["Use a glucometer as prescribed", "Record readings in a diary or app"]
            ))

        if "Hypertension" in diagnoses:
            lifestyle_recs.append(LifestyleRecommendation(
                category="Diet",
                description="Follow a DASH (Dietary Approaches to Stop Hypertension) eating plan, emphasizing fruits, vegetables, whole grains, and low-fat dairy, while reducing sodium.",
                rationale="The DASH diet is proven to lower blood pressure and reduce cardiovascular risk.",
                examples=["Limit processed foods and salty snacks", "Use herbs and spices instead of salt", "Increase potassium-rich foods like bananas and potatoes"]
            ))
            lifestyle_recs.append(LifestyleRecommendation(
                category="Exercise",
                description="Maintain regular physical activity, aiming for at least 150 minutes of moderate-intensity exercise per week.",
                rationale="Regular exercise strengthens the heart, improves circulation, and helps manage blood pressure.",
                examples=["Brisk walking, jogging, swimming", "Cycling"]
            ))
            lifestyle_recs.append(LifestyleRecommendation(
                category="Weight Management",
                description="If overweight or obese, aim for gradual weight loss, as even a small reduction can significantly impact blood pressure.",
                rationale="Excess weight places additional strain on the heart and blood vessels.",
                examples=["Combine diet and exercise for sustainable weight loss"]
            ))

        # Add a general recommendation if there are high severity warnings, even if specific conditions are not listed
        if any(w.justification == "High severity drug interactions detected that require immediate attention." for w in recommended_treatments):
            lifestyle_recs.append(LifestyleRecommendation(
                category="Urgent Consultation",
                description="Consult with your healthcare provider immediately to review medication regimen and potential risks.",
                rationale="Critical drug interactions have been identified that could pose serious health risks.",
                examples=["Contact your prescribing doctor or pharmacist", "Do not alter medications without medical advice"]
            ))

        return lifestyle_recs

# --- Controller AI Logic ---
class TreatmentPlanController:
    def __init__(self):
        self.parser = MedicalRecordParser()
        self.lab_analyzer = LabResultAnalyzer()
        self.interaction_checker = MedicationInteractionChecker()
        self.guideline_matcher = TreatmentGuidelineMatcher()
        self.lifestyle_recommender = LifestyleDietaryRecommender()

    async def generate_plan(self, raw_patient_data: Dict[str, Any]) -> TreatmentPlanOutput:
        logger.info("Starting treatment plan generation workflow.")

        # Step 1: Parse Medical Records
        raw_clinical_notes = raw_patient_data.get("clinical_notes", "")
        parsed_records = await asyncio.to_thread(self.parser.parse_medical_record, raw_clinical_notes)
        logger.info(f"Medical records parsed for patient: {parsed_records.demographics.patient_id}")

        # Step 2: Analyze Lab Results
        raw_lab_data = raw_patient_data.get("lab_reports", {})
        analyzed_labs = await asyncio.to_thread(self.lab_analyzer.analyze_lab_results, raw_lab_data)
        logger.info("Lab results analyzed.")

        # Step 3: Check Medication Interactions
        medication_interaction_warnings = await asyncio.to_thread(self.interaction_checker.check_interactions,
                                                                 parsed_records.medications, parsed_records.diagnoses,
                                                                 analyzed_labs, parsed_records.allergies)
        logger.info(f"Medication interaction checks completed. Warnings: {len(medication_interaction_warnings)}")

        # Step 4: Match Treatment Guidelines
        recommended_treatments = await asyncio.to_thread(self.guideline_matcher.match_guidelines,
                                                         parsed_records, analyzed_labs, medication_interaction_warnings)
        logger.info(f"Treatment guidelines matched. Recommendations: {len(recommended_treatments)}")

        # Step 5: Recommend Lifestyle & Dietary Changes
        lifestyle_recommendations = await asyncio.to_thread(self.lifestyle_recommender.recommend_lifestyle_changes,
                                                            parsed_records, recommended_treatments)
        logger.info(f"Lifestyle recommendations generated. Recommendations: {len(lifestyle_recommendations)}")

        # Final Output Aggregation
        overall_summary = (
            f"Personalized Treatment Plan for {parsed_records.demographics.name} (ID: {parsed_records.demographics.patient_id}).\n"
            f"Diagnoses: {', '.join([d.condition for d in parsed_records.diagnoses])}.\n"
            f"Abnormal Lab Findings: {', '.join([lr.marker for lr in analyzed_labs if lr.is_abnormal])}.\n"
            f"Medication Interaction Warnings: {len(medication_interaction_warnings)} found.\n"
            f"The plan integrates medical guidelines with personalized lifestyle adjustments and considers all identified health factors."
        )

        return TreatmentPlanOutput(
            patient_id=parsed_records.demographics.patient_id,
            generated_date=datetime.now().isoformat(),
            parsed_medical_records=parsed_records,
            analyzed_lab_results=analyzed_labs,
            medication_interaction_warnings=medication_interaction_warnings,
            recommended_treatments=recommended_treatments,
            lifestyle_recommendations=lifestyle_recommendations,
            overall_summary=overall_summary
        )

# --- FastAPI Application ---
app = FastAPI(
    title="AI-Powered Personalized Treatment Plan Generator",
    description="An API to generate personalized treatment plans for chronic diseases by orchestrating multiple AI tools."
)

controller = TreatmentPlanController()

@app.post("/generate_treatment_plan", response_model=TreatmentPlanOutput)
async def generate_treatment_plan_endpoint(raw_patient_data: Dict[str, Any]):
    """
    Generates a personalized treatment plan based on raw patient data.
    Expects a dictionary with "clinical_notes" (str) and "lab_reports" (dict).
    """
    try:
        plan = await controller.generate_plan(raw_patient_data)
        return plan
    except Exception as e:
        logger.error(f"Error generating treatment plan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    example_patient_data = {
        "clinical_notes": "Patient Jane Doe, 55 y.o. female, diagnosed with Type 2 Diabetes in 2010. Current medications include Metformin 500mg BID. No known allergies except Penicillin. Past treatment: dietary counseling. History of hypertension.",
        "lab_reports": {
            "glucose_fasting": 140.5,
            "hba1c": 7.2,
            "cholesterol_total": 220,
            "creatinine": 1.5
        }
    }
    logger.info("Starting FastAPI application...")
    print("\n\nTo interact with the API, run this script and then visit: http://127.0.0.1:8000/docs")
    print("You can use the provided example_patient_data in the /generate_treatment_plan endpoint.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
