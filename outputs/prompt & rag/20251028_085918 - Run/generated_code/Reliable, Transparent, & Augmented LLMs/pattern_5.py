
import random
from typing import List, Dict, Optional
from pydantic import BaseModel

# 1. Data Models using Pydantic
class PatientDemographics(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str

class MedicalHistory(BaseModel):
    conditions: List[str]
    medications: List[str]
    allergies: List[str]

class LabResult(BaseModel):
    test_name: str
    value: str
    unit: str
    reference_range: str

class DiagnosticImagingResult(BaseModel):
    scan_type: str
    findings: str
    impression: str

class DrugInteraction(BaseModel):
    drug1: str
    drug2: str
    interaction_type: str
    severity: str
    recommendation: str

class ClinicalRecommendation(BaseModel):
    recommendation_type: str
    details: str
    evidence_level: str

class MedAgentOutput(BaseModel):
    query: str
    reasoning_path: List[str]
    confidence_score: float
    recommendations: List[ClinicalRecommendation]
    raw_tool_outputs: Dict[str, any]
    abstention: bool = False
    abstention_reason: Optional[str] = None


# 2. Mock External Tools
class MockEHRSystem:
    def get_patient_data(self, patient_id: str) -> Optional[Dict]:
        print(f"[MockEHRSystem] Fetching data for patient: {patient_id}")
        if patient_id == "P001":
            return {
                "demographics": PatientDemographics(patient_id="P001", name="Alice Smith", age=45, gender="Female"),
                "history": MedicalHistory(conditions=["Hypertension", "Type 2 Diabetes"], medications=["Lisinopril", "Metformin"], allergies=["Penicillin"]),
                "lab_results": [
                    LabResult(test_name="HbA1c", value="7.2", unit="%", reference_range="<6.5"),
                    LabResult(test_name="Creatinine", value="0.9", unit="mg/dL", reference_range="0.6-1.2")
                ]
            }
        return None

class MockMedicalKnowledgeBase:
    def search_articles(self, query: str) -> List[str]:
        print(f"[MockMedicalKnowledgeBase] Searching for: {query}")
        if "hypertension management" in query.lower():
            return [
                "Guideline: JNC 8 Hypertension Management",
                "Article: Latest advances in blood pressure control"
            ]
        return [f"Article: {query} overview"]

class MockDiagnosticImagingAnalysisTool:
    def analyze_scan(self, image_id: str, patient_id: str) -> Optional[DiagnosticImagingResult]:
        print(f"[MockDiagnosticImagingAnalysisTool] Analyzing image {image_id} for patient {patient_id}")
        if image_id == "IMG001":
            return DiagnosticImagingResult(scan_type="Chest X-ray", findings="Mild cardiomegaly", impression="Consistent with mild cardiac enlargement.")
        return None

class MockDrugInteractionChecker:
    def check_interactions(self, drugs: List[str]) -> List[DrugInteraction]:
        print(f"[MockDrugInteractionChecker] Checking interactions for: {', '.join(drugs)}")
        interactions = []
        if "Lisinopril" in drugs and "Metformin" in drugs:
            interactions.append(DrugInteraction(drug1="Lisinopril", drug2="Metformin", interaction_type="No significant interaction", severity="Low", recommendation="Monitor as usual."))
        if "Warfarin" in drugs and "Aspirin" in drugs:
            interactions.append(DrugInteraction(drug1="Warfarin", drug2="Aspirin", interaction_type="Increased bleeding risk", severity="High", recommendation="Avoid concomitant use or monitor closely."))
        return interactions

class MockClinicalDecisionSupportSystem:
    def get_recommendations(self, symptoms: List[str], patient_data: Dict) -> List[ClinicalRecommendation]:
        print(f"[MockCDSS] Generating recommendations for symptoms: {', '.join(symptoms)} and patient data.")
        recommendations = []
        if "high blood pressure" in [s.lower() for s in symptoms] and patient_data.get("demographics"):
            recommendations.append(ClinicalRecommendation(recommendation_type="Lifestyle Modification", details="Advise dietary changes and increased physical activity.", evidence_level="Strong"))
            recommendations.append(ClinicalRecommendation(recommendation_type="Medication Adjustment", details="Consider adjusting Lisinopril dosage based on blood pressure readings.", evidence_level="Moderate"))
        return recommendations


# 3. Core LLM Agent and Orchestration
class MedAgent:
    def __init__(self):
        self.ehr = MockEHRSystem()
        self.kb = MockMedicalKnowledgeBase()
        self.imaging_tool = MockDiagnosticImagingAnalysisTool()
        self.drug_checker = MockDrugInteractionChecker()
        self.cdss = MockClinicalDecisionSupportSystem()
        self.reasoning_path = []
        self.confidence_score = 0.0
        self.raw_tool_outputs = {}

    def _simulate_llm_reasoning(self, query: str, context: Dict) -> Dict:
        # This is a highly simplified simulation of LLM reasoning.
        # In a real system, this would involve complex prompt engineering and LLM calls.
        self.reasoning_path = []
        self.confidence_score = random.uniform(0.6, 0.95) # Simulate varying confidence

        response_details = {}
        if "patient data" in query.lower():
            self.reasoning_path.append("Retrieved patient demographics and medical history from EHR.")
            response_details["patient_data_summary"] = context.get("ehr_data", {})
        
        if "treatment plan" in query.lower() or "recommendation" in query.lower():
            self.reasoning_path.append("Consulted Clinical Decision Support System for recommendations.")
            self.reasoning_path.append("Cross-referenced with medical knowledge base.")
            response_details["cdss_recommendations"] = context.get("cdss_recommendations", [])
            response_details["kb_articles"] = context.get("kb_articles", [])
        
        if "drug interaction" in query.lower():
            self.reasoning_path.append("Checked for drug-drug interactions.")
            response_details["drug_interactions"] = context.get("drug_interactions", [])

        if self.confidence_score < 0.7: # Simulate abstention for low confidence
            return {"abstain": True, "reason": "Insufficient information or low confidence in generating a definitive recommendation."}

        return {"abstain": False, "response_details": response_details}

    def process_query(self, query: str, patient_id: Optional[str] = None, symptoms: Optional[List[str]] = None, drugs_in_use: Optional[List[str]] = None, image_id: Optional[str] = None) -> MedAgentOutput:
        self.reasoning_path = []
        self.confidence_score = 0.0
        self.raw_tool_outputs = {}
        
        context = {"query": query}
        
        # Tool calls based on query context
        if patient_id:
            ehr_data = self.ehr.get_patient_data(patient_id)
            self.raw_tool_outputs["ehr_data"] = ehr_data
            if ehr_data:
                context["ehr_data"] = ehr_data
                context["patient_medications"] = ehr_data.get("history", {}).get("medications", [])
                context["patient_conditions"] = ehr_data.get("history", {}).get("conditions", [])
        
        if symptoms and context.get("ehr_data"):
            cdss_recommendations = self.cdss.get_recommendations(symptoms, context["ehr_data"])
            self.raw_tool_outputs["cdss_recommendations"] = cdss_recommendations
            context["cdss_recommendations"] = cdss_recommendations

        if drugs_in_use or context.get("patient_medications"):
            all_drugs = list(set((drugs_in_use or []) + (context.get("patient_medications", []))))
            if all_drugs:
                drug_interactions = self.drug_checker.check_interactions(all_drugs)
                self.raw_tool_outputs["drug_interactions"] = drug_interactions
                context["drug_interactions"] = drug_interactions

        if image_id and patient_id:
            imaging_result = self.imaging_tool.analyze_scan(image_id, patient_id)
            self.raw_tool_outputs["imaging_result"] = imaging_result
            if imaging_result:
                context["imaging_result"] = imaging_result

        # Simulate LLM reasoning and decision making
        llm_response = self._simulate_llm_reasoning(query, context)
        
        if llm_response["abstain"]:
            return MedAgentOutput(
                query=query,
                reasoning_path=self.reasoning_path,
                confidence_score=self.confidence_score,
                recommendations=[],
                raw_tool_outputs=self.raw_tool_outputs,
                abstention=True,
                abstention_reason=llm_response["reason"]
            )

        # Synthesize output
        recommendations = llm_response["response_details"].get("cdss_recommendations", [])
        
        final_reasoning_path = self.reasoning_path + ["Synthesized information from various tools to form a comprehensive response."]

        return MedAgentOutput(
            query=query,
            reasoning_path=final_reasoning_path,
            confidence_score=self.confidence_score,
            recommendations=recommendations,
            raw_tool_outputs=self.raw_tool_outputs
        )

    def human_in_the_loop_feedback(self, original_query: str, proposed_output: MedAgentOutput, feedback: str, corrected_output: Optional[MedAgentOutput] = None):
        print(f"\n--- Human-in-the-Loop Feedback ---")
        print(f"Original Query: {original_query}")
        print(f"Proposed Output Confidence: {proposed_output.confidence_score:.2f}")
        print(f"Feedback: {feedback}")
        if corrected_output:
            print(f"Corrected Output (partial): {corrected_output.recommendations}")
        print(f"[MedAgent] Incorporating feedback for future improvements...")


# Example Usage
if __name__ == "__main__":
    med_agent = MedAgent()

    print("\n--- Scenario 1: Diagnostic Support for a known patient ---")
    query1 = "What are the current recommendations for managing hypertension for patient P001, considering their existing medications?"
    symptoms1 = ["High blood pressure"]
    output1 = med_agent.process_query(query=query1, patient_id="P001", symptoms=symptoms1)
    print(output1.model_dump_json(indent=2))

    # Simulate human feedback
    print("\n--- Simulating Human Feedback for Scenario 1 ---")
    med_agent.human_in_the_loop_feedback(
        original_query=query1,
        proposed_output=output1,
        feedback="The recommendation to consider dosage adjustment is good, but emphasize diet more.",
        corrected_output=MedAgentOutput(
            query=query1,
            reasoning_path=[],
            confidence_score=output1.confidence_score,
            recommendations=[ClinicalRecommendation(recommendation_type="Lifestyle Modification", details="Strongly advise comprehensive dietary changes and increased physical activity.", evidence_level="Strong")],
            raw_tool_outputs={}
        )
    )

    print("\n--- Scenario 2: Drug Interaction Check ---")
    query2 = "Are there any known interactions between Warfarin and Aspirin?"
    drugs2 = ["Warfarin", "Aspirin"]
    output2 = med_agent.process_query(query=query2, drugs_in_use=drugs2)
    print(output2.model_dump_json(indent=2))

    print("\n--- Scenario 3: Query resulting in potential abstention ---")
    # This query might trigger low confidence in the simulated LLM
    query3 = "What is the most cutting-edge, experimental treatment for a rare genetic disorder not in standard databases?"
    output3 = med_agent.process_query(query=query3)
    print(output3.model_dump_json(indent=2))

    print("\n--- Scenario 4: Imaging Analysis and Patient Data ---")
    query4 = "Please interpret the Chest X-ray IMG001 for patient P001 and summarize their overall health status."
    output4 = med_agent.process_query(query=query4, patient_id="P001", image_id="IMG001")
    print(output4.model_dump_json(indent=2))

