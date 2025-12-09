from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException

class PatientRecord(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    medical_history: List[str]
    current_medications: List[str]
    lab_results: List[str]
    allergies: List[str]

class MedicalArticle(BaseModel):
    article_id: str
    title: str
    authors: List[str]
    publication_date: str
    summary: str
    keywords: List[str]

class ClinicalGuideline(BaseModel):
    guideline_id: str
    title: str
    version: str
    适用范围: str
    recommendations: List[str]
    evidence_level: Optional[str]

class DiagnosisRequest(BaseModel):
    patient_id: str
    symptoms: str

class DiagnosisResponse(BaseModel):
    patient_id: str
    query_symptoms: str
    augmented_diagnosis: str
    relevant_ehr_data: Optional[PatientRecord]
    relevant_research: List[MedicalArticle]
    relevant_guidelines: List[ClinicalGuideline]

class MockEHRDatabase:
    def __init__(self):
        self.patients = {
            "P001": PatientRecord(
                patient_id="P001",
                name="Alice Smith",
                age=45,
                gender="Female",
                medical_history=["Hypertension", "Type 2 Diabetes"],
                current_medications=["Lisinopril", "Metformin"],
                lab_results=["A1c: 7.2%", "BP: 140/90"],
                allergies=["Penicillin"]
            ),
            "P002": PatientRecord(
                patient_id="P002",
                name="Bob Johnson",
                age=60,
                gender="Male",
                medical_history=["Coronary Artery Disease", "Asthma"],
                current_medications=["Aspirin", "Albuterol"],
                lab_results=["Cholesterol: High", "FEV1: 70%"],
                allergies=[]
            )
        }

    def get_patient_record(self, patient_id: str) -> Optional[PatientRecord]:
        return self.patients.get(patient_id)

class MockPubMedDatabase:
    def __init__(self):
        self.articles = [
            MedicalArticle(
                article_id="MA001",
                title="Recent Advances in Hypertension Management",
                authors=["Dr. J. Doe", "Dr. K. Lee"],
                publication_date="2023-03-15",
                summary="Overview of new therapeutic strategies for hypertension.",
                keywords=["Hypertension", "Blood Pressure", "Treatment"]
            ),
            MedicalArticle(
                article_id="MA002",
                title="Metformin Efficacy in Type 2 Diabetes",
                authors=["Dr. A. Chen"],
                publication_date="2022-11-01",
                summary="A meta-analysis on metformin's long-term effects.",
                keywords=["Metformin", "Diabetes", "Efficacy"]
            ),
            MedicalArticle(
                article_id="MA003",
                title="Asthma Exacerbation Triggers",
                authors=["Dr. B. Singh"],
                publication_date="2023-01-20",
                summary="Study on common environmental triggers for asthma attacks.",
                keywords=["Asthma", "Triggers", "Respiratory"]
            )
        ]

    def search_articles(self, keyword: str, limit: int = 2) -> List[MedicalArticle]:
        results = [art for art in self.articles if keyword.lower() in art.title.lower() or keyword.lower() in ' '.join(art.keywords).lower()]
        return results[:limit]

class MockClinicalGuidelinesDB:
    def __init__(self):
        self.guidelines = [
            ClinicalGuideline(
                guideline_id="CG001",
                title="Hypertension Management Guidelines 2023",
                version="1.2",
                适用范围="Adults with essential hypertension",
                recommendations=["Start with ACE inhibitor or ARB", "Lifestyle modifications are crucial", "Target BP < 130/80 mmHg"],
                evidence_level="A"
            ),
            ClinicalGuideline(
                guideline_id="CG002",
                title="Type 2 Diabetes First-Line Treatment",
                version="3.0",
                适用范围="Newly diagnosed Type 2 Diabetes patients",
                recommendations=["Metformin is recommended as first-line therapy", "Consider SGLT2i or GLP-1 RA for specific indications"],
                evidence_level="B"
            ),
            ClinicalGuideline(
                guideline_id="CG003",
                title="Asthma Exacerbation Protocol",
                version="2.1",
                适用范围="Patients presenting with acute asthma exacerbation",
                recommendations=["Administer short-acting beta-agonists", "Consider oral corticosteroids", "Monitor oxygen saturation"],
                evidence_level="A"
            )
        ]

    def search_guidelines(self, keyword: str, limit: int = 1) -> List[ClinicalGuideline]:
        results = [gl for gl in self.guidelines if keyword.lower() in gl.title.lower() or keyword.lower() in gl.适用范围.lower() or any(keyword.lower() in rec.lower() for rec in gl.recommendations)]
        return results[:limit]

class KnowledgeAugmenter:
    def __init__(self, ehr_db: MockEHRDatabase, pubmed_db: MockPubMedDatabase, guidelines_db: MockClinicalGuidelinesDB):
        self.ehr_db = ehr_db
        self.pubmed_db = pubmed_db
        self.guidelines_db = guidelines_db

    def get_augmented_context(self, patient_id: str, query_symptoms: str) -> dict:
        context_data = {
            "relevant_ehr_data": None,
            "relevant_research": [],
            "relevant_guidelines": []
        }

        patient_record = self.ehr_db.get_patient_record(patient_id)
        if patient_record:
            context_data["relevant_ehr_data"] = patient_record

            search_terms = set([query_symptoms])
            search_terms.update(patient_record.medical_history)
            search_terms.update(patient_record.current_medications)

            for term in list(search_terms):
                articles = self.pubmed_db.search_articles(term)
                context_data["relevant_research"].extend(articles)

                guidelines = self.guidelines_db.search_guidelines(term)
                context_data["relevant_guidelines"].extend(guidelines)

            context_data["relevant_research"] = list({art.article_id: art for art in context_data["relevant_research"]}.values())
            context_data["relevant_guidelines"] = list({gl.guideline_id: gl for gl in context_data["relevant_guidelines"]}.values())

        return context_data

class LLMService:
    def __init__(self, augmenter: KnowledgeAugmenter):
        self.augmenter = augmenter

    def _format_context_for_llm(self, context_data: Dict[str, Any]) -> str:
        context_str_parts = []

        ehr_data: Optional[PatientRecord] = context_data.get("relevant_ehr_data")
        if ehr_data:
            context_str_parts.append(f"--- Patient Electronic Health Record (EHR) ---")
            context_str_parts.append(f"Patient ID: {ehr_data.patient_id}")
            context_str_parts.append(f"Name: {ehr_data.name}, Age: {ehr_data.age}, Gender: {ehr_data.gender}")
            context_str_parts.append(f"Medical History: {', '.join(ehr_data.medical_history)}")
            context_str_parts.append(f"Current Medications: {', '.join(ehr_data.current_medications)}")
            context_str_parts.append(f"Lab Results: {', '.join(ehr_data.lab_results)}")
            context_str_parts.append(f"Allergies: {', '.join(ehr_data.allergies) if ehr_data.allergies else 'None'}")
            context_str_parts.append(f"-------------------------------------------")

        research_articles: List[MedicalArticle] = context_data.get("relevant_research", [])
        if research_articles:
            context_str_parts.append(f"\n--- Relevant Medical Research Articles ---")
            for i, article in enumerate(research_articles):
                context_str_parts.append(f"Article {i+1}: {article.title} (Published: {article.publication_date})")
                context_str_parts.append(f"Summary: {article.summary}")
            context_str_parts.append(f"-------------------------------------------")

        guidelines: List[ClinicalGuideline] = context_data.get("relevant_guidelines", [])
        if guidelines:
            context_str_parts.append(f"\n--- Relevant Clinical Guidelines ---")
            for i, guideline in enumerate(guidelines):
                context_str_parts.append(f"Guideline {i+1}: {guideline.title} (Version: {guideline.version})")
                context_str_parts.append(f"Scope: {guideline.适用范围}")
                context_str_parts.append(f"Recommendations: {'; '.join(guideline.recommendations)}")
            context_str_parts.append(f"---------------------------------------")

        return "\n".join(context_str_parts)

    def _mock_llm_call(self, prompt: str) -> str:
        print(f"\n--- MOCK LLM PROMPT ---\n{prompt}\n------------------------\n")
        if "Hypertension" in prompt and "P001" in prompt:
            return "Based on patient P001's history of hypertension and current symptoms, consider adjusting Lisinopril dosage or adding a second-line agent as per recent hypertension guidelines. Consult the 2023 Hypertension Management Guidelines (CG001) for detailed recommendations and 'Recent Advances in Hypertension Management' (MA001) for new strategies."
        elif "Diabetes" in prompt and "P001" in prompt:
             return "Given patient P001's Type 2 Diabetes (A1c: 7.2%) and current Metformin, consider evaluating glycemic control and adherence. Refer to Type 2 Diabetes First-Line Treatment guidelines (CG002) and 'Metformin Efficacy in Type 2 Diabetes' (MA002) for further details."
        elif "Asthma" in prompt and "P002" in prompt:
            return "For patient P002 with a history of asthma and current symptoms, an acute asthma exacerbation protocol might be necessary. Administer short-acting beta-agonists. Refer to the Asthma Exacerbation Protocol (CG003) and 'Asthma Exacerbation Triggers' (MA003)."
        elif "P001" in prompt:
            return f"Based on patient P001's EHR and the provided context, the LLM suggests further investigation for {prompt.split('Symptoms: ')[-1].split('\n')[0]}. Patient has history of Hypertension and Type 2 Diabetes."
        elif "P002" in prompt:
            return f"Based on patient P002's EHR and the provided context, the LLM suggests further investigation for {prompt.split('Symptoms: ')[-1].split('\n')[0]}. Patient has history of Coronary Artery Disease and Asthma."
        else:
            return f"Based on the symptoms '{prompt.split('Symptoms: ')[-1].split('\n')[0]}' and available context, further medical evaluation is recommended. No specific patient record found or relevant guidelines immediately available."

    def get_augmented_diagnosis(self, patient_id: str, symptoms: str) -> str:
        context_data = self.augmenter.get_augmented_context(patient_id, symptoms)
        formatted_context = self._format_context_for_llm(context_data)

        prompt = f"You are a Clinical Decision Support AI. Provide a concise medical assessment and potential recommendations based on the following patient information and augmented knowledge. Prioritize factual accuracy and clinical relevance.\n\nPatient Query:\nPatient ID: {patient_id}\nSymptoms: {symptoms}\n\n{formatted_context}\n\nBased on the above, provide a clinical assessment and recommendations, referencing the provided context where appropriate:\n"
        return self._mock_llm_call(prompt)

app = FastAPI(
    title="Clinical Decision Support System API",
    description="An API for a Clinical Decision Support System leveraging LLMs augmented with external knowledge (EHR, medical research, clinical guidelines)."
)

ehr_db = MockEHRDatabase()
pubmed_db = MockPubMedDatabase()
guidelines_db = MockClinicalGuidelinesDB()
knowledge_augmenter = KnowledgeAugmenter(ehr_db, pubmed_db, guidelines_db)

llm_service = LLMService(knowledge_augmenter)

@app.post("/diagnose", response_model=DiagnosisResponse)
async def get_augmented_diagnosis_endpoint(request: DiagnosisRequest):
    patient_record = ehr_db.get_patient_record(request.patient_id)
    if not patient_record:
        raise HTTPException(status_code=404, detail=f"Patient with ID {request.patient_id} not found in EHR.")

    augmented_response_text = llm_service.get_augmented_diagnosis(request.patient_id, request.symptoms)

    context_data = knowledge_augmenter.get_augmented_context(request.patient_id, request.symptoms)

    return DiagnosisResponse(
        patient_id=request.patient_id,
        query_symptoms=request.symptoms,
        augmented_diagnosis=augmented_response_text,
        relevant_ehr_data=context_data.get("relevant_ehr_data"),
        relevant_research=context_data.get("relevant_research", []),
        relevant_guidelines=context_data.get("relevant_guidelines", [])
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Clinical Decision Support System is running."}