from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import chromadb
from chromadb.utils import embedding_functions
import spacy
import time
import os

class PatientSymptoms(BaseModel):
    symptoms: str = Field(..., description="A detailed description of the patient's symptoms.")
    patient_id: Optional[str] = Field(None, description="Optional unique identifier for the patient.")

class DiagnosisSuggestion(BaseModel):
    diagnosis: str = Field(..., description="The suggested diagnosis from the LLM.")
    confidence_score: float = Field(..., description="Confidence score for the diagnosis, between 0 and 1.")
    potential_causes: List[str] = Field(..., description="List of potential causes for the symptoms.")
    recommended_tests: List[str] = Field(..., description="List of recommended diagnostic tests.")

class ClinicalTrialInfo(BaseModel):
    trial_id: str = Field(..., description="Unique identifier for the clinical trial.")
    title: str = Field(..., description="Title of the clinical trial.")
    condition: str = Field(..., description="Medical condition targeted by the trial.")
    phase: str = Field(..., description="Phase of the clinical trial (e.g., Phase 1, 2, 3).")
    status: str = Field(..., description="Current recruitment status of the trial.")
    interventions: List[str] = Field(..., description="List of interventions/drugs being tested.")
    location: Optional[str] = Field(None, description="Geographic location of the trial site.")
    url: Optional[str] = Field(None, description="URL to more detailed trial information.")

class ProcessedMedicalLiterature(BaseModel):
    text_summary: str = Field(..., description="A summary of the processed medical literature.")
    extracted_entities: List[str] = Field(..., description="Key medical entities extracted from the text.")
    evidence_chains: List[str] = Field(..., description="Simplified representation of evidence chains found.")

class KGQueryResult(BaseModel):
    query: str = Field(..., description="The query made to the Knowledge Graph.")
    results: List[str] = Field(..., description="List of results from the KG query.")

class WebSearchResponse(BaseModel):
    query: str = Field(..., description="The query used for the web search.")
    snippets: List[str] = Field(..., description="List of relevant snippets from the web search results.")
    urls: List[str] = Field(..., description="List of URLs found during the search.")

class LLMService:
    def __init__(self):
        print("LLMService initialized. (Using a placeholder LLM)")

    def get_diagnostic_suggestion(self, patient_symptoms: PatientSymptoms) -> DiagnosisSuggestion:
        print(f"\n[LLMService] Generating diagnosis for symptoms: {patient_symptoms.symptoms}")
        
        if "fever" in patient_symptoms.symptoms.lower() and "cough" in patient_symptoms.symptoms.lower():
            diagnosis = "Viral Infection"
            confidence = 0.85
            causes = ["Influenza virus", "Common cold virus"]
            tests = ["PCR test for influenza", "Complete Blood Count"]
        elif "chest pain" in patient_symptoms.symptoms.lower() and "shortness of breath" in patient_symptoms.symptoms.lower():
            diagnosis = "Possible Cardiac Event"
            confidence = 0.92
            causes = ["Myocardial Infarction", "Angina"]
            tests = ["ECG", "Troponin levels", "Chest X-ray"]
        elif "type 2 diabetes" in patient_symptoms.symptoms.lower():
            diagnosis = "Type 2 Diabetes Management"
            confidence = 0.95
            causes = ["Insulin resistance", "Pancreatic beta cell dysfunction"]
            tests = ["HbA1c", "Fasting Glucose", "Oral Glucose Tolerance Test"]
        else:
            diagnosis = "Unspecified Condition"
            confidence = 0.60
            causes = ["Further investigation needed"]
            tests = ["Comprehensive blood panel", "Physical examination"]

        return DiagnosisSuggestion(
            diagnosis=diagnosis,
            confidence_score=confidence,
            potential_causes=causes,
            recommended_tests=tests
        )

class KnowledgeRetrieval:
    def __init__(self):
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name="clinical_trials",
            embedding_function=self.embedding_function
        )
        print("KnowledgeRetrieval initialized with ChromaDB and SentenceTransformer.")
        self.seed_database()

    def seed_database(self):
        if self.collection.count() == 0:
            print("[KnowledgeRetrieval] Seeding database with dummy data.")
            dummy_trials = [
                ClinicalTrialInfo(
                    trial_id="NCT01234567",
                    title="A Study of New Drug X for Type 2 Diabetes",
                    condition="Type 2 Diabetes",
                    phase="Phase 3",
                    status="Recruiting",
                    interventions=["Drug X", "Placebo"],
                    location="Multiple sites",
                    url="https://clinicaltrials.gov/ct2/show/NCT01234567"
                ),
                ClinicalTrialInfo(
                    trial_id="NCT07654321",
                    title="Immunotherapy Y for Advanced Melanoma",
                    condition="Melanoma",
                    phase="Phase 2",
                    status="Active, not recruiting",
                    interventions=["Immunotherapy Y"],
                    location="Cancer Research Center",
                    url="https://clinicaltrials.gov/ct2/show/NCT07654321"
                ),
                ClinicalTrialInfo(
                    trial_id="NCT08901234",
                    title="Vaccine Z Efficacy in Preventing Seasonal Flu",
                    condition="Influenza",
                    phase="Phase 3",
                    status="Completed",
                    interventions=["Vaccine Z", "Saline"],
                    location="Global",
                    url="https://clinicaltrials.gov/ct2/show/NCT08901234"
                ),
                 ClinicalTrialInfo(
                    trial_id="NCT01122334",
                    title="Study on Remdesivir for COVID-19 Treatment",
                    condition="COVID-19",
                    phase="Phase 3",
                    status="Completed",
                    interventions=["Remdesivir", "Placebo"],
                    location="Various Hospitals",
                    url="https://clinicaltrials.gov/ct2/show/NCT01122334"
                ),
                ClinicalTrialInfo(
                    trial_id="NCT05544332",
                    title="AI-Assisted Diagnostics for Early Cancer Detection",
                    condition="Early Cancer Detection",
                    phase="Phase 1",
                    status="Recruiting",
                    interventions=["AI Diagnostic Tool", "Standard Diagnostic"],
                    location="University Medical Center",
                    url="https://clinicaltrials.gov/ct2/show/NCT05544332"
                )
            ]
            
            self.collection.add(
                documents=[t.json() for t in dummy_trials],
                metadatas=[t.dict() for t in dummy_trials],
                ids=[t.trial_id for t in dummy_trials]
            )
            print(f"[KnowledgeRetrieval] Added {len(dummy_trials)} dummy clinical trials.")

    def retrieve_relevant_info(self, query: str, n_results: int = 3) -> List[ClinicalTrialInfo]:
        print(f"\n[KnowledgeRetrieval] Retrieving relevant clinical trials for query: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        
        relevant_trials = []
        if results['documents'] and results['metadatas']:
            for i in range(len(results['documents'][0])):
                trial_data = results['metadatas'][0][i]
                relevant_trials.append(ClinicalTrialInfo(**trial_data))
                
        print(f"[KnowledgeRetrieval] Found {len(relevant_trials)} relevant trials.")
        return relevant_trials

class KnowledgeConsolidationPipeline:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading en_core_web_sm model for SpaCy...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        print("KnowledgeConsolidationPipeline initialized with SpaCy.")

    def _extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "EVENT", "NORP", "GPE"] or len(ent.text.split()) > 1]
        medical_keywords = ["disease", "treatment", "therapy", "drug", "vaccine", "diagnosis", "symptom", "clinical trial"]
        for token in doc:
            if token.is_alpha and token.text.lower() in medical_keywords and token.text not in entities:
                entities.append(token.text)
        return list(set(entities))

    def _simulate_evidence_chaining(self, text: str, entities: List[str]) -> List[str]:
        chains = []
        if "treatment" in text.lower() and "disease" in text.lower():
            chains.append(f"Relationship: Treatment mentioned for a disease in the text.")
        if "drug" in text.lower() and "trial" in text.lower():
            chains.append(f"Relationship: Drug being evaluated in a trial.")
        if entities:
            chains.append(f"Entities found: {', '.join(entities[:3])} and others co-occur in text.")
        
        if not chains:
            chains.append("No explicit evidence chains identified in this simplified simulation.")
        return chains

    def process_medical_literature(self, literature_text: str) -> ProcessedMedicalLiterature:
        print("\n[KnowledgeConsolidationPipeline] Processing medical literature...")
        
        extracted_entities = self._extract_entities(literature_text)
        evidence_chains = self._simulate_evidence_chaining(literature_text, extracted_entities)
        
        summary = literature_text[:200] + "... (truncated)" if len(literature_text) > 200 else literature_text
        
        print(f"[KnowledgeConsolidationPipeline] Extracted entities: {extracted_entities[:5]}...")
        print(f"[KnowledgeConsolidationPipeline] Evidence chains: {evidence_chains}")

        return ProcessedMedicalLiterature(
            text_summary=summary,
            extracted_entities=extracted_entities,
            evidence_chains=evidence_chains
        )

class KGIntegration:
    def __init__(self):
        self.medical_kg = self._load_dummy_medical_kg()
        print("KGIntegration initialized with a dummy Medical Knowledge Graph.")

    def _load_dummy_medical_kg(self) -> Dict[str, Dict[str, List[str]]]:
        kg = {
            "Influenza": {
                "has_symptom": ["fever", "cough", "sore throat"],
                "has_treatment": ["antiviral drugs", "rest", "hydration"],
                "is_type_of": ["Viral Infection"]
            },
            "COVID-19": {
                "has_symptom": ["fever", "cough", "shortness of breath", "loss of taste"],
                "has_treatment": ["Remdesivir", "Paxlovid", "supportive care"],
                "is_type_of": ["Viral Infection"],
                "related_virus": ["SARS-CoV-2"]
            },
            "Type 2 Diabetes": {
                "has_symptom": ["frequent urination", "increased thirst", "fatigue"],
                "has_treatment": ["Metformin", "insulin", "dietary changes", "GLP-1 agonists"],
                "is_chronic_condition": ["True"]
            },
            "Melanoma": {
                "has_symptom": ["new or changing mole"],
                "has_treatment": ["surgery", "immunotherapy", "radiation"],
                "is_type_of": ["Cancer"]
            },
            "Immunotherapy": {
                "treats": ["Melanoma", "Lung Cancer"],
                "has_side_effect": ["fatigue", "skin rash"]
            }
        }
        return kg

    def relation_based_reasoning(self, entity: str, relation: str = "all") -> KGQueryResult:
        print(f"\n[KGIntegration] Performing relation-based reasoning for '{entity}' with relation '{relation}'.")
        results = []
        entity_lower = entity.lower()

        found_entity_data = None
        for kg_entity, data in self.medical_kg.items():
            if kg_entity.lower() == entity_lower:
                found_entity_data = data
                break
        
        if found_entity_data:
            if relation == "all":
                for rel, related_entities in found_entity_data.items():
                    results.extend([f"{entity} {rel.replace('_', ' ')} {re}" for re in related_entities])
            elif relation in found_entity_data:
                results.extend([f"{entity} {relation.replace('_', ' ')} {re}" for re in found_entity_data[relation]])
            else:
                results.append(f"No '{relation}' relation found for '{entity}'.")
        else:
            for kg_entity, data in self.medical_kg.items():
                for rel, related_list in data.items():
                    if entity_lower in [item.lower() for item in related_list]:
                        results.append(f"{entity} is related to {kg_entity} via {rel.replace('_', ' ')}.")

            if not results:
                results.append(f"'{entity}' not found directly or in related entities in the KG.")

        return KGQueryResult(query=f"Reasoning on {entity} (relation: {relation})", results=results)

class WebAgent:
    def __init__(self):
        print("WebAgent initialized. (Simulating controlled live web access)")
        self.approved_medical_sites = [
            "clinicaltrials.gov",
            "pubmed.ncbi.nlm.nih.gov",
            "fda.gov",
            "who.int"
        ]

    def search_medical_journals(self, query: str) -> WebSearchResponse:
        print(f"\n[WebAgent] Performing controlled web search for: '{query}' on approved sites.")
        
        time.sleep(1.5)
        
        snippets = []
        urls = []

        if "COVID-19" in query or "coronavirus" in query:
            snippets.append("Recent studies on COVID-19 vaccine booster efficacy.")
            snippets.append("Updates on Omicron variant treatment protocols from WHO.")
            urls.append("https://www.who.int/health-topics/coronavirus")
            urls.append("https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+vaccine")
        elif "Type 2 Diabetes new treatment" in query or "GLP-1 agonists efficacy" in query:
            snippets.append("Emerging GLP-1 receptor agonists show promise in Type 2 Diabetes management.")
            urls.append("https://pubmed.ncbi.nlm.nih.gov/?term=Type+2+Diabetes+GLP-1")
        elif "cancer clinical trials" in query:
            snippets.append("Latest phase 3 trials for novel immunotherapies in oncology.")
            urls.append("https://clinicaltrials.gov/search?cond=Cancer")
        else:
            snippets.append(f"No specific new information found for '{query}' on approved sites (simulated).")
            urls.append(f"https://www.example.com/search?q={query.replace(' ', '+')}")

        print(f"[WebAgent] Found {len(snippets)} simulated web snippets.")

        return WebSearchResponse(query=query, snippets=snippets, urls=urls)

def main():
    print("\n--- Medical Diagnosis Assistant Initialization ---")
    llm_service = LLMService()
    knowledge_retrieval = KnowledgeRetrieval()
    knowledge_pipeline = KnowledgeConsolidationPipeline()
    kg_integration = KGIntegration()
    web_agent = WebAgent()
    print("--- Initialization Complete ---\n")

    print("\n===== Scenario 1: Fever, Cough, and Shortness of Breath =====")
    patient_input1 = PatientSymptoms(symptoms="Patient presents with a persistent fever, dry cough, and occasional shortness of breath for the past 3 days. No significant medical history.")
    
    diagnosis1: DiagnosisSuggestion = llm_service.get_diagnostic_suggestion(patient_input1)
    print(f"LLM Initial Diagnosis: {diagnosis1.diagnosis} (Confidence: {diagnosis1.confidence_score:.2f})")
    print(f"Potential Causes: {', '.join(diagnosis1.potential_causes)}")
    print(f"Recommended Tests: {', '.join(diagnosis1.recommended_tests)}")

    relevant_trials1: List[ClinicalTrialInfo] = knowledge_retrieval.retrieve_relevant_info(
        query=f"Clinical trials for {diagnosis1.diagnosis} or related conditions like {', '.join(diagnosis1.potential_causes)}"
    )
    if relevant_trials1:
        print("\nRelevant Clinical Trials Found:")
        for trial in relevant_trials1:
            print(f"  - [{trial.trial_id}] {trial.title} (Condition: {trial.condition}, Phase: {trial.phase}, Status: {trial.status})")

    if relevant_trials1:
        sample_literature = f"Abstract: A phase 3 study of {relevant_trials1[0].interventions[0]} for {relevant_trials1[0].condition} showed promising results in reducing symptom severity. Side effects observed were mild and transient. This drug targets viral replication pathways."
        processed_literature1: ProcessedMedicalLiterature = knowledge_pipeline.process_medical_literature(sample_literature)
        print(f"\nProcessed Literature Summary: {processed_literature1.text_summary}")
        print(f"Extracted Entities: {', '.join(processed_literature1.extracted_entities)}")
        print(f"Evidence Chains: {', '.join(processed_literature1.evidence_chains)}")

    kg_results1: KGQueryResult = kg_integration.relation_based_reasoning(entity=diagnosis1.diagnosis, relation="has_treatment")
    print("\nKG Reasoning Results (Treatments for Diagnosis):")
    for res in kg_results1.results:
        print(f"  - {res}")

    web_search_query1 = f"latest guidelines for {diagnosis1.diagnosis} OR new clinical trials for {diagnosis1.diagnosis} in {diagnosis1.potential_causes[0] if diagnosis1.potential_causes else 'general'}"
    web_results1: WebSearchResponse = web_agent.search_medical_journals(web_search_query1)
    print("\nWeb Search Results:")
    for i, snippet in enumerate(web_results1.snippets):
        print(f"  - Snippet: {snippet}")
        if i < len(web_results1.urls):
            print(f"    URL: {web_results1.urls[i]}")

    print("\n===== End Scenario 1 =====\n")

    print("\n===== Scenario 2: Management of Type 2 Diabetes =====")
    patient_input2 = PatientSymptoms(symptoms="Patient is a 55-year-old male with a long history of Type 2 Diabetes, currently on Metformin, experiencing occasional hyperglycemic episodes. Inquiring about newer treatment options.")

    diagnosis2: DiagnosisSuggestion = llm_service.get_diagnostic_suggestion(patient_input2)
    print(f"LLM Initial Diagnosis: {diagnosis2.diagnosis} (Confidence: {diagnosis2.confidence_score:.2f})")

    relevant_trials2: List[ClinicalTrialInfo] = knowledge_retrieval.retrieve_relevant_info(
        query="Type 2 Diabetes new treatment options GLP-1 agonists SGLT2 inhibitors", n_results=2
    )
    if relevant_trials2:
        print("\nRelevant Clinical Trials Found for Type 2 Diabetes:")
        for trial in relevant_trials2:
            print(f"  - [{trial.trial_id}] {trial.title} (Condition: {trial.condition}, Phase: {trial.phase}, Status: {trial.status})")

    kg_results2: KGQueryResult = kg_integration.relation_based_reasoning(entity="Type 2 Diabetes", relation="has_treatment")
    print("\nKG Reasoning Results (Treatments for Type 2 Diabetes):")
    for res in kg_results2.results:
        print(f"  - {res}")
    
    web_search_query2 = "Type 2 Diabetes new treatment guidelines 2024 OR GLP-1 agonists efficacy"
    web_results2: WebSearchResponse = web_agent.search_medical_journals(web_search_query2)
    print("\nWeb Search Results for Type 2 Diabetes:")
    for i, snippet in enumerate(web_results2.snippets):
        print(f"  - Snippet: {snippet}")
        if i < len(web_results2.urls):
            print(f"    URL: {web_results2.urls[i]}")
    print("\n===== End Scenario 2 =====\n")

if __name__ == "__main__":
    main()
