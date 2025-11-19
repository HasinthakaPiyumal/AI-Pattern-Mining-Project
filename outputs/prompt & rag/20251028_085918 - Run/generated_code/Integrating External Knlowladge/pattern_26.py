import networkx as nx
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time

# 1. Pydantic Models
class PatientInfo(BaseModel):
    patient_id: str
    symptoms: List[str]
    medical_history: List[str]
    lab_results: Dict[str, Any]
    medications: List[str]

class DiagnosisOutput(BaseModel):
    diagnosis: str
    treatment_recommendation: List[str]
    supporting_evidence: List[str]
    confidence_score: float

# 2. Mock LLM Interface
class LLMInterface:
    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name

    def predict(self, prompt: str, structured_output_model: Optional[BaseModel] = None) -> Any:
        # Simulate LLM response
        time.sleep(0.1)
        if "diagnose" in prompt.lower() and structured_output_model == DiagnosisOutput:
            return DiagnosisOutput(
                diagnosis="Influenza A",
                treatment_recommendation=["Antiviral medication (Oseltamivir)", "Rest", "Hydration"],
                supporting_evidence=["Fever, cough, body aches", "Positive flu test (mock)"],
                confidence_score=0.95
            )
        elif "summarize" in prompt.lower():
            return f"Mock summary of: {prompt[-50:]}"
        return f"Mock LLM response to: {prompt}"

# 3. Mock External APIs
class ExternalMedicalAPIs:
    def pubmed_search(self, query: str) -> List[str]:
        time.sleep(0.05)
        return [f"Mock PubMed article 1 for {query}", f"Mock PubMed article 2 for {query}"]

    def ehr_lookup(self, patient_id: str) -> Dict[str, Any]:
        time.sleep(0.05)
        return {"patient_id": patient_id, "allergies": ["penicillin"], "past_conditions": ["asthma"]}

    def drug_interaction_check(self, drugs: List[str]) -> List[str]:
        time.sleep(0.05)
        if "Oseltamivir" in drugs and "Aspirin" in drugs:
            return ["Potential mild interaction between Oseltamivir and Aspirin (increased bleeding risk - mock)"]
        return ["No significant interactions found (mock)"]

# 4. Mock Vector Database
class MockVectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = [] # Simplified: store texts, not actual embeddings

    def add_documents(self, texts: List[str]):
        for text in texts:
            self.documents.append(text)
            self.embeddings.append(text) # In a real system, this would be an actual embedding

    def semantic_search(self, query: str, top_k: int = 2) -> List[str]:
        # Simulate semantic search by simple keyword matching for demo
        results = [doc for doc in self.documents if query.lower() in doc.lower()]
        return results[:top_k] if results else self.documents[:top_k]

# 5. Knowledge Base Manager (integrates vector DB and external APIs)
class KnowledgeBaseManager:
    def __init__(self):
        self.vector_db = MockVectorDB()
        self.external_apis = ExternalMedicalAPIs()
        self._load_initial_medical_knowledge()

    def _load_initial_medical_knowledge(self):
        # Simulate loading some initial medical documents
        self.vector_db.add_documents([
            "Influenza clinical guidelines: Symptoms include fever, cough, sore throat, body aches.",
            "Treatment for asthma includes bronchodilators and corticosteroids.",
            "Drug interaction information for common antibiotics.",
            "Latest research on COVID-19 variants and treatments."
        ])

    def retrieve_medical_info(self, query: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
        rag_docs = self.vector_db.semantic_search(query)
        pubmed_results = self.external_apis.pubmed_search(query)
        ehr_data = {} 
        if patient_id: 
            ehr_data = self.external_apis.ehr_lookup(patient_id)

        return {
            "rag_documents": rag_docs,
            "pubmed_articles": pubmed_results,
            "ehr_data": ehr_data
        }

    def check_drug_interactions(self, drugs: List[str]) -> List[str]:
        return self.external_apis.drug_interaction_check(drugs)

# 6. Knowledge Consolidation Pipeline
class KnowledgeConsolidationPipeline:
    def __init__(self):
        # Simplified medical ontology for entity linking
        self.medical_ontology = {
            "fever": "Symptom", "cough": "Symptom", "influenza": "Disease",
            "oseltamivir": "Drug", "asthma": "Disease", "penicillin": "Drug"
        }

    def entity_linking(self, text: str) -> List[Dict[str, str]]:
        linked_entities = []
        for term, etype in self.medical_ontology.items():
            if term in text.lower():
                linked_entities.append({"entity": term, "type": etype})
        return linked_entities

    def evidence_chaining(self, knowledge_elements: List[str]) -> List[str]:
        # Simulate connecting pieces of information
        chained_evidence = []
        if any("fever" in k.lower() for k in knowledge_elements) and any("cough" in k.lower() for k in knowledge_elements):
            chained_evidence.append("Patient exhibits common symptoms for respiratory infection.")
        if any("influenza" in k.lower() for k in knowledge_elements) and any("antiviral" in k.lower() for k in knowledge_elements):
            chained_evidence.append("Antiviral treatment is relevant for influenza.")
        return chained_evidence + knowledge_elements # Return original + chained

    def process(self, raw_knowledge: Dict[str, Any]) -> str:
        all_text = " ".join(
            raw_knowledge.get("rag_documents", []) +
            raw_knowledge.get("pubmed_articles", []) +
            [str(raw_knowledge.get("ehr_data", {}))] # Convert EHR dict to string
        )

        linked_entities = self.entity_linking(all_text)
        entities_str = ", ".join([f"{e['entity']} ({e['type']})" for e in linked_entities])

        evidence = self.evidence_chaining(raw_knowledge.get("rag_documents", []) + raw_knowledge.get("pubmed_articles", []))

        return f"Entities identified: {entities_str}. Consolidated Evidence: {'. '.join(evidence)}. Raw Data: {all_text}"

# 7. Medical Knowledge Graph
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self._build_initial_graph()

    def _build_initial_graph(self):
        # Nodes: diseases, symptoms, treatments, drugs
        self.graph.add_nodes_from(["Influenza", "Fever", "Cough", "Oseltamivir", "Asthma", "Bronchodilators", "Penicillin", "Allergy"])

        # Edges: relationships
        self.graph.add_edge("Influenza", "Fever", type="has_symptom")
        self.graph.add_edge("Influenza", "Cough", type="has_symptom")
        self.graph.add_edge("Influenza", "Oseltamivir", type="treated_by")
        self.graph.add_edge("Asthma", "Bronchodilators", type="treated_by")
        self.graph.add_edge("Penicillin", "Allergy", type="causes")

    def add_facts(self, entity1: str, relation: str, entity2: str):
        self.graph.add_edge(entity1, entity2, type=relation)

    def relation_based_reasoning(self, entity: str, relation_type: Optional[str] = None) -> List[str]:
        results = []
        if entity in self.graph:
            for neighbor, data in self.graph[entity].items():
                if relation_type is None or data.get("type") == relation_type:
                    results.append(f"{entity} {data.get('type', 'relates_to')} {neighbor}")
        return results

# 8. Web Agent Tools (Mocked)
class WebAgentTools:
    def browse_web(self, query: str, controlled_access: bool = True) -> str:
        time.sleep(0.2)
        if controlled_access:
            return f"Simulated controlled web search for '{query}': Found relevant forum discussion on {query} treatment innovations (mock data)."
        return "Unauthorized web access attempt (mock)."

# 9. Main System Orchestrator (simulating LangChain agent flow)
class MedicalDiagnosisSystemAgent:
    def __init__(self):
        self.llm = LLMInterface()
        self.kb_manager = KnowledgeBaseManager()
        self.kp_pipeline = KnowledgeConsolidationPipeline()
        self.medical_kg = MedicalKnowledgeGraph()
        self.web_agent = WebAgentTools()

        # Define 'tools' for the agent to use
        self.tools = {
            "retrieve_medical_info": self.kb_manager.retrieve_medical_info,
            "check_drug_interactions": self.kb_manager.check_drug_interactions,
            "process_knowledge": self.kp_pipeline.process,
            "reason_with_kg": self.medical_kg.relation_based_reasoning,
            "browse_web": self.web_agent.browse_web
        }

    def run_diagnosis(self, patient_info: PatientInfo) -> DiagnosisOutput:
        print(f"\n--- Starting Diagnosis for Patient: {patient_info.patient_id} ---")

        # Step 1: Initial LLM understanding and tool identification
        initial_query = f"Diagnose and recommend treatment for a patient with symptoms: {', '.join(patient_info.symptoms)}, history: {', '.join(patient_info.medical_history)}."
        print(f"Agent thinks: Initial query to LLM: '{initial_query[:70]}...'\n")

        # Step 2: External Knowledge Augmentation & RAG
        print("Agent calls: retrieve_medical_info tool...")
        raw_medical_knowledge = self.tools["retrieve_medical_info"](
            query=" ".join(patient_info.symptoms + patient_info.medical_history),
            patient_id=patient_info.patient_id
        )
        print(f"Retrieved knowledge: {list(raw_medical_knowledge.keys())}")

        # Step 3: Knowledge Consolidation Pipeline
        print("Agent calls: process_knowledge tool...")
        consolidated_context = self.tools["process_knowledge"](raw_medical_knowledge)
        print(f"Consolidated context (first 100 chars): {consolidated_context[:100]}...")

        # Step 4: LLM-KG Integration (example reasoning based on symptoms)
        print("Agent calls: reason_with_kg tool for symptom 'Fever'...")
        kg_insights = self.tools["reason_with_kg"]("Fever", relation_type="has_symptom")
        if "Influenza" not in patient_info.medical_history:
            self.medical_kg.add_facts("Patient", "has_symptom", patient_info.symptoms[0]) # Add a fact dynamically
        print(f"KG Insights for Fever: {kg_insights}")

        # Step 5: Conditional Browser-Assisted Access
        if "rare disease" in " ".join(patient_info.symptoms).lower(): # Simplified condition
            print("Agent decides: Need to browse web for rare disease info...")
            web_results = self.tools["browse_web"](f"latest treatment for {patient_info.symptoms[0]} rare disease")
            consolidated_context += f"\nWeb Search Results: {web_results}"
            print("Web results incorporated.")

        # Step 6: Final LLM Diagnosis and Recommendation Generation
        final_prompt = f"Based on patient information: {patient_info.model_dump_json()}, and the following medical context: {consolidated_context}. Provide a diagnosis and treatment plan. Also consider drug interactions for: {', '.join(patient_info.medications)}."
        
        print("Agent calls LLM for final diagnosis and recommendation...")
        # Simulate drug interaction check within the LLM's thought process or as a separate tool call
        drug_interactions = self.tools["check_drug_interactions"](patient_info.medications)
        final_prompt += f"\nDrug Interaction Check: {', '.join(drug_interactions)}"

        diagnosis_output = self.llm.predict(final_prompt, structured_output_model=DiagnosisOutput)

        print("--- Diagnosis Complete ---")
        return diagnosis_output

# Example Usage
if __name__ == "__main__":
    # Setup a patient case
    patient_case_1 = PatientInfo(
        patient_id="P1001",
        symptoms=["fever", "cough", "body aches"],
        medical_history=["seasonal allergies"],
        lab_results={
            "temperature": "102 F",
            "viral_panel": "Pending"
        },
        medications=["Acetaminophen"]
    )

    patient_case_2 = PatientInfo(
        patient_id="P1002",
        symptoms=["persistent headache", "blurred vision", "unexplained fatigue"],
        medical_history=["hypertension"],
        lab_results={
            "blood_pressure": "140/90",
            "MRI_brain": "Pending"
        },
        medications=["Lisinopril", "Aspirin"]
    )

    # Initialize the system agent
    medical_agent = MedicalDiagnosisSystemAgent()

    # Run diagnosis for patient 1
    diagnosis_1 = medical_agent.run_diagnosis(patient_case_1)
    print("\nPatient 1 Diagnosis Result:")
    print(diagnosis_1.model_dump_json(indent=2))

    # Run diagnosis for patient 2, simulating a more complex case potentially needing web search and drug interaction
    # Note: The mock outputs are simplified. In a real system, the LLM would interpret context much more deeply.
    diagnosis_2 = medical_agent.run_diagnosis(patient_case_2)
    print("\nPatient 2 Diagnosis Result:")
    print(diagnosis_2.model_dump_json(indent=2))

    # Demonstrate KG interaction directly
    print("\n--- Direct KG Interaction Example ---")
    print(medical_agent.medical_kg.relation_based_reasoning("Influenza", "has_symptom"))
    print(medical_agent.medical_kg.relation_based_reasoning("Asthma"))

    # Demonstrate drug interaction check directly
    print("\n--- Direct Drug Interaction Check Example ---")
    print(medical_agent.kb_manager.check_drug_interactions(["Oseltamivir", "Aspirin"]))