import json
import time
from typing import List, Dict, Any

# --- 1. Mock Implementations for External Services ---

class MockLLM:
    """Simulates a Large Language Model for generating responses."""
    def generate_response(self, prompt: str) -> str:
        print(f"[MockLLM] Generating response for prompt (first 100 chars): {prompt[:100]}...")
        time.sleep(0.5)  # Simulate LLM processing time
        if "glioblastoma" in prompt.lower() and "treatment" in prompt.lower():
            return f"Based on the provided context, for glioblastoma treatment, recent studies suggest a combination of temozolomide and radiation therapy, potentially augmented by novel targeted therapies depending on molecular markers. Patient history of hypertension is noted and would influence treatment choice. Further personalized recommendations require a full patient profile. Context used: {prompt}"
        elif "drug interaction" in prompt.lower():
            return f"Based on the context, potential drug interactions are important to consider. Always cross-reference with a comprehensive drug interaction database for the specific medications. Context used: {prompt}"
        else:
            return f"I am a Mock LLM. I received your query and some augmented context. Here's a generic response: '{prompt}' "

class MockEmbeddingModel:
    """Simulates an embedding model to convert text to vectors."""
    def encode(self, text: str) -> List[float]:
        # Very simplistic mock embedding: sum of ASCII values scaled and padded
        # In a real scenario, this would be a high-dimensional vector
        base_embedding = [float(ord(char)) for char in text[:100]]
        # Pad or truncate to a fixed size for demonstration
        vector_size = 768  # Common embedding size
        if len(base_embedding) < vector_size:
            base_embedding.extend([0.0] * (vector_size - len(base_embedding)))
        return base_embedding[:vector_size]

class MockVectorDatabase:
    """Simulates a vector database for storing and retrieving document embeddings."""
    def __init__(self):
        self.documents: List[Dict[str, Any]] = [] # {'id': str, 'text': str, 'embedding': List[float]}
        self.next_id = 0

    def add_document(self, text: str, embedding: List[float]):
        doc_id = str(self.next_id)
        self.documents.append({"id": doc_id, "text": text, "embedding": embedding})
        self.next_id += 1
        print(f"[MockVectorDB] Added document with ID: {doc_id}")

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Finds the top_k most similar documents using a simple dot product similarity."""
        if not self.documents:
            return []

        similarities = []
        for doc in self.documents:
            # Simple dot product for similarity (cosine similarity would be better in real world)
            dot_product = sum(q * d for q, d in zip(query_embedding, doc["embedding"]))
            # Normalize by magnitudes (skipped for simplicity in mock, assumes normalized vectors if real)
            similarities.append((dot_product, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in similarities[:top_k]]

class ExternalIntegrations:
    """Simulates various external API calls for medical data."""
    def search_pubmed(self, query: str) -> List[str]:
        print(f"[ExternalIntegrations] Searching PubMed for: {query}")
        time.sleep(0.3)
        if "glioblastoma" in query.lower():
            return [
                "Recent advances in glioblastoma treatment: A review (2023).",
                "Immunotherapy approaches for recurrent glioblastoma (2022).",
                "Temozolomide resistance mechanisms in GBM (2021)."
            ]
        return [f"Mock PubMed result for '{query}': No specific recent studies found, general medical knowledge available."]

    def get_fda_drug_info(self, drug_name: str) -> Dict[str, Any]:
        print(f"[ExternalIntegrations] Getting FDA info for drug: {drug_name}")
        time.sleep(0.2)
        if "temozolomide" in drug_name.lower():
            return {
                "drug_name": "Temozolomide",
                "indications": "Treatment of adult patients with newly diagnosed anaplastic astrocytoma.",
                "side_effects": "Nausea, vomiting, fatigue, myelosuppression."
            }
        return {"drug_name": drug_name, "status": "Mock FDA: Information not found.", "indications": "N/A"}

    def get_ehr_data(self, patient_id: str) -> Dict[str, Any]:
        print(f"[ExternalIntegrations] Retrieving EHR data for patient ID: {patient_id}")
        time.sleep(0.4)
        if patient_id == "P12345":
            return {
                "patient_id": "P12345",
                "age": 65,
                "gender": "Male",
                "diagnoses": ["Recurrent Glioblastoma", "Hypertension"],
                "medications": ["Lisinopril", "Aspirin"],
                "allergies": ["Penicillin"]
            }
        return {"patient_id": patient_id, "status": "Mock EHR: Patient not found or access denied."}

    def check_drug_interactions(self, drugs: List[str]) -> List[str]:
        print(f"[ExternalIntegrations] Checking drug interactions for: {', '.join(drugs)}")
        time.sleep(0.3)
        if "Lisinopril" in drugs and "Aspirin" in drugs:
            return ["Lisinopril and high-dose Aspirin may increase risk of renal dysfunction. Monitor kidney function."]
        return [f"Mock Drug Interaction Check: No significant interactions found for {', '.join(drugs)}."]

    def query_medical_knowledge_graph(self, concept: str) -> List[str]:
        print(f"[ExternalIntegrations] Querying medical knowledge graph for: {concept}")
        time.sleep(0.2)
        if "glioblastoma" in concept.lower():
            return ["Glioblastoma (GBM) is the most aggressive type of cancer that begins in the brain.",
                    "SNOMED CT: 363406001 | ICD-10: C71.0"]
        return [f"Mock Knowledge Graph: Basic definition for '{concept}'. No detailed graph info."]

# --- 2. Core Application Components ---

class DataProcessor:
    """Processes and consolidates raw data from external sources."""
    def extract_relevant_info(self, data: Dict[str, Any]) -> str:
        extracted = []
        if "pubmed_results" in data and data["pubmed_results"]:
            extracted.append("PubMed Research Snippets:")
            extracted.extend([f" - {res}" for res in data["pubmed_results"]])
        if "fda_info" in data and data["fda_info"] and data["fda_info"].get("drug_name"):
            extracted.append(f"FDA Drug Info for {data['fda_info']['drug_name']}:")
            extracted.append(f" - Indications: {data['fda_info'].get('indications', 'N/A')}")
            extracted.append(f" - Side Effects: {data['fda_info'].get('side_effects', 'N/A')}")
        if "ehr_data" in data and data["ehr_data"].get("patient_id"):
            extracted.append(f"EHR Data for Patient {data['ehr_data']['patient_id']}:")
            extracted.append(f" - Age: {data['ehr_data'].get('age')}, Gender: {data['ehr_data'].get('gender')}")
            extracted.append(f" - Diagnoses: {', '.join(data['ehr_data'].get('diagnoses', []))}")
            extracted.append(f" - Medications: {', '.join(data['ehr_data'].get('medications', []))}")
        if "drug_interactions" in data and data["drug_interactions"]:
            extracted.append("Potential Drug Interactions:")
            extracted.extend([f" - {interaction}" for interaction in data["drug_interactions"]])
        if "knowledge_graph_info" in data and data["knowledge_graph_info"]:
            extracted.append("Medical Knowledge Graph:")
            extracted.extend([f" - {info}" for info in data["knowledge_graph_info"]])

        return "\n".join(extracted)

class KnowledgeBase:
    """Manages the RAG system and vector database."""
    def __init__(self, embedding_model: MockEmbeddingModel, vector_db: MockVectorDatabase):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def ingest_medical_documents(self, documents: List[str]):
        print("[KnowledgeBase] Ingesting medical documents...")
        for doc_text in documents:
            embedding = self.embedding_model.encode(doc_text)
            self.vector_db.add_document(doc_text, embedding)
        print(f"[KnowledgeBase] Ingested {len(documents)} documents into vector DB.")

    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode(query)
        search_results = self.vector_db.search(query_embedding, top_k)
        return [doc["text"] for doc in search_results]

class LLMInterface:
    """Handles prompt engineering and interaction with the LLM."""
    def __init__(self, llm: MockLLM):
        self.llm = llm

    def generate_augmented_response(self, query: str, context: str) -> str:
        prompt = (
            f"You are a highly knowledgeable medical assistant. "
            f"Answer the following medical query based *only* on the provided context. "
            f"If the context does not contain enough information, state that you cannot provide a complete answer based on the given data. "
            f"Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Response:"
        )
        return self.llm.generate_response(prompt)


class OrchestrationLayer:
    """The central component coordinating all modules of the Medical Assistant."""
    def __init__(self):
        self.llm = MockLLM()
        self.embedding_model = MockEmbeddingModel()
        self.vector_db = MockVectorDatabase()
        self.external_integrations = ExternalIntegrations()
        self.data_processor = DataProcessor()
        self.knowledge_base = KnowledgeBase(self.embedding_model, self.vector_db)
        self.llm_interface = LLMInterface(self.llm)

        # Ingest some initial mock medical documents into the vector DB
        self.knowledge_base.ingest_medical_documents([
            "Glioblastoma is a highly aggressive brain tumor with poor prognosis.",
            "Standard treatment for newly diagnosed glioblastoma involves surgery, radiation, and temozolomide.",
            "Hypertension management is crucial for elderly patients, especially during cancer therapy.",
            "Drug-drug interactions can occur between ACE inhibitors (like Lisinopril) and NSAIDs (like Aspirin)."
        ])

    def process_query(self, query: str, patient_id: str = None) -> str:
        print(f"\n[OrchestrationLayer] Processing query: '{query}'")

        # 1. Retrieve relevant knowledge from internal RAG system
        rag_context_docs = self.knowledge_base.retrieve_relevant_knowledge(query)
        rag_context_text = "\n".join(rag_context_docs)
        print(f"[OrchestrationLayer] Retrieved RAG context (first 100 chars): {rag_context_text[:100]}...")

        # 2. Gather information from external sources based on query and patient context
        external_data = {}
        if "glioblastoma" in query.lower() or "brain tumor" in query.lower() or "treatment" in query.lower():
            external_data["pubmed_results"] = self.external_integrations.search_pubmed(query)
            external_data["knowledge_graph_info"] = self.external_integrations.query_medical_knowledge_graph("glioblastoma")
        
        # Check for drug names in query or potential for drug interaction check
        if any(d in query.lower() for d in ["temozolomide", "lisinopril", "aspirin", "drug interaction"]):
            external_data["fda_info"] = self.external_integrations.get_fda_drug_info("temozolomide") # Example
            
        if patient_id:
            ehr_data = self.external_integrations.get_ehr_data(patient_id)
            external_data["ehr_data"] = ehr_data
            if ehr_data.get("medications"):
                external_data["drug_interactions"] = self.external_integrations.check_drug_interactions(ehr_data["medications"])

        # 3. Process and consolidate external data
        processed_external_info = self.data_processor.extract_relevant_info(external_data)
        print(f"[OrchestrationLayer] Processed external info (first 100 chars): {processed_external_info[:100]}...")

        # 4. Combine all context for the LLM
        combined_context = f"\\n--- Retrieved Knowledge ---\\n{rag_context_text}\\n\\n--- External Data ---\\n{processed_external_info}"

        # 5. Generate augmented response using LLM
        final_response = self.llm_interface.generate_augmented_response(query, combined_context)

        return final_response

# --- 3. User Interface (CLI) ---

def main():
    print("---------------------------------------------------")
    print("  Dynamic Medical Assistant for Clinicians (Mock)  ")
    print("---------------------------------------------------")
    print("Type 'exit' to quit.")

    orchestrator = OrchestrationLayer()

    while True:
        query = input("\nClinician Query: ")
        if query.lower() == 'exit':
            break

        patient_id = input("Patient ID (optional, e.g., P12345): ")

        response = orchestrator.process_query(query, patient_id if patient_id else None)
        print("\n--- Medical Assistant Response ---")
        print(response)
        print("----------------------------------")

if __name__ == "__main__":
    main()
