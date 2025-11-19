"""
knowledge_base.py: Simulates a medical knowledge base.
"""

class MedicalKnowledgeBase:
    def __init__(self):
        self.documents = {
            "doc_001": {"type": "drug_info", "title": "Paracetamol Dosage", "content": "Paracetamol (Acetaminophen) is a common pain reliever. Adult dosage typically ranges from 500mg to 1000mg every 4-6 hours, not exceeding 4000mg in 24 hours. Pediatric dosage varies by weight. Always consult a physician."},
            "doc_002": {"type": "journal_article", "title": "Advances in Alzheimer's Treatment", "content": "Recent studies show promising results with new amyloid-beta targeting therapies for early-stage Alzheimer's disease. Clinical trials indicate a slowing of cognitive decline in select patient groups. Further research is ongoing."},
            "doc_003": {"type": "clinical_guideline", "title": "Hypertension Management Guidelines", "content": "2023 guidelines recommend lifestyle modifications as first-line treatment for hypertension. Pharmacological intervention often starts with ACE inhibitors or ARBs, aiming for a blood pressure target of less than 130/80 mmHg in most adults."},
            "doc_004": {"type": "drug_info", "title": "Insulin Administration", "content": "Insulin is administered via subcutaneous injection. Different types of insulin have varying onset and duration of action. Proper injection technique and site rotation are crucial to prevent complications."},
            "doc_005": {"type": "journal_article", "title": "Impact of Gut Microbiome on Diabetes", "content": "Emerging evidence suggests a significant role of the gut microbiome in the development and progression of type 2 diabetes. Microbiome modulation could be a future therapeutic strategy."},
            "doc_006": {"type": "clinical_guideline", "title": "Pediatric Asthma Protocol", "content": "Acute asthma exacerbations in children are managed with inhaled short-acting beta-agonists (SABAs) and systemic corticosteroids. Long-term control often involves inhaled corticosteroids (ICS)."}
        }

    def get_document_by_id(self, doc_id: str) -> dict:
        """Retrieves a document by its ID."""
        return self.documents.get(doc_id)

    def keyword_search(self, query: str, top_k: int = 3) -> list:
        """Performs a simple keyword search across document content and title."""
        results = []
        query_lower = query.lower()
        for doc_id, doc in self.documents.items():
            if query_lower in doc['title'].lower() or query_lower in doc['content'].lower():
                results.append((doc_id, doc['title'], doc['content']))
        # Sort results by some relevance score (simple for now: just return top_k)
        return results[:top_k]

    def get_all_documents_text(self) -> list[str]:
        """Returns a list of all document contents."""
        return [doc["content"] for doc in self.documents.values()]

    def get_document_ids(self) -> list[str]:
        """Returns a list of all document IDs."""
        return list(self.documents.keys())

    def get_document_metadata(self, doc_id: str) -> dict:
        """Returns metadata for a given document ID."""
        doc = self.documents.get(doc_id)
        if doc:
            return {"type": doc["type"], "title": doc["title"]}
        return {} 
