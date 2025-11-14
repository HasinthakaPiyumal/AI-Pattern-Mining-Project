from typing import List, Dict

class MedicalKnowledgeBase:
    def __init__(self):
        # In a real application, this would be a vector database (e.g., Chroma, Pinecone, Faiss)
        # and documents would be loaded and embedded.
        self.documents = [
            {
                "id": "doc1",
                "title": "Latest Protocols for Type 2 Diabetes",
                "content": "New guidelines recommend SGLT2 inhibitors and GLP-1 receptor agonists early for many patients with type 2 diabetes, especially those with cardiovascular or renal disease. Lifestyle modifications remain crucial, including diet and exercise. Metformin is still the first-line therapy for most, but personalized approaches are emphasized. Regular monitoring of HbA1c, blood pressure, and lipids is essential. Refer to the ADA 2023 guidelines for full details.",
                "source": "ADA 2023 Guidelines"
            },
            {
                "id": "doc2",
                "title": "Differential Diagnosis for Persistent Cough",
                "content": "Persistent cough, defined as lasting more than 8 weeks, has several common causes including upper airway cough syndrome (UACS), asthma, and gastroesophageal reflux disease (GERD). Less common causes include chronic bronchitis, bronchiectasis, and ACE inhibitor use. In elderly patients, consider heart failure or lung malignancy. A thorough history, physical exam, and sometimes imaging or pulmonary function tests are required for accurate diagnosis.",
                "source": "UpToDate Medical Journal"
            },
            {
                "id": "doc3",
                "title": "Drug Interactions: Metformin and Contrast Dye",
                "content": "Patients on metformin undergoing procedures involving iodinated contrast media are at risk for lactic acidosis. Metformin should be temporarily discontinued at the time of or prior to the procedure and withheld for 48 hours afterward, or until renal function has been re-evaluated and found to be normal. This precaution is critical, especially in patients with pre-existing renal impairment.",
                "source": "FDA Drug Safety Communication"
            },
            {
                "id": "doc4",
                "title": "Basic information about Paracetamol",
                "content": "Paracetamol (acetaminophen) is a common pain reliever and fever reducer. It is available over-the-counter and is generally safe when used as directed. Overdosing can lead to severe liver damage. It works by inhibiting prostaglandin synthesis, primarily in the central nervous system.",
                "source": "NHS UK"
            },
            {
                "id": "doc5",
                "title": "Hypertension Management Guidelines",
                "content": "Current guidelines for hypertension emphasize target blood pressure values, often less than 130/80 mmHg for many adults. First-line agents include ACE inhibitors, ARBs, thiazide diuretics, and calcium channel blockers. Lifestyle modifications such as reduced sodium intake, regular exercise, and weight management are fundamental to treatment. Regular follow-up is necessary to adjust medications and monitor for complications.",
                "source": "AHA/ACC 2024 Guidelines"
            },
            {
                "id": "doc6",
                "title": "Pediatric Asthma Exacerbation Management",
                "content": "Management of pediatric asthma exacerbations typically involves short-acting beta-agonists (SABAs), systemic corticosteroids, and oxygen therapy if hypoxic. Close monitoring of respiratory distress and oxygen saturation is crucial. In severe cases, magnesium sulfate or heliox may be considered. Education of parents on trigger avoidance and proper inhaler technique is also vital.",
                "source": "GINA 2023 Report"
            }
        ]
        # In a real system, documents would be embedded here for vector search
        self.document_embeddings = {}

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        # This is a highly simplified keyword-based retrieval for demonstration.
        # A real system would use semantic search with embeddings and a vector database.
        print(f"[KnowledgeBase] Retrieving documents for query: '{query}'")
        retrieved = []
        query_lower = query.lower()
        for doc in self.documents:
            if query_lower in doc["title"].lower() or query_lower in doc["content"].lower():
                retrieved.append(doc)
        
        # Sort by a very basic relevance score (e.g., number of query term occurrences)
        # For a real system, this would be similarity score from vector search
        retrieved.sort(key=lambda doc: doc["content"].lower().count(query_lower) + doc["title"].lower().count(query_lower), reverse=True)
        
        return retrieved[:top_k]

    def get_document_content(self, doc_id: str) -> str:
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc["content"]
        return ""

