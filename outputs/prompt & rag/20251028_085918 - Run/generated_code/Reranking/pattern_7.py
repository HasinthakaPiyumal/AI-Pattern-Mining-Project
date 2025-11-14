class MedicalKnowledgeBase:
    def __init__(self):
        # A very simplified medical knowledge base for demonstration
        self.documents = {
            "diabetes_symptoms": "Symptoms of diabetes include frequent urination, increased thirst, unexplained weight loss, fatigue, and blurred vision. Type 1 diabetes often develops quickly, while type 2 diabetes symptoms are usually milder and develop more slowly.",
            "hypertension_treatment": "Treatment for hypertension often involves lifestyle changes such as a healthy diet, regular exercise, and reduced sodium intake. Medications like ACE inhibitors, ARBs, diuretics, and beta-blockers may also be prescribed.",
            "influenza_prevention": "Influenza prevention includes annual vaccination, frequent hand washing, avoiding close contact with sick individuals, and covering coughs and sneezes. Antiviral drugs can be used for treatment.",
            "common_cold_vs_flu": "The common cold and flu share similar symptoms but the flu is typically more severe. Flu symptoms often include fever, body aches, extreme tiredness, and a dry cough. Colds are usually milder with symptoms like runny nose, sore throat, and sneezing.",
            "migraine_causes": "Migraine causes are not fully understood but are thought to involve changes in the brain and its interaction with the trigeminal nerve. Triggers can include stress, certain foods, changes in sleep patterns, hormonal shifts, and bright lights."
        }

    def retrieve_documents(self, query: str, top_n: int = 3) -> list:
        """
        Simulates retrieving relevant medical documents based on keywords in the query.
        In a real system, this would involve embeddings, vector databases, etc.
        """
        query_keywords = set(query.lower().split())
        scores = {}

        for doc_id, content in self.documents.items():
            doc_keywords = set(content.lower().split())
            # Simple keyword overlap scoring
            overlap = len(query_keywords.intersection(doc_keywords))
            if overlap > 0:
                scores[doc_id] = overlap
        
        # Sort by score and return top_n documents
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        retrieved_content = []
        for doc_id, _ in sorted_docs[:top_n]:
            retrieved_content.append((doc_id, self.documents[doc_id]))
        return retrieved_content