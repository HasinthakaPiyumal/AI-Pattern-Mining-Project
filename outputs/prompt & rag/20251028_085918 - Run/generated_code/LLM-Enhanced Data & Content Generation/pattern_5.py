class MedicalKnowledgeBase:
    def __init__(self):
        self.documents = [
            {
                "id": "doc1",
                "text": "Diabetes mellitus, commonly known as diabetes, is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy. With diabetes, your body either doesn't make enough insulin or can't effectively use the insulin it does make."
            },
            {
                "id": "doc2",
                "text": "Type 1 diabetes is an autoimmune disease where the body's immune system attacks and destroys the insulin-producing cells in the pancreas. It typically develops in children and young adults."
            },
            {
                "id": "doc3",
                "text": "Type 2 diabetes occurs when the body becomes resistant to insulin or doesn't make enough insulin. It's often associated with lifestyle factors like obesity and inactivity and is more common in adults."
            },
            {
                "id": "doc4",
                "text": "Managing diabetes involves monitoring blood sugar, healthy eating, regular physical activity, and sometimes medication or insulin therapy. Regular check-ups with a healthcare provider are crucial."
            },
            {
                "id": "doc5",
                "text": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Risk factors include age, family history, obesity, and an inactive lifestyle."
            },
            {
                "id": "doc6",
                "text": "Treatment for hypertension often includes lifestyle modifications like diet and exercise, and medications such as diuretics, ACE inhibitors, or calcium channel blockers. Regular monitoring of blood pressure is essential."
            }
        ]

    def retrieve_documents(self, query: str, top_k: int = 2) -> list:
        # In a real system, this would involve vector embeddings and similarity search.
        # For simplicity, we'll do a keyword-based search.
        results = []
        query_words = query.lower().split()
        for doc in self.documents:
            if any(word in doc["text"].lower() for word in query_words):
                results.append(doc)
        return results[:top_k]

    def get_document_by_id(self, doc_id: str) -> dict:
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None
