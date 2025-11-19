class MedicalKnowledgeBase:
    """
    Simulates a medical knowledge base for document storage and retrieval.
    In a real application, this would involve vector databases (e.g., Chroma, FAISS)
    and embedding models (e.g., Sentence-Transformers) for efficient semantic search.
    """
    def __init__(self):
        self._documents = []
        self._load_dummy_data()

    def _load_dummy_data(self):
        # Simulate loading various medical documents
        self._documents.append(
            "Aspirin is commonly used as a pain reliever and to reduce fever. It also has anti-inflammatory properties."
        )
        self._documents.append(
            "Diabetes Mellitus Type 2 is characterized by insulin resistance and relative insulin deficiency. Management often includes diet, exercise, and medication like Metformin."
        )
        self._documents.append(
            "Common symptoms of influenza include fever, cough, sore throat, and body aches. Antiviral medications can be used in some cases."
        )
        self._documents.append(
            "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. Lifestyle changes and medications like ACE inhibitors are standard treatments."
        )
        self._documents.append(
            "COVID-19 often presents with respiratory symptoms, fever, and fatigue. Vaccination and supportive care are crucial."
        )
        self._documents.append(
            "Metformin is an oral antidiabetic drug in the biguanide class, used to treat type 2 diabetes."
        )

    def add_document(self, document: str):
        self._documents.append(document)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Simulates retrieving relevant documents based on a query.
        In a real system, this would be a semantic search using embeddings.
        Here, it's a simple keyword-based matching for demonstration.
        """
        query_words = set(query.lower().split())
        scores = []
        for doc in self._documents:
            doc_words = set(doc.lower().split())
            common_words = query_words.intersection(doc_words)
            score = len(common_words) # Simple scoring based on common keywords
            if score > 0:
                scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:top_k]]

