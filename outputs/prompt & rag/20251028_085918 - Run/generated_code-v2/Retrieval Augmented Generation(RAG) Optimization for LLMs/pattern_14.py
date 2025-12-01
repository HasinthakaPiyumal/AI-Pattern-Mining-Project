from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts)

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, embeddings, documents):
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.documents[i] for i in indices[0]]

class MockLLMService:
    def generate(self, augmented_prompt):
        if "treatment for diabetes" in augmented_prompt.lower() and "insulin" in augmented_prompt.lower():
            return "Based on the provided context, common treatments for diabetes include insulin therapy, oral medications like metformin, and lifestyle modifications such as diet and exercise. Regular monitoring of blood glucose levels is crucial. (Ref: Contextual Document 1, Contextual Document 3)"
        elif "symptoms of hypertension" in augmented_prompt.lower() and "high blood pressure" in augmented_prompt.lower():
            return "Based on the provided context, hypertension often presents without clear symptoms, which is why it's called 'the silent killer'. However, severe cases might show headaches, shortness of breath, or nosebleeds. Regular blood pressure checks are essential for early detection. (Ref: Contextual Document 2)"
        elif "drug interactions of warfarin" in augmented_prompt.lower() and "vitamin k" in augmented_prompt.lower():
            return "Based on the provided context, Warfarin, an anticoagulant, has significant interactions with Vitamin K-rich foods and several medications. Patients on warfarin need consistent Vitamin K intake and careful management of co-administered drugs to prevent bleeding or clotting issues. (Ref: Contextual Document 4)"
        else:
            return f"Based on the provided context, I can answer your query. Query: {augmented_prompt.split('question:')[-1].strip()} (Ref: General Medical Knowledge)"

class MedicalResearchAssistant:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.knowledge_base = [
            "Document 1: Diabetes treatment often involves insulin therapy, oral hypoglycemic agents like metformin, and strict dietary control. Regular exercise is also recommended.",
            "Document 2: Hypertension, or high blood pressure, frequently has no symptoms. Regular blood pressure checks are vital. Severe hypertension can lead to headaches, vision problems, and nosebleeds.",
            "Document 3: Type 2 Diabetes management strategies include lifestyle changes, such as diet and physical activity, and pharmacological treatments like SGLT2 inhibitors and GLP-1 receptor agonists.",
            "Document 4: Warfarin is an anticoagulant medication. Its efficacy can be significantly altered by dietary Vitamin K intake and interactions with other drugs like NSAIDs and certain antibiotics.",
            "Document 5: Common cold symptoms include runny nose, sore throat, cough, and congestion. It is usually caused by rhinoviruses and typically resolves within 7-10 days.",
            "Document 6: The COVID-19 vaccine works by introducing a harmless piece of the virus's spike protein, prompting the immune system to produce antibodies and memory cells for future protection."
        ]
        self.llm_service = MockLLMService()

        initial_embeddings = self.embedding_service.encode(self.knowledge_base)
        self.vector_store = VectorStore(initial_embeddings.shape[1])
        self.vector_store.add(initial_embeddings, self.knowledge_base)

    def retrieve(self, query, k=3):
        query_embedding = self.embedding_service.encode([query])[0]
        return self.vector_store.search(query_embedding, k)

    def augment_prompt(self, user_query, retrieved_documents):
        context = "\n".join(retrieved_documents)
        return f"Context: {context}\n\nBased on the provided context, answer the following question: {user_query}"

    def answer_query(self, user_query):
        retrieved_docs = self.retrieve(user_query)
        augmented_prompt = self.augment_prompt(user_query, retrieved_docs)
        response = self.llm_service.generate(augmented_prompt)
        return response

def main():
    assistant = MedicalResearchAssistant()
    print("Medical Research Assistant AI (Type 'exit' to quit)")
    while True:
        query = input("\nEnter your medical query: ")
        if query.lower() == 'exit':
            break
        response = assistant.answer_query(query)
        print("\nAI Assistant: ", response)

if __name__ == "__main__":
    main()