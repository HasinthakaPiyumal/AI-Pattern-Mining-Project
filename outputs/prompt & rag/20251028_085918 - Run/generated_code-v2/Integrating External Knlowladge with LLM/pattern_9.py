from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class KnowledgeBase:
    def __init__(self, medical_documents):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = medical_documents
        self.document_embeddings = self.model.encode(medical_documents, convert_to_tensor=True)

    def retrieve_context(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), self.document_embeddings.cpu().numpy())
        top_k_indices = similarities.argsort()[0][::-1][:top_k]
        retrieved_contexts = [self.documents[i] for i in top_k_indices]
        return retrieved_contexts

class LLMIntegration:
    def simulate_llm_response(self, prompt):
        # This is a placeholder for actual LLM inference
        # In a real application, this would call an LLM API or a local model
        if "antibiotics" in prompt.lower() and "infection" in prompt.lower():
            return f"Based on the information, for bacterial infections, antibiotics like Amoxicillin or Azithromycin might be considered, depending on the specific type of infection and patient history. Always consult official medical guidelines and patient specifics before prescribing. Context: {prompt}"
        elif "diabetes" in prompt.lower() and "management" in prompt.lower():
            return f"Diabetes management often involves a combination of diet, exercise, and medication (e.g., Metformin, insulin). Regular monitoring of blood glucose levels is crucial. Context: {prompt}"
        elif "hypertension" in prompt.lower() and "treatment" in prompt.lower():
            return f"Treatment for hypertension can include lifestyle changes such as reduced sodium intake, regular physical activity, and medications like ACE inhibitors, ARBs, or diuretics. Context: {prompt}"
        else:
            return f"I'm a medical information assistant. I can provide general information based on my knowledge base. For specific medical advice, please consult a qualified healthcare professional. Context provided: {prompt}"

class MedicalAssistant:
    def __init__(self, medical_documents):
        self.knowledge_base = KnowledgeBase(medical_documents)
        self.llm_integration = LLMIntegration()

    def query(self, user_query, top_k=3):
        retrieved_contexts = self.knowledge_base.retrieve_context(user_query, top_k)
        
        augmented_prompt = f"User Query: {user_query}\n\nRelevant Medical Contexts:\n"
        for i, context in enumerate(retrieved_contexts):
            augmented_prompt += f"Context {i+1}: {context}\n"
        augmented_prompt += "\nBased on the provided query and relevant contexts, please provide a concise medical information summary."
        
        llm_response = self.llm_integration.simulate_llm_response(augmented_prompt)
        return llm_response

    def run_cli(self):
        print("Medical Information Assistant (CLI)")
        print("Type 'exit' to quit.")
        while True:
            user_input = input("\nEnter your medical query: ")
            if user_input.lower() == 'exit':
                break
            response = self.query(user_input)
            print(f"\nAssistant: {response}")

if __name__ == "__main__":
    # Simulate a knowledge base of medical documents
    sample_medical_docs = [
        "Amoxicillin is a penicillin antibiotic used to treat a variety of bacterial infections.",
        "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Management involves diet, exercise, and medication.",
        "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. Lifestyle modifications and medication are common treatments.",
        "Azithromycin is a macrolide antibiotic effective against a wide range of bacterial infections.",
        "Metformin is a first-line medication for type 2 diabetes, primarily working by decreasing glucose production by the liver.",
        "ACE inhibitors (e.g., Lisinopril) are a class of drugs used to treat hypertension and heart failure.",
        "Insulin therapy is crucial for type 1 diabetes and often necessary for advanced type 2 diabetes.",
        "Balanced diet with low sodium is recommended for managing hypertension.",
        "Regular physical activity helps in managing both diabetes and hypertension.",
        "Bacterial infections can be treated with appropriate antibiotics, whereas viral infections typically do not respond to antibiotics."
    ]

    assistant = MedicalAssistant(sample_medical_docs)
    assistant.run_cli()