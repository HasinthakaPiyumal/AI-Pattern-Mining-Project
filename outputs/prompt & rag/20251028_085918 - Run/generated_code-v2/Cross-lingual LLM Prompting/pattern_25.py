
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class KnowledgeBase:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.faq_data = [] # Stores {'query_en', 'answer_en', 'query_es', 'answer_es', 'embedding'}

    def add_faq_entry(self, query_en: str, answer_en: str, query_es: str, answer_es: str):
        # For simplicity, we'll embed the English query. For true cross-lingual search,
        # you might embed both or use a specialized cross-lingual retriever.
        embedding = self.model.encode(query_en, convert_to_tensor=False)
        self.faq_data.append({
            'query_en': query_en,
            'answer_en': answer_en,
            'query_es': query_es,
            'answer_es': answer_es,
            'embedding': embedding
        })

    def get_relevant_examples(self, user_query: str, target_lang: str, k: int = 3):
        user_query_embedding = self.model.encode(user_query, convert_to_tensor=False)
        
        # Prepare embeddings for comparison
        embeddings = np.array([entry['embedding'] for entry in self.faq_data])
        
        if len(embeddings) == 0:
            return []

        similarities = cosine_similarity([user_query_embedding], embeddings)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        relevant_examples = []
        for i in top_k_indices:
            entry = self.faq_data[i]
            relevant_examples.append({
                'source_query': entry['query_en'],
                'source_answer': entry['answer_en'],
                'target_query': entry[f'query_{target_lang}'],
                'target_answer': entry[f'answer_{target_lang}']
            })
        return relevant_examples

# Example Usage (for testing purposes)
if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.add_faq_entry(
        query_en="How do I reset my password?", 
        answer_en="You can reset your password by going to the 'Forgot Password' link on the login page.",
        query_es="¿Cómo restablezco mi contraseña?",
        answer_es="Puede restablecer su contraseña yendo al enlace 'Olvidé mi contraseña' en la página de inicio de sesión."
    )
    kb.add_faq_entry(
        query_en="What are your operating hours?", 
        answer_en="Our customer support is available 24/7.",
        query_es="¿Cuáles son sus horas de operación?",
        answer_es="Nuestro servicio de atención al cliente está disponible las 24 horas del día, los 7 días de la semana."
    )
    kb.add_faq_entry(
        query_en="Where can I find my order history?", 
        answer_en="Your order history is available in the 'My Account' section under 'Orders'.",
        query_es="¿Dónde puedo encontrar mi historial de pedidos?",
        answer_es="Su historial de pedidos está disponible en la sección 'Mi Cuenta' en 'Pedidos'."
    )

    print("Relevant examples for 'I forgot my password' (target_lang=es):")
    examples = kb.get_relevant_examples("I forgot my password", "es", k=1)
    for ex in examples:
        print(ex)

    print("\nRelevant examples for 'Need help with my purchase history' (target_lang=es):")
    examples = kb.get_relevant_examples("Need help with my purchase history", "es", k=1)
    for ex in examples:
        print(ex)
