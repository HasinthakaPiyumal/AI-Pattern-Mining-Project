import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class MultilingualFAQManager:
    def __init__(self):
        self.faqs = {}
        self._next_id = 0

    def add_faq(self, english_q, english_a, target_q, target_a, target_lang="es"):
        faq_id = self._next_id
        self.faqs[faq_id] = {
            "english_q": english_q,
            "english_a": english_a,
            f"target_q_{target_lang}": target_q,
            f"target_a_{target_lang}": target_a,
        }
        self._next_id += 1
        return faq_id

    def get_faq_by_id(self, faq_id):
        return self.faqs.get(faq_id)

    def get_all_english_questions(self):
        return {faq_id: data["english_q"] for faq_id, data in self.faqs.items()}

class EmbeddingModel:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

class CrossLingualFAQRetriever:
    def __init__(self, faq_manager, embedding_model):
        self.faq_manager = faq_manager
        self.embedding_model = embedding_model
        self.faq_embeddings = None
        self.faq_ids = None
        self._build_index()

    def _build_index(self):
        english_questions = self.faq_manager.get_all_english_questions()
        self.faq_ids = list(english_questions.keys())
        questions_list = list(english_questions.values())
        if questions_list:
            self.faq_embeddings = self.embedding_model.encode(questions_list)
        else:
            self.faq_embeddings = np.array([])

    def retrieve_top_k(self, query, top_k=3):
        if not self.faq_ids:
            return []

        query_embedding = self.embedding_model.encode([query])[0]

        # Calculate cosine similarity
        similarities = np.dot(self.faq_embeddings, query_embedding) / \
                       (np.linalg.norm(self.faq_embeddings, axis=1) * np.linalg.norm(query_embedding))

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.faq_ids[i] for i in top_k_indices]

class InCLTChatbot:
    def __init__(self, llm_model_name="distilgpt2", top_k_retrieval=2):
        self.faq_manager = MultilingualFAQManager()
        self._populate_faqs()
        self.embedding_model = EmbeddingModel()
        self.retriever = CrossLingualFAQRetriever(self.faq_manager, self.embedding_model)
        self.llm_pipeline = pipeline("text-generation", model=llm_model_name)
        self.top_k_retrieval = top_k_retrieval

    def _populate_faqs(self):
        # Sample FAQs (English and Spanish translations)
        self.faq_manager.add_faq(
            "What are your operating hours?",
            "Our operating hours are Monday to Friday, 9 AM to 5 PM local time.",
            "¿Cuáles son sus horarios de atención?",
            "Nuestro horario de atención es de lunes a viernes, de 9 a.m. a 5 p.m. hora local.",
            "es"
        )
        self.faq_manager.add_faq(
            "How can I reset my password?",
            "You can reset your password by visiting our website and clicking on the 'Forgot Password' link.",
            "¿Cómo puedo restablecer mi contraseña?",
            "Puede restablecer su contraseña visitando nuestro sitio web y haciendo clic en el enlace 'Olvidé mi contraseña'.",
            "es"
        )
        self.faq_manager.add_faq(
            "What payment methods do you accept?",
            "We accept major credit cards, PayPal, and bank transfers.",
            "¿Qué métodos de pago aceptan?",
            "Aceptamos las principales tarjetas de crédito, PayPal y transferencias bancarias.",
            "es"
        )
        self.faq_manager.add_faq(
            "How do I contact customer support?",
            "You can reach customer support via email at support@example.com or by calling 1-800-123-4567.",
            "¿Cómo me pongo en contacto con el servicio de atención al cliente?",
            "Puede comunicarse con el servicio de atención al cliente por correo electrónico a support@example.com o llamando al 1-800-123-4567.",
            "es"
        )

    def _construct_in_context_prompt(self, user_query, target_language):
        retrieved_faq_ids = self.retriever.retrieve_top_k(user_query, self.top_k_retrieval)
        
        prompt_parts = [
            f"You are a helpful customer support assistant. Answer the user's question in {target_language}."
            "Use the provided context and examples to formulate your answer. If the answer is not in the context, state that you don't know."
            "Here are some examples of questions and answers in English and their {target_language} translations:",
        ]

        for faq_id in retrieved_faq_ids:
            faq = self.faq_manager.get_faq_by_id(faq_id)
            if faq:
                prompt_parts.append(f"\nEnglish Question: {faq['english_q']}")
                prompt_parts.append(f"English Answer: {faq['english_a']}")
                prompt_parts.append(f"{target_language.capitalize()} Question: {faq[f'target_q_{target_language}']}")
                prompt_parts.append(f"{target_language.capitalize()} Answer: {faq[f'target_a_{target_language}']}")

        prompt_parts.append(f"\nNow, answer the following question in {target_language} based on the context provided:")
        prompt_parts.append(f"User Question ({target_language}): {user_query}")
        prompt_parts.append(f"Chatbot Answer ({target_language}):")

        return "\n".join(prompt_parts)

    def answer_query(self, user_query, target_language="es"):
        prompt = self._construct_in_context_prompt(user_query, target_language)
        
        # For distilgpt2, we might need to be careful with prompt length and output generation
        # Setting max_new_tokens to avoid very long or irrelevant generations
        generation = self.llm_pipeline(prompt, max_new_tokens=50, num_return_sequences=1, truncation=True)
        
        # Extract the generated text and try to find the answer part
        generated_text = generation[0]['generated_text']
        
        # Simple post-processing to get only the answer part after the final prompt
        answer_start_tag = f"Chatbot Answer ({target_language}):"
        if answer_start_tag in generated_text:
            answer = generated_text.split(answer_start_tag, 1)[1].strip()
            # Heuristic to remove potential prompt repetition or incomplete sentences from small LLM
            if "User Question" in answer:
                answer = answer.split("User Question", 1)[0].strip()
            return answer
        else:
            return generated_text # Fallback if extraction fails for some reason

if __name__ == "__main__":
    print("Initializing Multilingual Customer Support Chatbot...")
    try:
        chatbot = InCLTChatbot()
        print("Chatbot initialized. Type 'exit' to quit.")

        while True:
            user_input = input("\nEnter your question (e.g., '¿Cómo restablezco mi contraseña?') or 'exit': ")
            if user_input.lower() == 'exit':
                break
            
            # For simplicity, target language is hardcoded to 'es' (Spanish) for this example
            # In a real application, this would be determined by user settings or language detection.
            target_lang = "es"

            response = chatbot.answer_query(user_input, target_lang)
            print(f"\nChatbot ({target_lang}): {response}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have 'torch', 'numpy', 'sentence-transformers', and 'transformers' libraries installed.")
        print("You might need to install them using: pip install torch numpy sentence-transformers transformers")
