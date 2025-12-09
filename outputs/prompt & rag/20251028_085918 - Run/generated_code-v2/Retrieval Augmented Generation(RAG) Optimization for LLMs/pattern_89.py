from sentence_transformers import SentenceTransformer, util, CrossEncoder
import numpy as np

class IntelliDiagnosisAssistant:
    def __init__(self, 
                 embedding_model_name="all-MiniLM-L6-v2", 
                 reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 medical_knowledge_base=None):
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker_model = CrossEncoder(reranker_model_name)

        if medical_knowledge_base is None:
            self.medical_knowledge_base = [
                "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored or used for energy. With diabetes, your body either doesn't make enough insulin or can't effectively use the insulin it does make.",
                "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Blood pressure is determined by the amount of blood your heart pumps and the amount of resistance to blood flow in your arteries.",
                "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out and shortness of breath. For some people, asthma is a minor nuisance. For others, it can be a major problem that interferes with daily activities and may lead to a life-threatening asthma attack.",
                "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold. Adults can expect to have two or three colds a year. Infants and young children might have even more frequent colds.",
                "Myocardial infarction, commonly known as a heart attack, occurs when blood flow to the heart muscle is blocked, often by a blood clot. Without blood, the heart muscle begins to die. Symptoms include chest pain, shortness of breath, and pain in the left arm.",
                "A stroke occurs when the blood supply to part of your brain is interrupted or reduced, depriving brain tissue of oxygen and nutrients. Brain cells begin to die in minutes. A stroke is a medical emergency, and prompt treatment is crucial.",
                "Gastroesophageal reflux disease (GERD) is a chronic digestive disease. GERD occurs when stomach acid or, occasionally, stomach content, flows back into your food pipe (esophagus). The backwash (reflux) irritates the lining of your esophagus and causes signs and symptoms such as heartburn.",
                "Pneumonia is an infection that inflames air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.",
                "Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of arthritis are joint pain and stiffness, which typically worsen with age. The most common types of arthritis are osteoarthritis and rheumatoid arthritis.",
                "Migraine is a headache of varying intensity, often accompanied by nausea and sensitivity to light and sound. Migraine attacks can cause significant pain for hours to days. Medications can help prevent some migraines and make them less painful."
            ]
        else:
            self.medical_knowledge_base = medical_knowledge_base

        self.corpus_embeddings = self.embedding_model.encode(self.medical_knowledge_base, convert_to_tensor=True)

    def retrieve(self, query: str, top_n: int = 50) -> list:
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        
        # Calculate cosine similarity between query and corpus embeddings
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Get top-N results
        top_results_indices = np.argsort(-cos_scores.cpu().numpy())[:top_n]
        retrieved_contexts = [self.medical_knowledge_base[i] for i in top_results_indices]
        
        return retrieved_contexts

    def rerank(self, query: str, retrieved_contexts: list, top_k: int = 10) -> list:
        if not retrieved_contexts:
            return []

        # Create pairs for the cross-encoder
        sentence_pairs = [[query, context] for context in retrieved_contexts]
        
        # Get relevance scores
        rerank_scores = self.reranker_model.predict(sentence_pairs)
        
        # Sort contexts by rerank scores and select top-k
        ranked_indices = np.argsort(-rerank_scores)[:top_k]
        final_contexts = [retrieved_contexts[i] for i in ranked_indices]
        
        return final_contexts

    def _generate_answer_with_llm(self, query: str, contexts: list) -> str:
        # In a real application, this would involve calling a powerful medical LLM
        # (e.g., via transformers library, OpenAI API, Llama-2, Flan-T5, etc.)
        # For this example, we'll simulate the LLM's response.
        
        prompt = f"Given the following medical contexts, answer the question accurately and concisely:\n\n"
        for i, context in enumerate(contexts):
            prompt += f"Context {i+1}: {context}\n"
        prompt += f"\nQuestion: {query}\nAnswer:"
        
        # Simulate LLM response based on contexts
        if any("Diabetes" in c for c in contexts) and "blood sugar" in query:
            return "Based on the provided contexts, high blood sugar is a key characteristic of Diabetes mellitus, a metabolic disease where the body struggles to produce or effectively use insulin."
        elif any("Hypertension" in c for c in contexts) and "blood pressure" in query:
            return "The contexts indicate that Hypertension, or high blood pressure, is a chronic condition that can lead to heart disease due to the sustained force of blood against artery walls."
        elif any("Asthma" in c for c in contexts) and "breathing difficulty" in query:
             return "According to the contexts, Asthma is a condition causing narrowed, swollen airways and mucus production, leading to breathing difficulties, wheezing, and coughing."
        else:
            return f"Based on the provided medical contexts, and the question '{query}', a precise answer would be generated by a specialized medical LLM. This simulation provides a placeholder answer."

    def answer_query(self, query: str, top_n: int = 50, top_k: int = 5) -> str:
        # Step 1: Retrieve
        retrieved_contexts = self.retrieve(query, top_n)
        
        # Step 2: Rerank
        final_contexts = self.rerank(query, retrieved_contexts, top_k)
        
        # Step 3: Generate
        answer = self._generate_answer_with_llm(query, final_contexts)
        
        return answer

if __name__ == "__main__":
    # Example Usage
    assistant = IntelliDiagnosisAssistant()

    queries = [
        "What causes high blood sugar?",
        "What are the primary risks associated with sustained high blood pressure?",
        "Describe the symptoms of asthma.",
        "What happens during a heart attack?",
        "What is GERD and its main symptom?"
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        answer = assistant.answer_query(query, top_n=10, top_k=3) 
        print(f"Assistant's Answer: {answer}")

    # Example with a custom knowledge base (can be loaded from files in real app)
    custom_kb = [
        "Migraine is a severe headache often accompanied by sensitivity to light and sound.",
        "Common cold is a viral infection of the nose and throat.",
        "A common symptom of flu is fever and body aches.",
        "Bacterial infections can be treated with antibiotics."
    ]
    custom_assistant = IntelliDiagnosisAssistant(medical_knowledge_base=custom_kb)
    print(f"\n--- Query: What are symptoms of migraine? (Custom KB) ---")
    custom_answer = custom_assistant.answer_query("What are symptoms of migraine?", top_n=5, top_k=2)
    print(f"Assistant's Answer: {custom_answer}")