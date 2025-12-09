from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class MedicalResearchAssistant:
    def __init__(self, knowledge_base_texts, embedding_model_name="all-MiniLM-L6-v2", chunk_size=200):
        self.knowledge_base_texts = knowledge_base_texts
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.faiss_index = None
        self.text_chunks = []
        self._build_knowledge_base()

    def _chunk_text(self, text):
        chunks = []
        words = text.split()
        for i in range(0, len(words), self.chunk_size):
            chunks.append(" ".join(words[i:i + self.chunk_size]))
        return chunks

    def _build_knowledge_base(self):
        print("Building knowledge base...")
        all_chunks = []
        for text in self.knowledge_base_texts:
            all_chunks.extend(self._chunk_text(text))
        self.text_chunks = all_chunks

        if not self.text_chunks:
            print("No text chunks to process for the knowledge base.")
            return
        
        print(f"Embedding {len(self.text_chunks)} text chunks...")
        chunk_embeddings = self.embedding_model.encode(self.text_chunks)
        d = chunk_embeddings.shape[1]  # Dimension of embeddings
        self.faiss_index = faiss.IndexFlatL2(d) # L2 distance for similarity search
        self.faiss_index.add(np.array(chunk_embeddings).astype('float32'))
        print("Knowledge base built and FAISS index created.")

    def retrieve_context(self, query, top_k=3):
        if self.faiss_index is None:
            return []

        query_embedding = self.embedding_model.encode([query])[0]
        D, I = self.faiss_index.search(np.array([query_embedding]).astype('float32'), top_k)
        
        retrieved_chunks = []
        for idx in I[0]:
            if idx < len(self.text_chunks):
                retrieved_chunks.append(self.text_chunks[idx])
        return retrieved_chunks

    def _simulate_llm_response(self, prompt):
        # This is a mock LLM. In a real application, you would integrate with a real LLM like Mistral, Llama, etc.
        # using the transformers library or an API call.
        if "COVID-19 treatment" in prompt:
            return "Based on the retrieved context, common treatments for COVID-19 include antiviral medications like Paxlovid, supportive care, and in severe cases, oxygen therapy and corticosteroids. Vaccination is crucial for prevention. Always consult the latest clinical guidelines from reputable sources like the WHO or CDC."
        elif "diabetes management" in prompt:
            return "The retrieved information suggests that diabetes management involves a combination of diet control, regular exercise, blood glucose monitoring, and medication (oral hypoglycemics or insulin) as prescribed by a healthcare professional. Lifestyle modifications are key."
        elif "cancer types" in prompt:
            return "From the provided documents, common types of cancer include carcinoma (originating in skin or tissues covering internal organs), sarcoma (in bone or soft tissues), leukemia (of the blood), and lymphoma (of the immune system). Early diagnosis and tailored treatment are critical."
        else:
            return f"I'm a simulated AI assistant. Based on the context provided, here's a general answer to your query: '{query}'. Please refer to actual medical literature for definitive information."

    def get_answer(self, query):
        print(f"Processing query: '{query}'")
        retrieved_context = self.retrieve_context(query)
        
        if not retrieved_context:
            print("No relevant context found. Generating a general response.")
            llm_prompt = f"Answer the following medical question: {query}"
        else:
            context_str = "\n---\n".join(retrieved_context)
            llm_prompt = (
                f"You are a medical research assistant. Use the following medical context to answer the question below. "
                f"If the answer is not in the context, state that you cannot provide a definitive answer based on the given information.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
        
        print("Sending augmented prompt to LLM...")
        answer = self._simulate_llm_response(llm_prompt)
        return answer

if __name__ == "__main__":
    # Mock Medical Knowledge Base
    medical_data = [
        "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Symptoms range from mild to severe, including fever, cough, fatigue, and loss of taste or smell. Prevention involves vaccination, masking, and social distancing. Treatments may include antiviral drugs like remdesivir or Paxlovid, and supportive care for symptoms. Severe cases may require hospitalization, oxygen therapy, and corticosteroids. The CDC and WHO regularly update guidelines.",
        "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Type 1 diabetes is an autoimmune condition where the body does not produce insulin. Type 2 diabetes occurs when the body either doesn't produce enough insulin or can't effectively use the insulin it produces. Management typically includes dietary changes, regular physical activity, blood glucose monitoring, and medication, which can include oral drugs or insulin injections. Complications can include heart disease, kidney disease, and nerve damage. Early diagnosis and consistent management are crucial.",
        "Cancer is a disease characterized by the uncontrolled growth and spread of abnormal cells. There are over 100 types of cancer, including carcinomas (originating in epithelial cells), sarcomas (in connective tissue), leukemia (of the blood), and lymphoma (of the immune system). Treatment options vary widely depending on the type and stage of cancer and may include surgery, chemotherapy, radiation therapy, immunotherapy, and targeted therapy. Early detection significantly improves prognosis. Regular screenings and a healthy lifestyle can reduce risk.",
        "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Risk factors include obesity, lack of exercise, high salt intake, and family history. Treatment often involves lifestyle changes such as diet modification (e.g., DASH diet), regular exercise, and medication like diuretics, ACE inhibitors, or beta-blockers. Regular monitoring is essential.",
        "The human immune system is a complex network of cells, tissues, and organs that work together to protect the body from pathogens. It includes innate immunity (first line of defense) and adaptive immunity (specific and memory-based). Vaccinations work by stimulating the adaptive immune system to produce antibodies without causing the disease itself. Autoimmune diseases occur when the immune system mistakenly attacks the body's own healthy cells.",
        "Pharmacology is the branch of medicine concerned with the uses, effects, and modes of action of drugs. It studies how drugs interact with biological systems to produce therapeutic effects or adverse reactions. Drug development involves extensive research, clinical trials, and regulatory approval processes. Understanding pharmacokinetics (what the body does to the drug) and pharmacodynamics (what the drug does to the body) is fundamental.",
        "Neurology is the medical specialty dealing with disorders of the nervous system, including the brain, spinal cord, and nerves. Common neurological conditions include stroke, epilepsy, Parkinson's disease, Alzheimer's disease, and multiple sclerosis. Diagnosis often involves imaging techniques like MRI and CT scans, as well as neurological examinations. Treatment varies widely but may include medication, physical therapy, and surgery."
    ]

    # Initialize the Medical Research Assistant
    assistant = MedicalResearchAssistant(medical_data)

    # Example Queries
    queries = [
        "What are the current treatment options for COVID-19?",
        "How is type 2 diabetes typically managed?",
        "Can you explain different types of cancer?",
        "What causes hypertension?",
        "How do vaccinations work?",
        "What is the role of pharmacology in medicine?",
        "What diseases does neurology deal with?",
        "Tell me about quantum physics."
    ]

    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1} ---")
        response = assistant.get_answer(query)
        print(f"Assistant: {response}")
