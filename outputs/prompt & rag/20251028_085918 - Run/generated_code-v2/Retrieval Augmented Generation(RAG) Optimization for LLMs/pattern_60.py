from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import torch

class MedicalQASystem:
    def __init__(self, knowledge_base_docs, retriever_model_name='multi-qa-mpnet-base-dot-v1', generator_model_name='facebook/bart-large-cnn'):
        self.knowledge_base_docs = knowledge_base_docs
        self.retriever_model = SentenceTransformer(retriever_model_name)
        self.generator_tokenizer = BartTokenizer.from_pretrained(generator_model_name)
        self.generator_model = BartForConditionalGeneration.from_pretrained(generator_model_name)
        self.faiss_index = None
        self._build_faiss_index()

    def _build_faiss_index(self):
        print("Building FAISS index...")
        document_embeddings = self.retriever_model.encode(self.knowledge_base_docs, convert_to_tensor=True)
        self.faiss_index = faiss.IndexFlatL2(document_embeddings.shape[1])
        self.faiss_index.add(document_embeddings.cpu().numpy())
        print("FAISS index built.")

    def answer_question(self, query, top_k=5):
        query_embedding = self.retriever_model.encode([query], convert_to_tensor=True)
        
        D, I = self.faiss_index.search(query_embedding.cpu().numpy(), top_k)
        retrieved_doc_indices = I[0]
        
        retrieved_passages = [self.knowledge_base_docs[idx] for idx in retrieved_doc_indices]
        
        context = " ".join(retrieved_passages)
        
        input_text = f"question: {query} context: {context}"
        
        inputs = self.generator_tokenizer([input_text], max_length=1024, return_tensors='pt', truncation=True)
        
        # Generate a response
        summary_ids = self.generator_model.generate(
            inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True
        )
        
        answer = self.generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return answer, retrieved_passages

if __name__ == '__main__':
    # Demo Medical Knowledge Base
    medical_docs = [
        "Insulin is a hormone produced by the pancreas that helps regulate blood sugar. Diabetes mellitus is a chronic condition that affects how your body turns food into energy.",
        "Type 1 diabetes is an autoimmune disease where the body does not produce insulin. Type 2 diabetes occurs when the body doesn't use insulin properly and can't keep blood sugar at normal levels.",
        "Symptoms of diabetes include increased thirst, frequent urination, blurred vision, and fatigue. Early diagnosis and management are crucial.",
        "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
        "Common treatments for hypertension include lifestyle changes like diet and exercise, and medications such as ACE inhibitors, diuretics, and beta-blockers.",
        "A heart attack occurs when the flow of blood to the heart is blocked, most often by a buildup of fat, cholesterol and other substances, which form a plaque in the arteries that feed the heart (coronary arteries).",
        "Stroke is a medical emergency that occurs when the blood supply to part of your brain is interrupted or severely reduced, depriving brain tissue of oxygen and nutrients.",
        "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment."
    ]

    qa_system = MedicalQASystem(medical_docs)

    while True:
        question = input("\nEnter your medical question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        answer, retrieved_docs = qa_system.answer_question(question)
        print(f"\nAnswer: {answer}")
        print("\nRetrieved Passages:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. {doc}")

