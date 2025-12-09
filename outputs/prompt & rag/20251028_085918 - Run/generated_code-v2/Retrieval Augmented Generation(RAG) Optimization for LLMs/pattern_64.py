import torch
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np

class ClinicalQnAAssistant:
    def __init__(self, doc_encoder_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 query_encoder_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 generator_model_name="facebook/bart-large-cnn"):

        self.document_encoder = SentenceTransformer(doc_encoder_model_name)
        self.query_encoder = SentenceTransformer(query_encoder_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

        self.document_index = None
        self.documents = []

        # Freeze the document encoder for the fixed index pattern
        for param in self.document_encoder.parameters():
            param.requires_grad = False

    def build_document_index(self, documents):
        self.documents = documents
        print("Encoding documents...")
        document_embeddings = self.document_encoder.encode(documents, convert_to_tensor=False)
        print("Building FAISS index...")
        self.document_index = faiss.IndexFlatL2(document_embeddings.shape[1])
        self.document_index.add(document_embeddings)
        print(f"FAISS index built with {self.document_index.ntotal} documents.")

    def retrieve_documents(self, query, k=3):
        if self.document_index is None:
            raise ValueError("Document index has not been built. Call build_document_index first.")
        
        query_embedding = self.query_encoder.encode([query], convert_to_tensor=False)
        distances, indices = self.document_index.search(query_embedding, k)
        
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs

    def generate_answer(self, query, retrieved_docs):
        context = " ".join(retrieved_docs)
        prompt = f"Based on the following medical context, answer the question.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        inputs = self.tokenizer([prompt], max_length=1024, return_tensors="pt", truncation=True)
        output_sequences = self.generator.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        answer = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return answer

    def answer_query(self, query, k=3):
        retrieved_docs = self.retrieve_documents(query, k)
        answer = self.generate_answer(query, retrieved_docs)
        return answer, retrieved_docs

def conceptual_fine_tune(assistant, fine_tuning_data):
    print("\n--- Starting Conceptual Fine-tuning Phase ---")
    
    # Verify document encoder is frozen
    for name, param in assistant.document_encoder.named_parameters():
        if param.requires_grad:
            print(f"WARNING: Document encoder parameter {name} is NOT frozen.")
    print("Document encoder parameters are confirmed frozen.")

    # Enable gradients for query encoder and generator (they are by default if not explicitly frozen)
    for param in assistant.query_encoder.parameters():
        param.requires_grad = True
    for param in assistant.generator.parameters():
        param.requires_grad = True

    # In a real scenario, you would set up an optimizer, loss function, and training loop here.
    # This is a placeholder to demonstrate the fine-tuning scope.
    print(f"Simulating fine-tuning with {len(fine_tuning_data)} data points.")
    print("Query encoder and Generator are now set to be fine-tuned.")
    print("Document encoder and Document index remain fixed.")
    print("--- Conceptual Fine-tuning Phase Complete ---")

if __name__ == "__main__":
    # 1. Dummy Document Corpus
    medical_documents = [
        "Insulin resistance is a pathological condition in which cells fail to respond normally to the hormone insulin. This can lead to high blood sugar.",
        "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). It's more common in adults.",
        "Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated.",
        "A myocardial infarction, commonly known as a heart attack, occurs when blood flow to a part of the heart is blocked for a long enough time.",
        "Common symptoms of influenza (flu) include fever, cough, sore throat, muscle aches, and fatigue. It is caused by the influenza virus.",
        "The COVID-19 pandemic is an ongoing global pandemic of coronavirus disease 2019 caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).",
        "Pneumonia is an inflammatory condition of the lung primarily affecting the small air sacs known as alveoli. It is usually caused by infection with viruses or bacteria.",
        "Antibiotics are a type of antimicrobial drug used in the treatment and prevention of bacterial infections. They are ineffective against viruses.",
        "Vaccines stimulate the body's immune system to protect against infection or disease. They are crucial for public health.",
        "Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body."
    ]

    # Initialize the Assistant
    assistant = ClinicalQnAAssistant()

    # 2. Pre-computation Phase: Build Document Index (Fixed)
    assistant.build_document_index(medical_documents)

    # 3. Inference Phase (Before Fine-tuning)
    print("\n--- Inference Before Fine-tuning ---")
    query1 = "What are the symptoms of the flu?"
    answer1, docs1 = assistant.answer_query(query1)
    print(f"Query: {query1}")
    print(f"Retrieved Docs: {docs1}")
    print(f"Answer: {answer1}")

    query2 = "What is insulin resistance?"
    answer2, docs2 = assistant.answer_query(query2)
    print(f"\nQuery: {query2}")
    print(f"Retrieved Docs: {docs2}")
    print(f"Answer: {answer2}")

    # 4. Conceptual Fine-tuning Phase
    # In a real application, fine_tuning_data would be a dataset of (question, relevant_context, ideal_answer) tuples.
    dummy_fine_tuning_data = [
        {"question": "What causes a heart attack?", "relevant_context": "blocked blood flow to the heart", "ideal_answer": "A heart attack is caused by blocked blood flow to a part of the heart."},
        {"question": "What is hypertension?", "relevant_context": "persistently elevated blood pressure", "ideal_answer": "Hypertension is a medical condition where blood pressure in the arteries is persistently elevated."},
    ]
    conceptual_fine_tune(assistant, dummy_fine_tuning_data)

    # 5. Inference Phase (After Conceptual Fine-tuning)
    print("\n--- Inference After Conceptual Fine-tuning ---")
    query3 = "What leads to high blood pressure?"
    answer3, docs3 = assistant.answer_query(query3)
    print(f"Query: {query3}")
    print(f"Retrieved Docs: {docs3}")
    print(f"Answer: {answer3}")

    query4 = "Tell me about diabetes."
    answer4, docs4 = assistant.answer_query(query4)
    print(f"\nQuery: {query4}")
    print(f"Retrieved Docs: {docs4}")
    print(f"Answer: {answer4}")