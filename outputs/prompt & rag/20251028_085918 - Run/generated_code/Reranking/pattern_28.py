import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
import gradio as gr
import os

# --- Configuration ---
LM_MODEL_NAME = "distilgpt2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./medical_knowledge_db"
TOP_K_RETRIEVAL = 5
TOP_K_RERANK = 3

# --- Initialize Models and DB ---
try:
    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm_model = AutoModelForCausalLM.from_pretrained(LM_MODEL_NAME)
    text_generator = pipeline(
        "text-generation",
        model=lm_model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        truncation=True
    )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection_name = "medical_docs"
    try:
        medical_collection = client.get_collection(
            name=collection_name,
            embedding_function=lambda texts: embedding_model.encode(texts).tolist()
        )
    except Exception:
        medical_collection = client.create_collection(
            name=collection_name,
            embedding_function=lambda texts: embedding_model.encode(texts).tolist()
        )

    if medical_collection.count() == 0:
        dummy_medical_docs = [
            {"id": "doc1", "content": "COVID-19 symptoms often include fever, cough, fatigue, and loss of taste or smell. Serious cases can lead to pneumonia and acute respiratory distress syndrome (ARDS)."},
            {"id": "doc2", "content": "Diabetes mellitus is a chronic metabolic disease characterized by high blood glucose levels. Type 1 diabetes is an autoimmune condition, while type 2 diabetes often results from insulin resistance."},
            {"id": "doc3", "content": "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. Lifestyle changes like diet and exercise, along with medication, are common treatments."},
            {"id": "doc4", "content": "The liver is a vital organ that performs many functions, including detoxification, protein synthesis, and production of biochemicals necessary for digestion. Liver diseases include hepatitis, cirrhosis, and fatty liver disease."},
            {"id": "doc5", "content": "A myocardial infarction, commonly known as a heart attack, occurs when blood flow to a part of the heart is blocked, usually by a blood clot. Symptoms include chest pain, shortness of breath, and discomfort in other areas of the upper body."},
            {"id": "doc6", "content": "Antibiotics are medications that fight bacterial infections. They work by killing bacteria or slowing their growth. It\'s crucial to complete the full course of antibiotics to prevent antibiotic resistance."},
            {"id": "doc7", "content": "Vaccines stimulate the immune system to produce antibodies, providing immunity against infectious diseases. They are a safe and effective way to prevent widespread outbreaks."},
            {"id": "doc8", "content": "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, coughing, chest tightness, and shortness of breath."},
            {"id": "doc9", "content": "Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body. Treatments include surgery, chemotherapy, radiation therapy, and immunotherapy."},
            {"id": "doc10", "content": "Stroke occurs when blood supply to part of the brain is interrupted or reduced, depriving brain tissue of oxygen and nutrients. It is a medical emergency that requires prompt treatment."},
        ]
        doc_ids = [doc["id"] for doc in dummy_medical_docs]
        contents = [doc["content"] for doc in dummy_medical_docs]
        medical_collection.add(documents=contents, ids=doc_ids)
        print("Medical knowledge base populated with dummy data.")

except Exception as e:
    print(f"Error initializing models or database: {e}")

# --- Functions for Dynamic Knowledge Grounding ---

def retrieve_documents(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    try:
        results = medical_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'distances']
        )
        retrieved_docs = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    "content": results['documents'][0][i],
                    "distance": results['distances'][0][i]
                })
        return retrieved_docs
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        return []

def rerank_documents_zero_shot(query: str, documents: list[dict], top_k: int = TOP_K_RERANK) -> list[str]:
    if not documents:
        return []
    sorted_docs = sorted(documents, key=lambda x: x.get('distance', float('inf')))
    return [doc['content'] for doc in sorted_docs[:top_k]]

def ground_lm_and_generate_response(query: str, relevant_docs: list[str]) -> str:
    context = ""
    if relevant_docs:
        context = "\n\nRefer to the following medical information:\n"
        for i, doc in enumerate(relevant_docs):
            context += f"Document {i+1}: {doc}\n"
        context += "\n"

    prompt = f"{context}Clinician's Question: {query}\n\nAnswer the question based on the provided medical information. If the information is insufficient, state that."

    try:
        response = text_generator(prompt, max_new_tokens=300, num_return_sequences=1, do_sample=True, temperature=0.7)[0]['generated_text']
        answer_prefix = "Answer the question based on the provided medical information. If the information is insufficient, state that."
        if answer_prefix in response:
            answer = response.split(answer_prefix, 1)[1].strip()
            if "Clinician's Question:" in answer:
                answer = answer.split("Clinician's Question:", 1)[0].strip()
            return answer
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# --- Main Application Logic ---

def medical_qa_system(query: str) -> str:
    retrieved_docs = retrieve_documents(query)
    reranked_content = rerank_documents_zero_shot(query, retrieved_docs)
    final_answer = ground_lm_and_generate_response(query, reranked_content)

    sources = "\n\nSources:\n"
    if reranked_content:
        for i, doc_text in enumerate(reranked_content):
            sources += f"- Document {i+1}: {doc_text[:100]}...\n"
    else:
        sources += "- No specific documents were used for grounding."

    return final_answer + sources

# --- Gradio Interface ---

if __name__ == "__main__":
    if 'text_generator' not in globals() or text_generator is None:
        print("Models or database not initialized. Please check error messages above.")
        def dummy_qa(query):
            return "System is not initialized. Cannot answer questions."
        iface = gr.Interface(
            fn=dummy_qa,
            inputs=gr.Textbox(lines=2, label="Enter your medical question"),
            outputs=gr.Textbox(label="AI Answer"),
            title="AI-Powered Medical Q&A for Clinicians (Initialization Failed)",
            description="The AI system could not initialize. Please check logs."
        )
    else:
        iface = gr.Interface(
            fn=medical_qa_system,
            inputs=gr.Textbox(lines=2, label="Enter your medical question (e.g., 'What are the symptoms of COVID-19?')"),
            outputs=gr.Textbox(label="AI-Powered Answer"),
            title="AI-Powered Medical Q&A for Clinicians",
            description="Get accurate, evidence-based medical information by querying our AI system, which dynamically grounds its answers with relevant medical documents."
        )
    iface.launch(share=False)