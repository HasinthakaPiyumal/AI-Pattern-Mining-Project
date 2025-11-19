import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

knowledge_base_documents = [
    "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose).",
    "Symptoms of type 2 diabetes include increased thirst, frequent urination, and blurred vision.",
    "Treatment for type 2 diabetes often involves lifestyle changes, medication, and insulin therapy.",
    "Hypertension, also known as high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
    "Common symptoms of hypertension include headaches, shortness of breath, or nosebleeds, but it often has no symptoms.",
    "Managing hypertension typically involves diet modifications (low sodium), regular exercise, maintaining a healthy weight, and sometimes medication.",
    "Influenza (flu) is a contagious respiratory illness caused by flu viruses. It can cause mild to severe illness, and at times can lead to death.",
    "Symptoms of the flu include fever, cough, sore throat, muscle or body aches, headaches, and fatigue.",
    "Annual flu vaccination is recommended for most people aged 6 months and older.",
    "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus.",
    "Common symptoms of COVID-19 include fever, cough, fatigue, and loss of taste or smell.",
    "Vaccination, mask-wearing, and social distancing are effective measures to prevent the spread of COVID-19.",
    "Antibiotics are medicines that fight bacterial infections in people and animals. They work by killing the bacteria or making it difficult for the bacteria to grow and multiply.",
    "Antibiotics do not work on viruses, such as those that cause colds, flu, or COVID-19.",
    "Misuse of antibiotics can lead to antibiotic resistance, making infections harder to treat.",
    "The human heart has four chambers: two atria and two ventricles. It pumps blood throughout the body.",
    "A balanced diet rich in fruits, vegetables, and whole grains is crucial for maintaining good health.",
    "Regular physical activity can reduce the risk of many chronic diseases, including heart disease and stroke.",
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
knowledge_base_embeddings = embedding_model.encode(knowledge_base_documents, convert_to_tensor=True)

llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm_model = AutoModelForCausalLM.from_pretrained("gpt2")
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_pipeline = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=llm_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def retrieve_documents(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, knowledge_base_embeddings)[0]
    top_indices = torch.topk(similarities, k=top_k).indices.tolist()
    retrieved_docs = [knowledge_base_documents[i] for i in top_indices]
    return retrieved_docs

def medical_chatbot(user_message: str):
    medical_keywords = ["diabetes", "hypertension", "flu", "covid", "antibiotic", "heart", "health", "symptoms", "treatment", "disease", "vaccination", "blood pressure", "medication", "therapy", "virus", "bacteria"]
    needs_grounding = any(keyword in user_message.lower() for keyword in medical_keywords)

    context = ""
    sources = []
    if needs_grounding:
        retrieved_docs = retrieve_documents(user_message)
        context = "\n".join(retrieved_docs)
        sources = [f"Source: {doc}" for doc in retrieved_docs]

    if context:
        prompt = f"Based on the following medical information, answer the question accurately and provide concise, factual information:\n\nMedical Information:\n{context}\n\nQuestion: {user_message}\n\nAnswer:"
    else:
        prompt = f"Answer the following question:\n\nQuestion: {user_message}\n\nAnswer:"

    response = llm_pipeline(prompt, max_new_tokens=200, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    generated_text = response[0]["generated_text"]
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    if len(generated_text.split()) > 150:
        generated_text = " ".join(generated_text.split()[:150]) + "..."

    final_response = generated_text
    if sources:
        final_response += "\n\n" + "\n".join(sources)

    return final_response

iface = gr.Interface(
    fn=medical_chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask a medical question..."),
    outputs="text",
    title="Dynamic Knowledge Grounded Medical Chatbot",
    description="This chatbot provides information by dynamically retrieving and incorporating relevant medical knowledge from a knowledge base. It aims to reduce factual inaccuracies and provide source attribution."
)

if __name__ == "__main__":
    iface.launch()