import gradio as gr
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

generator = pipeline("text-generation", model="gpt2")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def mock_web_search(query):
    mock_results = {
        "latest treatment protocols for X disease": [
            {"title": "New Guidelines for X Disease Treatment", "url": "http://mock-medical-journal.com/x-disease-treatment", "content": "Recent studies suggest a multi-drug regimen for X disease, combining therapy A with therapy B. Early intervention is crucial, and patient monitoring for side effects of therapy B is recommended. Clinical trials show a 20% improvement in patient outcomes."},
            {"title": "Understanding X Disease", "url": "http://mock-health-org.org/about-x-disease", "content": "X disease is a complex autoimmune disorder affecting millions. While there is no cure, various treatments aim to manage symptoms and slow progression. Lifestyle modifications play a significant role. New immunomodulators are under investigation."},
        ],
        "contraindications for drug Y when combined with Z condition": [
            {"title": "Drug Y Safety Profile", "url": "http://mock-pharmacy-guide.com/drug-y-safety", "content": "Drug Y is contraindicated in patients with severe hepatic impairment. Co-administration with Z condition can lead to increased risk of cardiovascular events. Dosage adjustment may be necessary for elderly patients. Consult a physician before use."},
            {"title": "Interactions of Drug Y", "url": "http://mock-medical-data.com/drug-y-interactions", "content": "Patients with Z condition should avoid Drug Y due to potential for synergistic adverse effects on cardiac function. Alternative treatments should be explored. Monitoring of blood pressure and heart rate is essential if co-administration is unavoidable."},
        ],
        "default": [
            {"title": "General Medical Information", "url": "http://mock-general-medical.com/info", "content": "Medical science is constantly evolving. Always consult with a qualified healthcare professional for diagnosis and treatment. This information is for educational purposes only. Recent advances in gene therapy show promise for various conditions. Prevention is key."}
        ]
    }
    for key, value in mock_results.items():
        if query.lower() in key.lower():
            return value
    return mock_results["default"]


def extract_and_rank_references(query, search_results, top_n=3):
    passages = []
    sources = []
    
    for result in search_results:
        content = result["content"]
        url = result["url"]
        for paragraph in content.split(". "):
            paragraph = paragraph.strip()
            if len(paragraph) > 50:
                passages.append(paragraph + ".")
                sources.append(url)
    
    if not passages:
        return [], []

    query_embedding = sentence_model.encode([query])
    passage_embeddings = sentence_model.encode(passages)
    
    similarities = cosine_similarity(query_embedding, passage_embeddings)[0]
    
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    ranked_passages = []
    ranked_sources = []
    for i in top_indices:
        ranked_passages.append(passages[i])
        ranked_sources.append(sources[i])
        
    return ranked_passages, ranked_sources

def generate_answer_with_references(query, references):
    llm_prompt = f"Based on the following information, answer the medical query:\n\nMedical Query: {query}\n\nInformation:\n"
    if references:
        for i, ref in enumerate(references):
            llm_prompt += f"[{i+1}] {ref}\n"
        llm_prompt += "\nSynthesize an answer using only the provided information and cite the references using their numbers (e.g., [1], [2]). If the information is insufficient, state that.\nAnswer:"
    else:
        llm_prompt += "No specific references found. Provide a general answer based on common medical knowledge, or state that specific information could not be retrieved.\nAnswer:"

    generated_text = generator(llm_prompt, max_new_tokens=200, num_return_sequences=1, do_sample=True, temperature=0.7)[0]["generated_text"]

    answer_start = generated_text.find("Answer:")
    if answer_start != -1:
        answer = generated_text[answer_start + len("Answer:"):].strip()
    else:
        answer = generated_text.strip()

    if references and not re.search(r"\\[\\d+\\\]", answer):
        answer += " (Refer to provided references for details)."
    
    return answer

def medical_verifier_app(query: str):
    search_results = mock_web_search(query)
    
    ranked_passages, ranked_sources = extract_and_rank_references(query, search_results, top_n=5)
    
    answer = generate_answer_with_references(query, ranked_passages)
    
    formatted_references = ""
    if ranked_passages:
        formatted_references += "<h3>References:</h3><ul>"
        for i, (passage, source) in enumerate(zip(ranked_passages, ranked_sources)):
            formatted_references += f"<li>[{i+1}] \"{passage}\" (Source: <a href=\"{source}\" target=\"_blank\">{source}</a>)</li>"
        formatted_references += "</ul>"
    else:
        formatted_references = "<p>No specific references could be retrieved for this query.</p>"
        
    final_output = f"<h2>Answer:</h2><p>{answer}</p>{formatted_references}"
    
    return final_output


iface = gr.Interface(
    fn=medical_verifier_app,
    inputs=gr.Textbox(lines=5, label="Enter your medical query here:"),
    outputs=gr.HTML(label="Medical Information Verifier Output"),
    title="Medical Information Verifier AI",
    description="Ask a medical question and get an AI-generated answer with verifiable references from simulated medical sources."
)

if __name__ == "__main__":
    iface.launch()