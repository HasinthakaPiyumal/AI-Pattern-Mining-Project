import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. Knowledge Base Setup
medical_texts = [
    "Symptoms of influenza include fever, cough, sore throat, and body aches. Rest and fluids are recommended.",
    "Common cold symptoms are milder than flu, usually runny nose, sneezing, and mild sore throat. No fever typically.",
    "Pneumonia can cause severe cough, shortness of breath, chest pain, and high fever. Antibiotics are often needed.",
    "Migraine headaches are characterized by severe throbbing pain, sensitivity to light/sound, and nausea. Triptans can help.",
    "Tension headaches usually cause a dull, aching pain around the head. Stress and muscle tension are common triggers.",
    "Diabetes symptoms include frequent urination, increased thirst, unexplained weight loss, and fatigue. Insulin or oral medications manage blood sugar.",
    "Hypertension (high blood pressure) often has no symptoms but can lead to heart disease. Lifestyle changes and medication are key.",
    "Asthma causes wheezing, coughing, chest tightness, and shortness of breath. Inhalers are used for quick relief and long-term control.",
    "Allergies manifest as sneezing, itchy eyes, runny nose, or skin rashes. Antihistamines are common treatments.",
    "Strep throat presents with a sudden sore throat, pain when swallowing, fever, and sometimes white patches on tonsils. Antibiotics are prescribed."
]

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for medical texts
text_embeddings = embedding_model.encode(medical_texts)

# Build FAISS index
dimension = text_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension) # Using Inner Product for similarity
faiss_index.add(text_embeddings)

# 2. Retrieval Module
def retrieve_documents(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)

    retrieved_docs = []
    for i, idx in enumerate(indices[0]):
        # FAISS returns cosine distance, convert to similarity (1 - distance) for weighting
        # or directly use inner product if normalized embeddings, here we use IP which is cosine sim if normalized
        similarity = distances[0][i]
        retrieved_docs.append({
            "text": medical_texts[idx],
            "relevance_score": float(similarity) # P(d_i | Q)
        })
    return retrieved_docs

# 3. Generative Model Simulation
def simulate_generative_model(query: str, document_context: str):
    # This is a highly simplified simulation. In a real system, this would be an LLM call.
    # It returns a dummy answer and a 'generation probability' based on context relevance.
    
    # A very basic heuristic: if query terms are in the context, higher confidence.
    query_terms = set(query.lower().split())
    context_terms = set(document_context.lower().split())
    overlap = len(query_terms.intersection(context_terms))
    
    # Simulate a generated answer based on the context
    if "flu" in document_context.lower() and "fever" in query.lower():
        generated_answer = f"Based on symptoms like fever and context about flu: Consider influenza. Recommend rest and fluids."
        generation_probability = 0.8 + (overlap * 0.05) # Higher confidence
    elif "cold" in document_context.lower() and "runny nose" in query.lower():
        generated_answer = f"Based on symptoms like runny nose and context about common cold: Likely common cold. Recommend symptomatic relief."
        generation_probability = 0.75 + (overlap * 0.05)
    elif "pneumonia" in document_context.lower() and "shortness of breath" in query.lower():
        generated_answer = f"Based on symptoms like shortness of breath and context about pneumonia: Investigate for pneumonia. Antibiotics may be needed."
        generation_probability = 0.9 + (overlap * 0.05)
    elif "headache" in query.lower() and "migraine" in document_context.lower():
        generated_answer = f"Based on headache and migraine context: Suggest migraine. Consider triptans and rest."
        generation_probability = 0.85 + (overlap * 0.05)
    elif "headache" in query.lower() and "tension" in document_context.lower():
        generated_answer = f"Based on headache and tension context: Suggest tension headache. Stress reduction and pain relief."
        generation_probability = 0.7 + (overlap * 0.05)
    elif "diabetes" in document_context.lower() and "thirst" in query.lower():
        generated_answer = f"Based on increased thirst and diabetes context: Evaluate for diabetes. Blood sugar management is crucial."
        generation_probability = 0.9
    elif "asthma" in document_context.lower() and "wheezing" in query.lower():
        generated_answer = f"Based on wheezing and asthma context: Consider asthma exacerbation. Bronchodilators and corticosteroids."
        generation_probability = 0.92
    else:
        generated_answer = f"Synthesizing information from context: '{document_context}'. Patient query: '{query}'. Further investigation may be needed."
        generation_probability = 0.5 + (overlap * 0.02) # Default confidence
    
    return {
        "generated_text": generated_answer,
        "generation_probability": min(1.0, generation_probability) # Cap at 1.0
    }

# 4. Marginalization Core Logic
def marginalize_and_recommend(patient_query: str, top_k_docs: int = 3):
    retrieved_docs = retrieve_documents(patient_query, top_k_docs)

    # Normalize relevance scores to sum to 1, if they are not already probabilities
    # FAISS inner product on normalized vectors already gives cosine similarity, which can be treated as P(d|Q) directly here.
    # However, let's ensure they are positive and sum to 1 for proper marginalization if using as weights.
    
    # Ensure relevance scores are positive and rescale for marginalization
    # The FAISS inner product scores can be negative if vectors point in opposite directions.
    # For P(d|Q) we ideally want a non-negative value. Let's shift and normalize.
    scores = np.array([doc["relevance_score"] for doc in retrieved_docs])
    min_score = np.min(scores)
    if min_score < 0:
        scores = scores - min_score # Shift to make all scores non-negative
    
    sum_scores = np.sum(scores)
    if sum_scores == 0:
        # Fallback if all scores are zero or negative after shifting (unlikely with IP on normalized vectors)
        document_weights = np.ones(len(retrieved_docs)) / len(retrieved_docs)
    else:
        document_weights = scores / sum_scores # P(d_i | Q) normalized
    
    # Store results for each document
    document_results = []
    for i, doc in enumerate(retrieved_docs):
        simulated_output = simulate_generative_model(patient_query, doc["text"])
        
        document_results.append({
            "document_text": doc["text"],
            "relevance_score": doc["relevance_score"],
            "normalized_weight": document_weights[i],
            "generated_answer_cond_d": simulated_output["generated_text"],
            "generation_prob_cond_d": simulated_output["generation_probability"]
        })

    # Perform marginalization (simplified combination for text generation and overall score)
    # For text, we'll concatenate weighted answers or pick the highest-weighted answer's generation.
    # For overall confidence, we sum P(Y|Q,d_i) * P(d_i|Q)
    
    marginalized_prob_sum = 0.0
    combined_recommendation_parts = []

    for i, res in enumerate(document_results):
        marginalized_prob_sum += res["generation_prob_cond_d"] * res["normalized_weight"]
        combined_recommendation_parts.append(
            f"[{i+1}] (Weight: {res['normalized_weight']:.2f}) {res['generated_answer_cond_d']}"
        )

    # Choose a simple way to present the combined recommendation
    # For this demo, we'll list the weighted individual recommendations.
    # In a real system, the LLM would synthesize a final answer based on all contexts.
    final_recommendation = "\n-- Individual Document-Weighted Insights --\n" + "\n".join(combined_recommendation_parts)
    final_recommendation += f"\n\nOverall Marginalized Confidence: {marginalized_prob_sum:.2f}"

    return final_recommendation

# 5. Gradio User Interface
if __name__ == "__main__":
    interface = gr.Interface(
        fn=marginalize_and_recommend,
        inputs=[
            gr.Textbox(label="Patient Symptoms/Medical History (Query)", placeholder="e.g., patient has a severe cough, shortness of breath, and high fever")
        ],
        outputs=gr.Markdown(label="Clinical Recommendation"),
        title="Clinical Decision Support System (CDSS) with Marginalization",
        description="Enter patient symptoms and medical history to get a synthesized clinical recommendation by marginalizing over multiple retrieved medical documents. This system combines information from various sources to provide robust advice."
    )

    interface.launch(share=True)