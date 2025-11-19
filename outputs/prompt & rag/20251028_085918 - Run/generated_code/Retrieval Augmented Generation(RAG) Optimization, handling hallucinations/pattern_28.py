import gradio as gr
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from collections import defaultdict
import random
import time

# --- 1. Load NLP model for Query Processor ---
nlp = spacy.load("en_core_web_sm")

# --- 2. Initialize Sentence Transformer for embeddings ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 3. Mock Knowledge Bases (for demonstration) ---
medical_documents = [
    "Symptoms of influenza include fever, cough, sore throat, and body aches. Treatment often involves rest and fluids.",
    "Diabetes mellitus is a metabolic disease that causes high blood sugar. Type 1 diabetes is an autoimmune reaction.",
    "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. Lifestyle changes and medication are common treatments.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, and chills.",
    "Migraine is a severe headache often accompanied by nausea, vomiting, and extreme sensitivity to light and sound.",
    "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon. Symptoms include sudden pain that begins on the right side of the lower abdomen.",
    "Common cold symptoms are generally milder than flu symptoms and include a runny or stuffy nose, sore throat, and cough.",
    "Asthma is a condition in which your airways narrow and swell and may produce extra mucus, making breathing difficult.",
    "Dengue fever is a mosquito-borne tropical disease caused by the dengue virus. Symptoms typically begin three to fourteen days after infection.",
    "Malaria is a mosquito-borne infectious disease affecting humans and other animals caused by parasitic protozoans.",
    "Allergies occur when your immune system reacts to a foreign substance like pollen, bee venom, or pet dander.",
    "Bronchitis is an inflammation of the lining of your bronchial tubes, which carry air to and from your lungs. Often causes a cough with mucus.",
    "Gastroenteritis, also known as stomach flu, is an inflammation of the stomach and intestines, typically due to viral or bacterial infection."
]

patient_histories = [
    {"id": "P001", "history": "Patient P001 has a history of seasonal allergies and mild asthma. No known chronic conditions.", "conditions": ["seasonal allergies", "asthma"]},
    {"id": "P002", "history": "Patient P002, 55 years old, has hypertension controlled with medication. Reports occasional headaches.", "conditions": ["hypertension", "headaches"]},
    {"id": "P003", "history": "Patient P003 is a 30-year-old female with no significant medical history. Recently traveled to a tropical region.", "conditions": []}
]

# --- 4. ChromaDB Knowledge Base Setup ---
client = chromadb.Client()
medical_collection = client.get_or_create_collection(name="medical_docs")

# Add documents to ChromaDB
if medical_collection.count() == 0:
    doc_ids = [f"doc_{i}" for i in range(len(medical_documents))]
    medical_collection.add(
        documents=medical_documents,
        embeddings=embedding_model.encode(medical_documents).tolist(),
        ids=doc_ids
    )
    print(f"Added {len(medical_documents)} documents to ChromaDB.")

# --- 5. Query Processor ---
class QueryProcessor:
    def process_query(self, query: str):
        doc = nlp(query.lower())
        entities = [ent.text for ent in doc.ents]
        keywords = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        # Simple heuristic for complexity
        complexity = "complex" if len(keywords) > 5 or len(entities) > 1 else "simple"
        
        return {"original_query": query, "keywords": keywords, "entities": entities, "complexity": complexity}

# --- 6. Adaptive Retriever ---
class AdaptiveRetriever:
    def __init__(self, medical_collection, patient_histories, embedding_model):
        self.medical_collection = medical_collection
        self.patient_histories = patient_histories
        self.embedding_model = embedding_model

    def retrieve(self, processed_query: dict, patient_id: str = None, iteration: int = 0):
        query_text = processed_query["original_query"]
        query_complexity = processed_query["complexity"]
        
        retrieved_contexts = []

        # Dynamic Strategy Selection
        if query_complexity == "simple":
            # Semantic search for general medical knowledge
            results = self.medical_collection.query(
                query_embeddings=self.embedding_model.encode([query_text]).tolist(),
                n_results=3
            )
            retrieved_contexts.extend(results['documents'][0])
        else: # complex query
            # Hybrid approach: more semantic search + keyword matching for patient history
            results_semantic = self.medical_collection.query(
                query_embeddings=self.embedding_model.encode([query_text]).tolist(),
                n_results=5
            )
            retrieved_contexts.extend(results_semantic['documents'][0])

            # Integrate patient history if available
            if patient_id:
                for patient in self.patient_histories:
                    if patient["id"] == patient_id:
                        retrieved_contexts.append(f"Patient History for {patient_id}: {patient['history']}")
                        # Add patient's known conditions as keywords for further retrieval if complex
                        if query_complexity == "complex":
                            for condition in patient['conditions']:
                                keyword_results = self.medical_collection.query(
                                    query_texts=[condition],
                                    n_results=2
                                )
                                retrieved_contexts.extend(keyword_results['documents'][0])
                        break
        
        # Simulate Context Re-ranking (simple deduplication and truncation for demo)
        unique_contexts = list(dict.fromkeys(retrieved_contexts))
        final_contexts = "\n".join(unique_contexts[:5]) # Limit to top 5 unique contexts
        
        # Simulate Iterative Refinement based on iteration count
        if iteration > 0:
            final_contexts += f"\n(Refined context from iteration {iteration})"
            # In a real system, this would involve re-querying or expanding search based on LLM feedback

        return final_contexts

# --- 7. Language Model (LLM) Integration ---
# Placeholder for a real LLM call (e.g., OpenAI API, local Llama model)
class LLMIntegration:
    def call_llm(self, prompt: str, context: str):
        full_prompt = f"Given the following medical context and patient information:\n\nContext:\n{context}\n\nBased on this, please provide a diagnostic hypothesis or answer the query:\nQuery: {prompt}\n\nDiagnostic Hypothesis:"
        
        # Simulate LLM response delay
        time.sleep(1.5)
        
        # Simple mock LLM responses based on keywords
        if "influenza" in context.lower() or "fever" in prompt.lower() and "cough" in prompt.lower():
            return "Considering the symptoms and context, a possible diagnostic hypothesis is influenza (flu)."
        elif "hypertension" in context.lower() or "blood pressure" in prompt.lower():
            return "Based on the provided information, hypertension should be considered, and its management evaluated."
        elif "diabetes" in context.lower() or "blood sugar" in prompt.lower():
            return "The information suggests evaluating for diabetes mellitus."
        elif "pneumonia" in context.lower() or "lung" in context.lower() and "fluid" in context.lower():
            return "Pneumonia is a strong possibility given the description."
        elif "appendicitis" in context.lower() or "lower abdomen pain" in prompt.lower():
            return "Acute appendicitis should be ruled out, further investigation is recommended."
        elif "migraine" in context.lower() and "headache" in prompt.lower():
            return "The symptoms are highly suggestive of a migraine."
        elif "dengue" in context.lower() or "tropical region" in context.lower() and "fever" in prompt.lower():
            return "Given the travel history and symptoms, dengue fever is a consideration."
        elif "asthma" in context.lower() or "difficulty breathing" in prompt.lower():
            return "Asthma exacerbation is a possible diagnosis."
        elif "allergies" in context.lower() or "pollen" in context.lower():
            return "Symptoms appear consistent with an allergic reaction."
        else:
            return f"Based on the provided context and query, here is a general diagnostic insight: {full_prompt[:150]}... (This is a simulated LLM response)"

# --- 8. Self-Reflection and Confidence Module ---
class SelfReflectionModule:
    def assess_confidence(self, llm_response: str, retrieved_context: str):
        # Simple heuristic: if response is short or generic, confidence is lower.
        # In a real system, this would involve another smaller LLM, keyword analysis, or specific confidence scores from the primary LLM.
        if len(llm_response.split()) < 15 or "general diagnostic insight" in llm_response.lower():
            return 0.4 # Low confidence
        if "possible diagnostic hypothesis" in llm_response.lower() or "should be considered" in llm_response.lower():
            return 0.7 # Medium confidence
        return 0.9 # High confidence

    def decide_action(self, confidence_score: float, iteration: int):
        if confidence_score < 0.6 and iteration < 2: # Allow up to 2 iterations for re-retrieval
            return "retrieve_more" # Not confident, try to retrieve more information
        elif confidence_score >= 0.6:
            return "generate_answer" # Confident enough to provide an answer
        else:
            return "abstain" # Still not confident after iterations, abstain or ask for more info from user

# --- Main Application Logic (Orchestrator) ---
query_processor = QueryProcessor()
adaptive_retriever = AdaptiveRetriever(medical_collection, patient_histories, embedding_model)
llm_integration = LLMIntegration()
reflection_module = SelfReflectionModule()

def diagnose(query: str, patient_id: str = None):
    processed_query = query_processor.process_query(query)
    current_iteration = 0
    
    all_retrieved_contexts = []
    final_diagnosis = ""
    action = "retrieve_more"

    while action == "retrieve_more" and current_iteration < 3: # Max 3 retrieval iterations
        retrieved_context = adaptive_retriever.retrieve(processed_query, patient_id, current_iteration)
        all_retrieved_contexts.append(f"--- Iteration {current_iteration + 1} Retrieval ---\n" + retrieved_context)
        
        # Combine all contexts for LLM for this iteration
        combined_context_for_llm = "\n\n".join(all_retrieved_contexts)
        
        llm_response = llm_integration.call_llm(processed_query["original_query"], combined_context_for_llm)
        confidence = reflection_module.assess_confidence(llm_response, combined_context_for_llm)
        action = reflection_module.decide_action(confidence, current_iteration)
        
        if action == "generate_answer":
            final_diagnosis = llm_response
            break
        elif action == "abstain":
            final_diagnosis = "The system is unable to provide a confident diagnosis based on the current information. Please provide more details or consult a human expert."
            break
        
        current_iteration += 1
        if current_iteration == 3 and action == "retrieve_more":
            final_diagnosis = "After multiple retrieval attempts, the system cannot generate a sufficiently confident diagnosis. It suggests further investigation or expert consultation."

    if not final_diagnosis and all_retrieved_contexts: # If loop finished without breaking and no final_diagnosis set, use last LLM response
        final_diagnosis = llm_response # Fallback to the last LLM response

    return final_diagnosis, "\n\n".join(all_retrieved_contexts)

# --- Gradio UI ---
interface = gr.Interface(
    fn=diagnose,
    inputs=[
        gr.Textbox(label="Patient Symptoms/Query", placeholder="e.g., 'Patient has a high fever, cough, and body aches.' or 'What are the treatments for hypertension?'"),
        gr.Dropdown(label="Select Patient ID (Optional)", choices=[p['id'] for p in patient_histories] + [None], value=None)
    ],
    outputs=[
        gr.Textbox(label="Diagnostic Hypothesis/Answer"),
        gr.Textbox(label="Retrieved Medical Contexts", show_copy_button=True)
    ],
    title="Medical Diagnostic Assistant with Adaptive RAG",
    description="An AI assistant leveraging Adaptive RAG to provide diagnostic hypotheses and answers based on dynamic knowledge retrieval. Select a patient ID to integrate their history.",
    examples=[
        ["Patient P001 reports sneezing and watery eyes, especially in spring.", "P001"],
        ["Patient P002 has elevated blood pressure readings and occasional dizziness.", "P002"],
        ["What are the symptoms of pneumonia?", None],
        ["A 30-year-old female recently returned from Thailand with a sudden onset of high fever and muscle pain.", "P003"],
        ["Severe headache with nausea and sensitivity to light.", None]
    ]
)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    interface.launch()
