import streamlit as st
import random
from typing import List, Dict, Any

# --- Mock LLM and RAG Components (to avoid external API keys for a self-contained example) ---
class MockEmbeddings:
    """A mock embedding class."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Return dummy embeddings for demonstration
        return [[random.random() for _ in range(1536)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        # Return a dummy embedding for queries
        return [random.random() for _ in range(1536)]

class MockVectorStore:
    """A simple in-memory mock vector store for demonstration."""
    def __init__(self, texts: List[str], embeddings: MockEmbeddings):
        self.texts = texts
        self.embeddings = embeddings
        self.vector_index = {i: embeddings.embed_documents([text])[0] for i, text in enumerate(texts)}

    def similarity_search(self, query: str, k: int = 2) -> List[str]:
        query_vec = self.embeddings.embed_query(query)
        # In a real scenario, this would be a proper similarity search.
        # For mock, we'll just return random relevant-ish texts or the first k.
        st.info(f"[Mock RAG]: Searching knowledge base for '{query}'...")
        # Simple mock: just return a few relevant-looking entries based on keywords or random
        results = []
        query_lower = query.lower()
        for text in self.texts:
            if any(keyword in text.lower() for keyword in query_lower.split() if len(keyword) > 3):
                results.append(text)
                if len(results) >= k: # Limit results
                    break
        if not results: # Fallback if no keyword match
            results = random.sample(self.texts, min(k, len(self.texts)))
        return results

class MockLLM:
    """A mock LLM that simulates Chain-of-Thought reasoning for medical diagnostics."""
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature # Not used in mock but for API consistency

    def invoke(self, prompt: str) -> str:
        return self._generate_response(prompt)

    def _generate_response(self, prompt: str) -> str:
        st.info(f"[Mock LLM]: Processing prompt... (Truncated prompt: {prompt[:100]}...)")
        
        if "patient symptoms" in prompt.lower() and "medical history" in prompt.lower():
            # Simulate a multi-step diagnostic process
            diagnosis_steps = [
                "Step 1: Analyze primary symptoms and patient history.",
                "Step 2: Consider common conditions associated with these symptoms.",
                "Step 3: Evaluate test results for abnormalities and correlations.",
                "Step 4: Formulate differential diagnoses.",
                "Step 5: Verify against medical knowledge and refine."
            ]
            
            if "sore throat" in prompt.lower() and "fever" in prompt.lower():
                possible_condition = "Streptococcal Pharyngitis (Strep Throat)"
                reasoning = "Based on acute onset of sore throat, fever, and absence of cough, Strep Throat is a primary consideration. Need to verify with rapid strep test results."
                confidence = "High"
            elif "chest pain" in prompt.lower() and "shortness of breath" in prompt.lower():
                possible_condition = "Myocardial Infarction (Heart Attack)"
                reasoning = "Acute chest pain radiating to the arm, accompanied by shortness of breath, suggests cardiac event. Elevated troponin levels and ECG changes would confirm. This requires urgent medical attention."
                confidence = "Urgent - High"
            elif "fatigue" in prompt.lower() and "joint pain" in prompt.lower():
                possible_condition = "Rheumatoid Arthritis or Chronic Fatigue Syndrome"
                reasoning = "Chronic fatigue and generalized joint pain are non-specific and could indicate various autoimmune conditions or syndromes. Requires further immunological markers and specialist consultation."
                confidence = "Medium - Requires further investigation"
            else:
                possible_condition = "Unspecified Condition"
                reasoning = "The provided symptoms are broad. Further specific details are needed for a more precise diagnosis. Consider a wide range of possibilities from viral infections to chronic diseases."
                confidence = "Low - Requires more data"

            return (f"Reasoning Steps:\n" +
                    "\n".join(diagnosis_steps) +
                    f"\n\nPotential Diagnosis: {possible_condition}\n" +
                    f"Reasoning: {reasoning}\n" +
                    f"Confidence: {confidence}")
        elif "verify" in prompt.lower() and "diagnosis" in prompt.lower():
            return "Verification complete. Diagnosis appears consistent with available medical knowledge. No major inconsistencies found." if random.random() > 0.1 else "Verification found minor inconsistencies. Consider re-evaluating diagnostic path."
        else:
            return f"Mock LLM response to: {prompt[:50]}..."

# --- Knowledge Base Setup ---
MEDICAL_KNOWLEDGE_BASE = [
    "Streptococcal Pharyngitis (Strep Throat) symptoms include sudden sore throat, pain with swallowing, fever, red spots on the roof of the mouth, swollen tonsils, and sometimes a rash. Cough is typically absent.",
    "Myocardial Infarction (Heart Attack) presents with chest pain often radiating to the left arm, shortness of breath, sweating, nausea, and lightheadedness. Elevated cardiac biomarkers (troponin) and ECG changes are diagnostic.",
    "Rheumatoid Arthritis is a chronic inflammatory disorder affecting joints, causing pain, swelling, stiffness, and fatigue. It is an autoimmune disease.",
    "Chronic Fatigue Syndrome (ME/CFS) is characterized by extreme fatigue that isn't improved by rest and can worsen with physical or mental activity.",
    "Common cold symptoms include runny nose, sneezing, sore throat, and cough. Fever is usually mild or absent.",
    "Influenza (Flu) symptoms are similar to common cold but typically more severe, including high fever, body aches, fatigue, and headache.",
    "Diabetes Mellitus symptoms include frequent urination, increased thirst, unexplained weight loss, fatigue, and blurred vision.",
    "Hypertension (High Blood Pressure) often has no symptoms until it causes severe complications.",
    "Asthma is a chronic lung disease that inflames and narrows the airways, causing wheezing, shortness of breath, chest tightness, and coughing.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing."
]

# Initialize mock components
mock_embeddings = MockEmbeddings()
mock_vector_store = MockVectorStore(MEDICAL_KNOWLEDGE_BASE, mock_embeddings)
mock_llm = MockLLM()

# --- LangChain-like Orchestration (simplified for demo) ---
def run_diagnostic_chain(
    patient_data: Dict[str, Any],
    num_reasoning_paths: int = 3
) -> List[Dict[str, Any]]:
    
    st.subheader("Running Diagnostic Chain...")
    diagnoses_with_reasoning = []

    for i in range(num_reasoning_paths):
        st.markdown(f"### Reasoning Path {i+1}")
        
        # 1. Chain-of-Thought (CoT) Prompting
        cot_prompt = f"""Act as a highly experienced medical diagnostician. Based on the following patient data, provide a step-by-step reasoning process leading to a potential diagnosis and your confidence level. Focus on breaking down the problem, considering differential diagnoses, and justifying your conclusions. If you need more information, state it.

        Patient Symptoms: {patient_data['symptoms']}
        Medical History: {patient_data['history']}
        Lab Results: {patient_data['lab_results']}
        Imaging Reports: {patient_data['imaging_reports']}

        Provide your detailed Chain-of-Thought reasoning, potential diagnosis, and confidence level. """
        
        llm_response = mock_llm.invoke(cot_prompt)
        st.text_area(f"LLM Initial Reasoning Path {i+1}:", llm_response, height=200)

        # Extracting potential diagnosis from LLM response (simplified)
        potential_diagnosis = "Unknown"
        reasoning_summary = "No specific reasoning provided."
        confidence_level = "Uncertain"

        if "Potential Diagnosis:" in llm_response:
            potential_diagnosis = llm_response.split("Potential Diagnosis:")[1].split('\n')[0].strip()
        if "Reasoning:" in llm_response:
            reasoning_summary = llm_response.split("Reasoning:")[1].split('\n')[0].strip()
        if "Confidence:" in llm_response:
            confidence_level = llm_response.split("Confidence:")[1].split('\n')[0].strip()

        # 2. Chain-of-Verification
        st.markdown(f"#### Verifying Reasoning Path {i+1}...")
        verification_queries = [
            f"Is '{potential_diagnosis}' consistent with '{patient_data['symptoms']}'?",
            f"What are the key diagnostic criteria for '{potential_diagnosis}'?",
            f"Are there any contradictions between the patient's data and '{potential_diagnosis}'?"
        ]
        
        verification_results = []
        for query in verification_queries:
            rag_results = mock_vector_store.similarity_search(query, k=2)
            verification_results.extend(rag_results)
            st.text(f"  - RAG Query: '{query[:50]}...' -> Found: {len(rag_results)} relevant documents.")

        verification_prompt = f"""Based on the following reasoning path and retrieved medical knowledge, please verify the proposed diagnosis. Identify any inconsistencies or areas needing further clarification.

        Proposed Diagnosis: {potential_diagnosis}
        LLM Reasoning: {llm_response}
        Retrieved Medical Knowledge: {'\n'.join(verification_results)}

        Verification outcome:"""
        
        verification_response = mock_llm.invoke(verification_prompt)
        st.text_area(f"LLM Verification Output {i+1}:", verification_response, height=100)

        is_consistent = "consistent" in verification_response.lower() and "inconsistencies found" not in verification_response.lower()
        if not is_consistent:
            st.warning(f"Path {i+1} flagged for potential inconsistencies or low confidence during verification.")
            # Simulate re-prompting or adjustment if inconsistencies found
            adjustment_prompt = f"""Based on the verification feedback: '{verification_response}', re-evaluate the diagnosis for patient with symptoms: {patient_data['symptoms']}. Adjust reasoning as needed.
            Revised Potential Diagnosis and Reasoning:"""
            adjusted_response = mock_llm.invoke(adjustment_prompt)
            st.info(f"Path {i+1} adjusted based on verification feedback.")
            llm_response = adjusted_response # Use adjusted response for aggregation
            
            if "Potential Diagnosis:" in adjusted_response:
                potential_diagnosis = adjusted_response.split("Potential Diagnosis:")[1].split('\n')[0].strip()
            if "Reasoning:" in adjusted_response:
                reasoning_summary = adjusted_response.split("Reasoning:")[1].split('\n')[0].strip()
            if "Confidence:" in adjusted_response:
                confidence_level = adjusted_response.split("Confidence:")[1].split('\n')[0].strip()


        diagnoses_with_reasoning.append({
            "diagnosis": potential_diagnosis,
            "reasoning": reasoning_summary,
            "full_llm_response": llm_response,
            "confidence": confidence_level,
            "verified": is_consistent
        })
    
    # 3. Aggregation and Confidence Scoring (Self-Consistency)
    st.subheader("Aggregating Results (Self-Consistency)")
    final_suggestions = {}
    for diag in diagnoses_with_reasoning:
        d = diag["diagnosis"]
        if d not in final_suggestions:
            final_suggestions[d] = {"count": 0, "reasonings": [], "confidences": [], "verified_count": 0}
        
        final_suggestions[d]["count"] += 1
        final_suggestions[d]["reasonings"].append(diag["reasoning"])
        final_suggestions[d]["confidences"].append(diag["confidence"])
        if diag["verified"]:
            final_suggestions[d]["verified_count"] += 1

    aggregated_output = []
    for diagnosis, data in final_suggestions.items():
        avg_confidence = "".join(set(data["confidences"])) if len(set(data["confidences"])) == 1 else "Mixed"
        if "High" in data["confidences"] and data["verified_count"] > data["count"] / 2:
            final_confidence = "High (Verified)"
        elif "Medium" in data["confidences"] and data["verified_count"] > data["count"] / 2:
            final_confidence = "Medium (Verified)"
        else:
            final_confidence = avg_confidence + " (Verification Mixed/Low)"
        
        aggregated_output.append({
            "diagnosis": diagnosis,
            "confidence": final_confidence,
            "support_count": data["count"],
            "summary_reasoning": f"Multiple paths suggest this diagnosis. Sample reasoning: {data['reasonings'][0]} "
        })
    
    # Sort by confidence/support
    aggregated_output.sort(key=lambda x: (x["confidence"].count("High") * 10 + x["confidence"].count("Medium") * 5 + x["support_count"]), reverse=True)

    return aggregated_output

# --- Streamlit UI --- 
st.set_page_config(layout="wide", page_title="Intelligent Medical Diagnostic Assistant")

st.title("🧠 Intelligent Medical Diagnostic Assistant")
st.markdown("---\nThis application assists healthcare professionals in diagnosing complex medical conditions using **Enhanced LLM Reasoning and Reliability** techniques like Chain-of-Thought (CoT), Self-Consistency, and Chain-of-Verification (CoV).")

with st.sidebar:
    st.header("Patient Data Input")
    symptoms = st.text_area("Patient Symptoms (e.g., severe sore throat, fever, no cough)",
                            "Persistent fatigue, generalized joint pain, occasional skin rash, low-grade fever")
    history = st.text_area("Medical History (e.g., allergic to penicillin, diabetes type 2)",
                           "No significant past medical history, no known allergies.")
    lab_results = st.text_area("Lab Results (e.g., CRP elevated, ESR normal)",
                               "CBC normal, ESR elevated (35 mm/hr), ANA positive (1:160, speckled pattern), CRP slightly elevated.")
    imaging_reports = st.text_area("Imaging Reports (e.g., Chest X-ray clear)",
                                   "No recent imaging reports available.")
    
    num_reasoning_paths = st.slider("Number of Reasoning Paths (Self-Consistency)", 1, 5, 3)

    if st.button("Get Diagnostic Suggestions"): 
        if not symptoms and not history:
            st.warning("Please enter at least patient symptoms or medical history.")
        else:
            patient_data = {
                "symptoms": symptoms,
                "history": history,
                "lab_results": lab_results,
                "imaging_reports": imaging_reports
            }
            
            with st.spinner("AI is analyzing patient data and generating diagnoses..."): 
                st.session_state.diagnostic_results = run_diagnostic_chain(patient_data, num_reasoning_paths)


st.header("Diagnostic Suggestions")

if "diagnostic_results" in st.session_state and st.session_state.diagnostic_results:
    for i, result in enumerate(st.session_state.diagnostic_results):
        st.markdown(f"### {i+1}. {result['diagnosis']} ")
        st.markdown(f"**Confidence:** {result['confidence']} | **Support from Paths:** {result['support_count']}/{num_reasoning_paths}")
        with st.expander("View Detailed Reasoning"): 
            st.write(result['summary_reasoning'])
else:
    st.info("Enter patient data in the sidebar and click 'Get Diagnostic Suggestions' to begin.")

st.markdown("--- ")
st.caption("Disclaimer: This AI assistant is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any medical concerns.")
