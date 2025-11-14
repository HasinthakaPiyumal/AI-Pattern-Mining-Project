import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import uuid # For unique IDs for suggestions

# --- Pydantic Models ---
class PatientInput(BaseModel):
    symptoms: str = Field(..., description="Patient's reported symptoms")
    medical_history: Optional[str] = Field(None, description="Patient's relevant medical history")
    age: Optional[int] = Field(None, description="Patient's age")
    gender: Optional[str] = Field(None, description="Patient's gender")

class DiagnosticSuggestion(BaseModel):
    suggestion_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the suggestion")
    diagnosis: str = Field(..., description="Proposed diagnosis or differential diagnosis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0) in the diagnosis")
    explanation: str = Field(..., description="Detailed explanation supporting the diagnosis")
    suggested_tests: List[str] = Field([], description="Recommended diagnostic tests")
    disclaimer: str = Field("This is an AI-generated suggestion and should not replace professional medical advice.", description="Standard medical disclaimer")

class Feedback(BaseModel):
    suggestion_id: str
    rating: int = Field(..., ge=1, le=5, description="User rating of the suggestion (1-5)")
    comments: Optional[str] = None

# --- 1. Medical Knowledge Base (ChromaDB & Sentence-Transformers) ---
# Using a pre-trained Sentence Transformer model for embeddings
# Note: This requires 'sentence-transformers' to be installed (pip install sentence-transformers)
# And `hnswlib` for chromadb client to work efficiently (pip install hnswlib)
embedding_function = None
try:
    from sentence_transformers import SentenceTransformer
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
except ImportError:
    st.error("Please install `sentence-transformers` for the embedding function: `pip install sentence-transformers`")
    st.warning("RAG functionality will be disabled due to missing `sentence-transformers`.")
except Exception as e:
    st.error(f"Error loading SentenceTransformer: {e}")
    st.warning("RAG functionality will be disabled.")

# Initialize ChromaDB client
# In a real application, this would connect to a persistent DB or cloud service
client = chromadb.Client()
collection_name = "medical_knowledge"
kb_collection = None
if embedding_function:
    try:
        kb_collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}. Please ensure `hnswlib` is installed (`pip install hnswlib`).")
        st.warning("RAG functionality will be disabled.")

# Populate with dummy medical data (RAG source)
if kb_collection and kb_collection.count() == 0:
    st.info("Populating medical knowledge base with dummy data...")
    documents = [
        "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness, and at times can lead to death. Symptoms include fever, cough, sore throat, muscle aches, and fatigue. Diagnosis is often clinical, but can be confirmed with a rapid influenza diagnostic test. Treatment includes antiviral drugs if started early.",
        "Common cold is a viral infection of the nose and throat. Symptoms include runny nose, sneezing, sore throat, and cough. It is generally milder than the flu. No specific antiviral treatment, but rest and symptom relief are recommended.",
        "Migraine is a severe headache often accompanied by symptoms such as throbbing in the head, sensitivity to light, sound, or smells, nausea, and vomiting. Triggers can include stress, certain foods, and hormonal changes. Treatments include pain relievers, triptans, and preventative medications.",
        "Tension headache is the most common type of headache. It causes mild to moderate pain that feels like a tight band around the head. Stress and muscle tension are common causes. Over-the-counter pain relievers usually provide relief.",
        "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon. Symptoms typically include sudden pain that begins around your navel and shifts to your lower right abdomen, nausea, vomiting, and fever. It requires prompt medical attention and often surgery.",
        "Gastroenteritis, often called stomach flu, is an inflammation of the lining of the intestines caused by a virus, bacteria, or parasites. Symptoms include diarrhea, vomiting, abdominal cramps, and sometimes fever. Hydration is key to treatment."
    ]
    metadatas = [
        {"source": "CDC", "condition": "Influenza"},
        {"source": "Mayo Clinic", "condition": "Common Cold"},
        {"source": "WHO", "condition": "Migraine"},
        {"source": "NIH", "condition": "Tension Headache"},
        {"source": "Johns Hopkins", "condition": "Appendicitis"},
        {"source": "WebMD", "condition": "Gastroenteritis"}
    ]
    ids = [f"doc{i}" for i in range(len(documents))]
    try:
        kb_collection.add(documents=documents, metadatas=metadatas, ids=ids)
        st.success(f"Knowledge base populated with {kb_collection.count()} documents.")
    except Exception as e:
        st.error(f"Failed to add documents to ChromaDB: {e}")

# --- 2. Generative AI Model (LLM) Wrapper (Simulated) ---
# In a real application, this would interface with OpenAI, Google Gemini, or a local model (e.g., Llama 3 via Transformers)
class LLM_Wrapper:
    def __init__(self, model_name="Simulated_LLM"):
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        # Simulate LLM's response based on keywords in the prompt
        # This aims to return a JSON string matching DiagnosticSuggestion
        if "influenza" in prompt.lower() or "flu" in prompt.lower() or ("fever" in prompt.lower() and "cough" in prompt.lower() and "aches" in prompt.lower()):
            return json.dumps(DiagnosticSuggestion(
                diagnosis="Influenza (Flu)",
                confidence=0.85,
                explanation="Based on your symptoms and the retrieved medical knowledge, influenza is a strong possibility. It is characterized by fever, body aches, and respiratory symptoms.",
                suggested_tests=["Rapid influenza diagnostic test"],
            ).dict())
        elif "cold" in prompt.lower() or "runny nose" in prompt.lower() or ("sneezing" in prompt.lower() and "sore throat" in prompt.lower()):
            return json.dumps(DiagnosticSuggestion(
                diagnosis="Common Cold",
                confidence=0.75,
                explanation="Your symptoms are consistent with a common cold, a mild viral infection of the upper respiratory tract.",
                suggested_tests=["None specific"],
            ).dict())
        elif "migraine" in prompt.lower() or "throbbing head" in prompt.lower() or ("light sensitivity" in prompt.lower() and "nausea" in prompt.lower()):
            return json.dumps(DiagnosticSuggestion(
                diagnosis="Migraine",
                confidence=0.90,
                explanation="Severe, throbbing headache with light sensitivity strongly suggests a migraine. Triggers should be identified.",
                suggested_tests=["Neurological exam (if severe/new onset)"],
            ).dict())
        elif "abdominal pain" in prompt.lower() and "lower right" in prompt.lower() and "nausea" in prompt.lower() and "fever" in prompt.lower():
            return json.dumps(DiagnosticSuggestion(
                diagnosis="Possible Appendicitis",
                confidence=0.95,
                explanation="Sudden, shifting abdominal pain to the lower right, accompanied by nausea and fever, is highly suspicious for appendicitis. Immediate medical evaluation is crucial.",
                suggested_tests=["Physical examination", "Blood tests (CBC)", "Imaging (ultrasound/CT scan)"],
            ).dict())
        elif "diarrhea" in prompt.lower() or "vomiting" in prompt.lower() or "stomach cramps" in prompt.lower():
            return json.dumps(DiagnosticSuggestion(
                diagnosis="Gastroenteritis (Stomach Flu)",
                confidence=0.80,
                explanation="Diarrhea, vomiting, and stomach cramps are classic symptoms of gastroenteritis. Focus on hydration.",
                suggested_tests=["Stool sample (if severe/persistent)"],
            ).dict())
        elif "headache" in prompt.lower() and "band around head" in prompt.lower():
            return json.dumps(DiagnosticSuggestion(
                diagnosis="Tension Headache",
                confidence=0.70,
                explanation="A headache that feels like a tight band around the head is characteristic of a tension headache, often related to stress.",
                suggested_tests=["Over-the-counter pain relievers"],
            ).dict())
        else:
            # Default general response if no specific match
            return json.dumps(DiagnosticSuggestion(
                diagnosis="Non-specific symptoms",
                confidence=0.60,
                explanation="Your symptoms are non-specific, and further information or medical evaluation is needed to determine a precise diagnosis. Consider consulting a healthcare professional.",
                suggested_tests=["General physical exam"],
            ).dict())

# --- 3. LLM Orchestration & Prompt Engineering Module ---
class PromptEngineer:
    def __init__(self, llm_wrapper: LLM_Wrapper, kb_collection: Optional[chromadb.api.models.Collection.Collection]):
        self.llm = llm_wrapper
        self.kb_collection = kb_collection

    def _retrieve_relevant_knowledge(self, query: str, top_k: int = 2) -> str:
        if not self.kb_collection:
            return ""
        try:
            results = self.kb_collection.query(
                query_texts=[query],
                n_results=top_k
            )
            retrieved_docs = results['documents'][0] if results and results['documents'] else []
            return "\n\nRelevant Medical Knowledge:\n" + "\n---\n".join(retrieved_docs) if retrieved_docs else ""
        except Exception as e:
            st.warning(f"Error during knowledge retrieval: {e}")
            return ""

    def generate_prompt(self, patient_input: PatientInput, prompt_type: str = "role_based") -> str:
        base_prompt = "You are an AI-powered Medical Diagnostic Assistant. Your goal is to provide a comprehensive and accurate diagnostic suggestion, explanation, and recommended tests based on the patient's input. Always prioritize patient safety and provide a clear disclaimer that this is not a substitute for professional medical advice. Structure your output as a JSON object matching the DiagnosticSuggestion Pydantic model. DO NOT include any text outside the JSON object.\n\n"

        context = f"Patient Symptoms: {patient_input.symptoms}\n"
        if patient_input.medical_history:
            context += f"Patient Medical History: {patient_input.medical_history}\n"
        if patient_input.age:
            context += f"Patient Age: {patient_input.age}\n"
        if patient_input.gender:
            context += f"Patient Gender: {patient_input.gender}\n"

        # RAG Integration
        relevant_knowledge = self._retrieve_relevant_knowledge(patient_input.symptoms + (patient_input.medical_history or ""))
        context += relevant_knowledge

        if prompt_type == "role_based":
            role_prompt = "Act as a highly experienced diagnostician specializing in general medicine. Focus on differential diagnoses and clear reasoning. \
                           Given the following patient information, provide a diagnostic suggestion. \
                           Ensure your output is a valid JSON object adhering to the DiagnosticSuggestion schema. \
                           DO NOT include any text outside the JSON object.\n\n"
            return base_prompt + role_prompt + context
        elif prompt_type == "few_shot":
            # Simplified few-shot: in a real app, examples would be dynamically selected from KB
            # Example for Migraine
            few_shot_example = json.dumps(DiagnosticSuggestion(
                diagnosis="Migraine",
                confidence=0.90,
                explanation="The patient's description of a severe, throbbing headache with sensitivity to light and sound, accompanied by nausea, is highly consistent with a migraine attack. It is important to identify potential triggers for future prevention.",
                suggested_tests=["Neurological examination (if new onset or atypical)"],
            ).dict())
            few_shot_context = f"Example Input: Patient Symptoms: Severe throbbing headache, sensitive to light and sound, nausea. Patient Age: 35. Patient Gender: Female.\nExample Output: {few_shot_example}\n\n"
            return base_prompt + few_shot_context + "Now, for the following patient:\n" + context
        else: # Zero-shot as default if prompt_type is not recognized
            return base_prompt + "Given the following patient information, provide a diagnostic suggestion. \
                           Ensure your output is a valid JSON object adhering to the DiagnosticSuggestion schema. \
                           DO NOT include any text outside the JSON object.\n\n" + context

    def get_diagnostic_suggestion(self, patient_input: PatientInput, prompt_type: str = "role_based") -> Optional[DiagnosticSuggestion]:
        prompt = self.generate_prompt(patient_input, prompt_type)
        llm_raw_response = self.llm.generate_response(prompt)
        
        try:
            # Try to parse the LLM's response as JSON and validate with Pydantic
            response_dict = json.loads(llm_raw_response)
            suggestion = DiagnosticSuggestion(**response_dict)
            return suggestion
        except (json.JSONDecodeError, ValidationError) as e:
            st.error(f"LLM response parsing error: {e}\nRaw response: {llm_raw_response}")
            # Fallback for ill-formatted responses
            return DiagnosticSuggestion(
                diagnosis="Error in AI response",
                confidence=0.1,
                explanation=f"The AI encountered an issue generating a structured response. Please try again or provide clearer input. Original error: {e}",
                suggested_tests=[]
            )

# --- 4. Quality Assurance & Evaluation Framework (Simplified) ---
class QualityAssurance:
    def __init__(self, kb_collection: Optional[chromadb.api.models.Collection.Collection]):
        self.kb_collection = kb_collection

    def llm_autorating(self, suggestion: DiagnosticSuggestion, patient_input: PatientInput) -> float:
        # Simulate an LLM-based autorating. In a real system, a separate, more robust LLM would do this.
        score = suggestion.confidence * 0.9 # Base score on confidence
        if "error" in suggestion.diagnosis.lower() or "issue" in suggestion.explanation.lower():
            score *= 0.5 # Penalize for internal errors
        
        # Simple check for relevance based on symptoms
        if any(keyword in suggestion.diagnosis.lower() for keyword in patient_input.symptoms.lower().split()):
            score += 0.05 # Small bonus for direct symptom mention in diagnosis

        return min(1.0, max(0.0, score))

    def factual_consistency_checker(self, suggestion: DiagnosticSuggestion) -> bool:
        if not self.kb_collection:
            return True # Cannot check without KB, assume true for safety
        # Simulate checking if diagnosis or key terms from explanation exist in KB
        query = f"{suggestion.diagnosis} {suggestion.explanation}"
        try:
            results = self.kb_collection.query(query_texts=[query], n_results=1)
            # A very low distance means high similarity, indicating consistency
            if results and results['distances'] and results['distances'][0][0] < 0.3: # Threshold for similarity
                return True
        except Exception as e:
            st.warning(f"Factual consistency check failed: {e}")
            return False # Assume false if check fails
        return False

    def ethical_alignment_check(self, suggestion: DiagnosticSuggestion) -> bool:
        # Constitutional AI Principles (simplified checks)
        # 1. Prioritize patient safety: Assume suggestions do this unless explicit negative keywords
        if any(keyword in suggestion.explanation.lower() for keyword in ["harmful", "dangerous", "unethical"]):
            return False
        # 2. Do not provide definitive diagnoses, only suggestions (checked by disclaimer and phrasing)
        if "definite diagnosis" in suggestion.explanation.lower() or "certainly have" in suggestion.explanation.lower():
            return False
        # 3. Always include disclaimer
        if suggestion.disclaimer not in suggestion.explanation and "AI-generated suggestion and should not replace professional medical advice" not in suggestion.explanation:
            return False # This is a strict check
        # 4. Avoid speculative language (hard to check without deeper NLP)
        # 5. Do not recommend off-label treatments (not covered in this simplified example)
        return True

# --- 5. FastAPI Backend Simulation (Functions) ---
# In a real setup, these would be `@app.post` endpoints
llm_wrapper = LLM_Wrapper()
if kb_collection is None:
    st.warning("Knowledge base is not initialized or failed to load. RAG functionality will be limited.")
prompt_engineer = PromptEngineer(llm_wrapper, kb_collection)
quality_assurance = QualityAssurance(kb_collection)

def diagnose_patient_backend(patient_input: PatientInput, prompt_type: str = "role_based") -> DiagnosticSuggestion:
    # Store the suggestion in session_state to allow feedback after refresh
    st.session_state.current_suggestion = prompt_engineer.get_diagnostic_suggestion(patient_input, prompt_type)
    return st.session_state.current_suggestion

def submit_feedback_backend(feedback: Feedback):
    # In a real app, this would save feedback to a database
    st.success(f"Feedback received for suggestion {feedback.suggestion_id}: Rating {feedback.rating}, Comments: {feedback.comments}")
    print(f"Logged Feedback: {feedback.dict()}") # Log to console for demonstration

# --- 6. Streamlit Frontend ---
st.set_page_config(layout="wide", page_title="AI Medical Diagnostic Assistant")

st.title("🩺 AI-Powered Medical Diagnostic Assistant")
st.markdown("---Disclaimer: This assistant provides AI-generated suggestions for informational purposes only and should NOT replace professional medical advice, diagnosis, or treatment.---")

# Input Form
st.header("Patient Information")
with st.form("patient_form"):
    symptoms = st.text_area("Describe patient's symptoms (e.g., 'severe throbbing headache, sensitive to light and sound, nausea')", key="symptoms_input")
    medical_history = st.text_area("Relevant medical history (optional)", key="history_input")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (optional)", min_value=0, max_value=120, key="age_input")
    with col2:
        gender = st.selectbox("Gender (optional)", [None, "Male", "Female", "Other"], key="gender_input")
    
    prompt_strategy = st.radio(
        "Choose AI Prompting Strategy:",
        ("Role-Based", "Few-Shot", "Zero-Shot"),
        key="prompt_strategy_radio",
        horizontal=True
    )

    submitted = st.form_submit_button("Get Diagnostic Suggestion")

    if submitted:
        if not symptoms:
            st.error("Please describe the patient's symptoms.")
        else:
            patient_input = PatientInput(
                symptoms=symptoms,
                medical_history=medical_history if medical_history else None,
                age=age if age > 0 else None,
                gender=gender
            )
            
            with st.spinner("Generating suggestion..."):
                # Convert prompt_strategy to lowercase and replace hyphens for function call
                suggestion = diagnose_patient_backend(patient_input, prompt_strategy.lower().replace('-','_'))

            if suggestion:
                st.subheader("Diagnostic Suggestion")
                st.write(f"**Diagnosis:** {suggestion.diagnosis}")
                st.progress(suggestion.confidence, text=f"Confidence: {suggestion.confidence:.2f}")
                st.write(f"**Explanation:** {suggestion.explanation}")
                if suggestion.suggested_tests:
                    st.write(f"**Suggested Tests:** {', '.join(suggestion.suggested_tests)}")
                st.info(suggestion.disclaimer)

                st.subheader("Quality Assurance Checks (Simplified)")
                qa_score = quality_assurance.llm_autorating(suggestion, patient_input)
                st.write(f"- **AI Autorating Score:** {qa_score:.2f} (out of 1.0)")
                
                if quality_assurance.factual_consistency_checker(suggestion):
                    st.success("- **Factual Consistency:** Appears consistent with medical knowledge.")
                else:
                    st.warning("- **Factual Consistency:** May have inconsistencies with medical knowledge. Further review recommended.")
                
                if quality_assurance.ethical_alignment_check(suggestion):
                    st.success("- **Ethical Alignment:** Appears to follow ethical guidelines.")
                else:
                    st.warning("- **Ethical Alignment:** Potential ethical concern detected. Review immediately.")

                st.subheader("Provide Feedback")
                # Use a unique key for the feedback form after submission
                feedback_form_key = f"feedback_form_{suggestion.suggestion_id}"
                with st.form(feedback_form_key):
                    feedback_rating = st.slider("Rate the helpfulness of this suggestion:", min_value=1, max_value=5, value=3, key=f"rating_{suggestion.suggestion_id}")
                    feedback_comments = st.text_area("Any additional comments? (optional)", key=f"comments_{suggestion.suggestion_id}")
                    feedback_submitted = st.form_submit_button("Submit Feedback", key=f"submit_{suggestion.suggestion_id}")

                    if feedback_submitted:
                        feedback_obj = Feedback(
                            suggestion_id=suggestion.suggestion_id,
                            rating=feedback_rating,
                            comments=feedback_comments if feedback_comments else None
                        )
                        submit_feedback_backend(feedback_obj)
                        st.success("Thank you for your feedback!")
            else:
                st.error("Could not generate a diagnostic suggestion. Please check inputs and try again.")

# Handle displaying previous suggestion for feedback if the page reloads (e.g., due to code changes or external factors)
# This is a fallback and generally handled better with Streamlit's form submission mechanism
if 'current_suggestion' in st.session_state and st.session_state.current_suggestion and not submitted:
    st.subheader("Previous Suggestion (for feedback)")
    suggestion = st.session_state.current_suggestion
    st.write(f"**Diagnosis:** {suggestion.diagnosis}")
    st.progress(suggestion.confidence, text=f"Confidence: {suggestion.confidence:.2f}")
    st.write(f"**Explanation:** {suggestion.explanation}")
    if suggestion.suggested_tests:
        st.write(f"**Suggested Tests:** {', '.join(suggestion.suggested_tests)}")
    st.info(suggestion.disclaimer)

    # Only show feedback form if it hasn't been submitted for this suggestion yet
    if f"feedback_form_{suggestion.suggestion_id}" not in st.session_state:
        with st.form(f"feedback_form_reloaded_{suggestion.suggestion_id}"):
            feedback_rating_reloaded = st.slider("Rate the helpfulness of this suggestion:", min_value=1, max_value=5, value=3, key=f"rating_reloaded_{suggestion.suggestion_id}")
            feedback_comments_reloaded = st.text_area("Any additional comments? (optional)", key=f"comments_reloaded_{suggestion.suggestion_id}")
            feedback_submitted_reloaded = st.form_submit_button("Submit Feedback", key=f"submit_reloaded_{suggestion.suggestion_id}")

            if feedback_submitted_reloaded:
                feedback_obj = Feedback(
                    suggestion_id=suggestion.suggestion_id,
                    rating=feedback_rating_reloaded,
                    comments=feedback_comments_reloaded if feedback_comments_reloaded else None
                )
                submit_feedback_backend(feedback_obj)
                st.success("Thank you for your feedback!")
                st.session_state.current_suggestion = None # Clear after feedback
