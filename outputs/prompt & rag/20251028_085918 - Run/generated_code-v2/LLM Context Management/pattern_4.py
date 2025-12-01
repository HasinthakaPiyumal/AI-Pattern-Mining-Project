from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import Any, List, Mapping, Optional
import os

# --- 1. Simulate Medical Knowledge Base (Non-Parametric Memory) ---
medical_documents = [
    "Symptoms of common cold include runny nose, sore throat, cough, congestion, slight body aches or mild headache, sneezing, low-grade fever, and a general feeling of being unwell (malaise). Treatment is typically rest and fluids.",
    "Influenza (flu) symptoms are often more severe and come on more suddenly than a cold. They include fever, chills, muscle aches, cough, sore throat, runny or stuffy nose, headache, and fatigue. Antiviral drugs can be prescribed in some cases.",
    "Diabetes mellitus is a chronic condition that affects how your body turns food into energy. Symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
    "Hypertension (high blood pressure) often has no symptoms. When symptoms do occur, they can include headaches, shortness of breath, nosebleeds. Regular monitoring is crucial. Lifestyle changes and medication are common treatments.",
    "Appendicitis typically presents with sudden pain that begins on the right side of the lower abdomen. It often starts near the navel and moves. Other symptoms include nausea, vomiting, loss of appetite, fever, and constipation or diarrhea. Emergency surgery is usually required.",
    "Migraine headaches are characterized by severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Treatment involves pain relievers and preventive medications.",
    "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out, and shortness of breath. Inhalers are a common treatment.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. Antibiotics are used for bacterial pneumonia."
]

# --- 2. Embedding Model ---
# Using a Sentence-BERT model for generating embeddings
# Download if not already present. A small, efficient model is chosen.
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# --- 3. Vector Store Creation (FAISS for efficient similarity search) ---
print("Creating FAISS vector store...")
vectorstore = FAISS.from_texts(texts=medical_documents, embedding=embeddings)
print("FAISS vector store created.")

# --- 4. Retriever ---
retriever = vectorstore.as_retriever()

# --- 5. Mock LLM (Parametric Memory - for demonstration purposes) ---
# In a real application, this would be a powerful pre-trained LLM (e.g., from OpenAI, Google, Hugging Face)
# For local execution without external API keys or heavy downloads, we use a simple mock.
class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "mock_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Simulate LLM behavior based on prompt content
        if "diagnostic suggestion" in prompt.lower() and "medical evidence" in prompt.lower():
            return f"Based on the provided context, a potential diagnostic suggestion is: \"Simulated Diagnosis\". This is supported by the medical evidence stating: \"Simulated Supporting Evidence\". Please consult a healthcare professional for a definitive diagnosis."
        else:
            return "I am a mock LLM. My response is simulated based on the input structure."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "MockLLM"}

print("Initializing Mock LLM...")
mock_llm = MockLLM()
print("Mock LLM initialized.")

# --- 6. Prompt Template for RAG ---
# This template guides the LLM to use the retrieved context.
QUERY_PROMPT = PromptTemplate.from_template(
    """You are a medical diagnostic assistant. Use the following pieces of medical context to provide a diagnostic suggestion and potential treatment recommendations for the patient's symptoms.
    If you don't know the answer or can't make a confident diagnosis from the provided information, state that you cannot provide a definitive diagnosis and advise seeking professional medical advice.

    Context: {context}

    Patient Symptoms: {question}

    Diagnostic Suggestion and Treatment Recommendations:
    """
)

# --- 7. RAG Chain Construction ---
print("Constructing RAG chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=mock_llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    chain_type_kwargs={"prompt": QUERY_PROMPT}
)
print("RAG chain constructed.")

# --- 8. Diagnostic Function ---
def get_diagnostic_suggestion(symptoms: str) -> dict:
    """
    Provides a diagnostic suggestion and supporting evidence based on patient symptoms.
    """
    print(f"\nProcessing symptoms: '{symptoms}'")
    result = qa_chain({"query": symptoms})
    
    diagnosis_output = result["result"]
    source_documents = result["source_documents"]
    
    print("\n--- Diagnostic Output ---")
    print(diagnosis_output)
    
    print("\n--- Supporting Medical Evidence (from Non-Parametric Memory) ---")
    for i, doc in enumerate(source_documents):
        print(f"Source {i+1}: {doc.page_content}")
    
    return {"diagnostic_suggestion": diagnosis_output, "supporting_evidence": [doc.page_content for doc in source_documents]}

# --- 9. Example Usage ---
if __name__ == "__main__":
    print("\n--- Medical Diagnostic Assistant Demo ---")
    
    # Example 1: Symptoms matching a known condition
    patient_symptoms_1 = "I have a runny nose, sore throat, and a mild cough. I feel generally unwell."
    get_diagnostic_suggestion(patient_symptoms_1)
    
    # Example 2: Symptoms matching another known condition
    patient_symptoms_2 = "Sudden sharp pain in the lower right abdomen, nausea, and a slight fever."
    get_diagnostic_suggestion(patient_symptoms_2)

    # Example 3: Symptoms that might be vague or require more context
    patient_symptoms_3 = "I feel very tired all the time and have been drinking a lot more water than usual."
    get_diagnostic_suggestion(patient_symptoms_3)

    # Example 4: Symptoms that are not directly in the knowledge base (mock LLM will still respond generically)
    patient_symptoms_4 = "My left knee hurts when I climb stairs and it sometimes clicks."
    get_diagnostic_suggestion(patient_symptoms_4)

    print("\n--- Demo End ---")
    print("Note: This is a simplified demonstration using a mock LLM and a small knowledge base. "
          "A real system would integrate with a powerful LLM and extensive, up-to-date medical databases.")
