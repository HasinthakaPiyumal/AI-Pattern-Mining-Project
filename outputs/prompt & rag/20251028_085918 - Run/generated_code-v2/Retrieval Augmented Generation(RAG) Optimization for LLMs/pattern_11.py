import streamlit as st
import os
import random

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration --- #
# Set your OpenAI API key in your environment variables or replace os.getenv with your key
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it as an environment variable.")
    st.stop()

CONFIDENCE_THRESHOLD = 0.75 # Simulated confidence threshold
MAX_RETRIEVAL_STEPS = 3 # Maximum iterations for retrieval

# --- Simulated Medical Database --- #
MEDICAL_DOCUMENTS = [
    "Symptoms of influenza include fever, cough, sore throat, and body aches.",
    "Treatment for Type 2 Diabetes often involves diet control, exercise, and medications like Metformin.",
    "Hypertension, or high blood pressure, can lead to heart disease and stroke.",
    "Common side effects of antibiotics include nausea, diarrhea, and allergic reactions.",
    "The COVID-19 vaccine helps prevent severe illness and transmission.",
    "First aid for minor burns involves cooling the burn with cool water and covering it with a sterile dressing.",
    "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways.",
    "Migraines are severe headaches often accompanied by throbbing pain, sensitivity to light and sound, and nausea.",
    "Causes of anemia can include iron deficiency, vitamin deficiencies, and chronic diseases.",
    "Understanding cardiovascular risk factors is crucial for preventing heart disease.",
    "Pulmonary embolism is a serious condition where a blood clot blocks an artery in the lungs.",
    "Osteoporosis is a condition where bones become brittle and fragile from loss of tissue, typically as a result of hormonal changes, or deficiency of calcium or vitamin D.",
    "Symptoms of a heart attack include chest pain, shortness of breath, pain in the left arm, and sweating."
]

# --- Initialize LLM and Embeddings --- #
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store from simulated documents
vectorstore = FAISS.from_texts(MEDICAL_DOCUMENTS, embeddings)
retriever = vectorstore.as_retriever()

# --- Confidence Simulation --- #
def get_llm_confidence(response_text: str) -> float:
    # This is a simulated confidence score.
    # In a real system, this would come from the LLM's internal token probabilities
    # or a separate confidence model.
    # For demonstration, we'll assign a higher confidence for longer, more detailed answers
    # and introduce some randomness.
    base_confidence = min(1.0, len(response_text) / 200.0 + random.uniform(-0.1, 0.1))
    return max(0.2, base_confidence)

# --- Iterative QA Logic --- #
def iterative_qa(question: str):
    st.write(f"Processing question: '{question}'")
    current_context_documents = []
    confidence = 0.0
    iterations = 0
    final_answer = "No reliable answer could be generated after several attempts."

    while confidence < CONFIDENCE_THRESHOLD and iterations < MAX_RETRIEVAL_STEPS:
        iterations += 1
        st.write(f"\n--- Iteration {iterations} ---")

        # Retrieve relevant documents
        new_documents = retriever.get_relevant_documents(question)
        current_context_documents.extend([doc.page_content for doc in new_documents])
        current_context_documents = list(set(current_context_documents)) # Remove duplicates

        # Prepare prompt with augmented context
        context_str = "\n".join(current_context_documents)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful medical assistant. Answer the user's question based ONLY on the provided context. If you cannot find the answer, state that you don't have enough information. Be precise and concise."),
            ("human", f"Context: {context_str}\nQuestion: {question}")
        ])

        chain = prompt_template | llm # Simple chain
        
        # Invoke the LLM
        try:
            response = chain.invoke({"question": question, "context": context_str})
            llm_answer = response.content if hasattr(response, 'content') else str(response)
            confidence = get_llm_confidence(llm_answer)

            st.write(f"Retrieved {len(new_documents)} new documents.")
            st.write(f"Current Answer Confidence: {confidence:.2f}")
            st.write(f"Current Answer (Partial/Refined): {llm_answer[:200]}...")

            if confidence >= CONFIDENCE_THRESHOLD:
                final_answer = llm_answer
                st.success(f"Confidence threshold met! Final answer generated in {iterations} iterations.")
                break
            else:
                st.info("Confidence too low, attempting further retrieval...")
                final_answer = llm_answer # Keep the latest answer even if low confidence

        except Exception as e:
            st.error(f"An error occurred during LLM invocation: {e}")
            break

    st.write("\n--- Final Result ---")
    st.write(f"Final Answer after {iterations} iterations:")
    st.success(final_answer)
    st.write(f"Simulated Final Confidence: {confidence:.2f}")
    st.write(f"Total documents considered: {len(current_context_documents)}")

# --- Streamlit UI --- #
st.set_page_config(page_title="Medical Information Assistant", layout="wide")
st.title("🧠 Medical Information Assistant")
st.markdown("Ask a medical question and the assistant will iteratively retrieve information to provide a confident answer.")

question_input = st.text_area("Enter your medical question here:", height=100)

if st.button("Get Answer"): # Added a button to trigger the action
    if question_input:
        with st.spinner("Generating answer..."):
            iterative_qa(question_input)
    else:
        st.warning("Please enter a question.")
