from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os

# Set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Setup LLM and Embeddings
llm = OpenAI(temperature=0.7)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Simulate Knowledge Bases
# Dummy Internal Knowledge Base (EHR, internal guidelines)
internal_docs = [
    Document(page_content="Patient John Doe, 45, presented with persistent cough, fever 102F, and shortness of breath for 3 days. Diagnosed with pneumonia 6 months ago. Lab results show elevated CRP and white blood cell count.", metadata={"source": "EHR_JohnDoe"}),
    Document(page_content="Patient Jane Smith, 60, diabetic, complaining of fatigue and frequent urination. Blood glucose 250 mg/dL. History of hypertension. Recent kidney function tests show elevated creatinine.", metadata={"source": "EHR_JaneSmith"}),
    Document(page_content="Hospital guideline for pneumonia: Initial treatment typically involves broad-spectrum antibiotics. Consider chest X-ray and sputum culture for confirmation.", metadata={"source": "Hospital_Guidelines"}),
]

# Dummy External Knowledge Base (PubMed abstracts, clinical guidelines)
external_docs = [
    Document(page_content="Pneumonia treatment guidelines: Common antibiotics include azithromycin, doxycycline, or amoxicillin-clavulanate. Duration usually 5-7 days. Follow-up chest X-ray may be considered.", metadata={"source": "PubMed_Pneumonia"}),
    Document(page_content="Type 2 Diabetes management: First-line therapy is often metformin. Lifestyle modifications (diet, exercise) are crucial. Regular monitoring of HbA1c, blood pressure, and kidney function is recommended.", metadata={"source": "PubMed_Diabetes"}),
    Document(page_content="Differential diagnosis for persistent cough includes pneumonia, acute bronchitis, asthma exacerbation, GERD, and post-nasal drip. A detailed patient history is essential.", metadata={"source": "Clinical_Review_Cough"}),
    Document(page_content="Acute Kidney Injury (AKI) in diabetic patients: Can be caused by various factors including dehydration, certain medications (e.g., NSAIDs, ACE inhibitors), or progression of diabetic nephropathy.", metadata={"source": "Clinical_Review_AKI"}),
]

# Combine and vectorize documents
all_docs = internal_docs + external_docs
vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3. Define Prompts
initial_query_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical diagnostic assistant. Analyze the patient's case. Formulate an initial hypothesis and suggest a concise search query to find more relevant medical information. Output only the search query. If you also want to provide an initial hypothesis, output it before the search query, separated by '---'."),
    ("human", "Patient Case: {patient_query}")
])

iterative_refinement_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical diagnostic assistant. Given the patient's case, your current understanding (Previous Hypothesis: {current_hypothesis}), and the following retrieved medical documents:\n\n{retrieved_documents}\n\nRefine your hypothesis, synthesize the new information, and if more information is needed, suggest a new, more specific search query. If you have sufficient information to provide a preliminary diagnosis, supporting evidence, and suggested next steps, state 'DIAGNOSIS_READY: ' followed by your comprehensive diagnostic recommendation. Otherwise, output a new search query to gather more information. If you also want to provide an updated hypothesis, output it before the search query, separated by '---'."),
    ("human", "Patient Case: {patient_query}")
])

final_synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical diagnostic assistant. Based on all gathered information and your final understanding (Final Hypothesis: {final_hypothesis}), provide a comprehensive diagnostic recommendation for the patient. Include supporting evidence from the provided documents and suggest clear next steps (e.g., further tests, treatment plan). Be professional and thorough.\n\nRetrieved Documents: {all_retrieved_documents}"),
    ("human", "Patient Case: {patient_query}")
])

# 4. Implement Chains for LLM interaction
initial_query_chain = initial_query_prompt | llm | StrOutputParser()

# 5. Implement the Multi-step RAG Loop
def clinical_diagnosis_assistant(patient_case: str, max_iterations: int = 3):
    print(f"\n--- Starting Clinical Diagnosis for: {patient_case} ---")
    current_hypothesis = "No initial hypothesis."
    current_search_query = patient_case # Start with the full case as a query
    all_retrieved_documents_content = []

    # Initial Processing and Retrieval
    print("\nInitial LLM processing and query generation...")
    initial_response = initial_query_chain.invoke({"patient_query": patient_case})
    
    if "---" in initial_response:
        parts = initial_response.split("---")
        current_hypothesis = parts[0].strip()
        current_search_query = parts[1].strip()
    else:
        current_search_query = initial_response.strip()

    print(f"Initial Hypothesis: {current_hypothesis}")
    print(f"Initial Search Query: {current_search_query}")

    retrieved_docs = retriever.invoke(current_search_query)
    current_retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs])
    all_retrieved_documents_content.extend([doc.page_content for doc in retrieved_docs])
    print(f"Initial Retrieved Documents: {len(retrieved_docs)} documents")

    # Iterative Refinement Loop
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1}/{max_iterations} ---")
        print(f"Current Hypothesis: {current_hypothesis}")
        print(f"Documents for LLM in this iteration:\n{current_retrieved_content}")

        iteration_response = (iterative_refinement_prompt | llm | StrOutputParser()).invoke({
            "patient_query": patient_case,
            "current_hypothesis": current_hypothesis,
            "retrieved_documents": current_retrieved_content
        })

        if iteration_response.startswith("DIAGNOSIS_READY:"):
            final_diagnosis_text = iteration_response.replace("DIAGNOSIS_READY: ", "").strip()
            print(f"\n--- DIAGNOSIS READY (from iteration {i+1}) ---")
            print(final_diagnosis_text)
            return final_diagnosis_text
        else:
            if "---" in iteration_response:
                parts = iteration_response.split("---")
                current_hypothesis = parts[0].strip()
                current_search_query = parts[1].strip()
            else:
                current_search_query = iteration_response.strip()
            
            print(f"Updated Hypothesis: {current_hypothesis}")
            print(f"New Search Query: {current_search_query}")
            
            retrieved_docs = retriever.invoke(current_search_query)
            current_retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs])
            all_retrieved_documents_content.extend([doc.page_content for doc in retrieved_docs])
            print(f"Retrieved {len(retrieved_docs)} new documents.")

    # Final Synthesis if not ready in loop
    print("\n--- Max iterations reached. Performing final synthesis. ---")
    final_diagnosis = (final_synthesis_prompt | llm | StrOutputParser()).invoke({
        "patient_query": patient_case,
        "final_hypothesis": current_hypothesis,
        "all_retrieved_documents": "\n".join(all_retrieved_documents_content)
    })
    print(f"\n--- Final Diagnosis ---")
    print(final_diagnosis)
    return final_diagnosis


# --- Example Usage ---
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set. Please set it to run the example.")
        print("Example: export OPENAI_API_KEY='your_api_key_here'")
    else:
        # Complex Case 1
        patient_case_1 = "A 60-year-old male with a history of type 2 diabetes and hypertension presents with increased fatigue, frequent urination, and recent blood tests showing elevated creatinine. He denies any chest pain or fever."
        clinical_diagnosis_assistant(patient_case_1, max_iterations=3)

        print("\n" + "="*80 + "\n")

        # Complex Case 2
        patient_case_2 = "A 45-year-old patient, John Doe, complains of a persistent cough, shortness of breath, and a low-grade fever (100.5F) for a week. He had pneumonia 6 months ago. What is the most likely diagnosis and suggested steps?"
        clinical_diagnosis_assistant(patient_case_2, max_iterations=3)
