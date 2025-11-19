import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain.retrievers import MergerRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel, Field
import os

# --- Configuration ---
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- Pydantic Model for Structured Output ---
class DiagnosticOutput(BaseModel):
    differential_diagnoses: list[str] = Field(description="List of potential differential diagnoses.")
    recommended_tests: list[str] = Field(description="List of recommended diagnostic tests.")
    treatment_considerations: list[str] = Field(description="General treatment considerations based on potential diagnoses.")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating the system's confidence in its response.")
    sources: list[str] = Field(description="List of source documents or citations used.")

# --- Knowledge Base Simulation ---
# Dummy medical documents
medical_documents = [
    "Patient presents with fever, cough, and shortness of breath. Consider pneumonia, bronchitis, or influenza. Chest X-ray and viral panel recommended. Treatment may include antibiotics or antivirals.",
    "Symptoms of type 2 diabetes include frequent urination, increased thirst, and unexplained weight loss. Diagnosis involves blood glucose tests. Management includes diet, exercise, and medication like metformin.",
    "Myocardial infarction (heart attack) symptoms: chest pain radiating to arm, shortness of breath, sweating. Immediate ECG and blood tests (troponin) are crucial. Treatment involves angioplasty or thrombolysis.",
    "Common cold typically presents with runny nose, sore throat, sneezing. It's a viral infection, usually self-limiting. Rest and fluids are primary treatment. Antibiotics are ineffective.",
    "Hypertension (high blood pressure) is often asymptomatic. Regular blood pressure monitoring is key. Lifestyle changes (diet, exercise) and antihypertensive medications (e.g., ACE inhibitors) are common treatments.",
    "Migraine headaches are characterized by severe throbbing pain, sensitivity to light and sound, nausea. Triptans and NSAIDs are common treatments. Avoiding triggers is important.",
    "Appendicitis presents with sharp pain starting around the navel and moving to the lower right abdomen, fever, nausea, vomiting. Surgical removal of the appendix (appendectomy) is the standard treatment."
]

# Initialize Embedding Model
embedding_model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name, 
                                      model_kwargs={'device': 'cpu'},
                                      encode_kwargs={'normalize_embeddings': True})

# Create FAISS Vector Store
vectorstore = FAISS.from_texts(medical_documents, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create BM25 Keyword Retriever
bm25_retriever = BM25Retriever.from_texts(medical_documents)
bm25_retriever.k = 3

# --- LLM Setup ---
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# --- Query Analysis (Simplified) ---
def analyze_query_complexity(query: str) -> str:
    if any(word in query.lower() for word in ["interaction", "specific", "precise", "mechanism"]):
        return "specific"
    return "general"

# --- Dynamic Retrieval Strategist (Simplified) ---
def dynamic_retrieval_strategist(query: str, complexity: str):
    if complexity == "specific":
        return bm25_retriever  # Prioritize keyword for specific queries
    return vector_retriever # Semantic search for general queries

# --- Iterative Context Refinement (Simplified) ---
def re_rank_documents(documents, query):
    # For simplicity, we'll just return the documents as is.
    # In a real scenario, this would involve a re-ranking model (e.g., cross-encoder).
    return documents

def assess_confidence(llm_response_text: str, retrieved_context: str) -> float:
    # Simplified confidence scoring based on presence of key phrases and context length
    if "I am unable to provide a definitive diagnosis" in llm_response_text:
        return 0.2
    if len(retrieved_context) < 100: # If context is very short, confidence might be lower
        return 0.5
    if "differential_diagnoses" in llm_response_text and "recommended_tests" in llm_response_text:
        return 0.9
    return 0.7

# --- RAG Chain ---
rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a highly accurate medical diagnostic assistant. Your goal is to provide differential diagnoses, recommended tests, and treatment considerations based ONLY on the provided context. If the context does not contain enough information, state that you cannot provide a definitive answer and suggest further investigation. Output your response in JSON format according to the Pydantic schema: {schema}"),
    ("human", "Context: {context}\n\nPatient Query: {query}")
])

parser = JsonOutputParser(pydantic_object=DiagnosticOutput)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def create_rag_chain(retriever):
    return (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | rag_prompt_template
        | llm
        | parser
    )

# --- Gradio Interface ---
def diagnose_patient(query: str):
    complexity = analyze_query_complexity(query)
    current_retriever = dynamic_retrieval_strategist(query, complexity)

    # First retrieval attempt
    retrieved_docs = current_retriever.invoke(query)
    re_ranked_docs = re_rank_documents(retrieved_docs, query) # Placeholder for re-ranking

    initial_chain = create_rag_chain(current_retriever)
    initial_response_dict = initial_chain.invoke(query)

    # Simulate iterative refinement/self-reflection
    confidence = assess_confidence(str(initial_response_dict), format_docs(re_ranked_docs))

    if confidence < 0.6 and complexity == "general": # If low confidence on a general query, try hybrid retrieval
        print("Low confidence detected for general query, attempting hybrid retrieval...")
        # Combine retrievers for a broader search
        hybrid_retriever = MergerRetriever(retrievers=[vector_retriever, bm25_retriever])
        hybrid_retriever_docs = hybrid_retriever.invoke(query)
        final_docs = re_rank_documents(hybrid_retriever_docs, query)
        final_chain = create_rag_chain(hybrid_retriever)
        final_response_dict = final_chain.invoke(query)
        final_response_dict["confidence_score"] = assess_confidence(str(final_response_dict), format_docs(final_docs))
    else:
        final_response_dict = initial_response_dict
        final_response_dict["confidence_score"] = confidence

    # Add sources from the final retrieved documents
    final_response_dict["sources"] = list(set([doc.page_content for doc in retrieved_docs]))

    # Format output for Gradio
    output_str = f"**Differential Diagnoses:**\n- " + "\n- ".join(final_response_dict.get("differential_diagnoses", ["N/A"]))
    output_str += f"\n\n**Recommended Tests:**\n- " + "\n- ".join(final_response_dict.get("recommended_tests", ["N/A"]))
    output_str += f"\n\n**Treatment Considerations:**\n- " + "\n- ".join(final_response_dict.get("treatment_considerations", ["N/A"]))
    output_str += f"\n\n**Confidence Score:** {final_response_dict.get("confidence_score", 0.0):.2f}"
    output_str += f"\n\n**Sources:**\n- " + "\n- ".join(final_response_dict.get("sources", ["No sources found."]))

    return output_str

if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY") == "YOUR_OPENAI_API_KEY":
        print("WARNING: Please set your OPENAI_API_KEY environment variable or replace 'YOUR_OPENAI_API_KEY' in the script.")

    gr.Interface(
        fn=diagnose_patient,
        inputs=gr.Textbox(lines=5, placeholder="Enter patient symptoms or medical query here..."),
        outputs=gr.Markdown(),
        title="Medical Diagnostic Assistant (Adaptive RAG)",
        description="An AI-powered assistant for healthcare professionals, leveraging Adaptive RAG to provide accurate and context-aware diagnostic support. \n\nExample queries:\n- 'Patient has fever, cough, and shortness of breath.'\n- 'What are the symptoms of type 2 diabetes?'\n- 'What is the treatment for appendicitis?'\n- 'Tell me about myocardial infarction.'"
    ).launch()