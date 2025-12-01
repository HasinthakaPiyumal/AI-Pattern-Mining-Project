import streamlit as st
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any, List
import os

class MockLLM:
    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name

    def invoke(self, prompt: str) -> str:
        if "multi-step iterative" in prompt:
            return f"MockLLM ({self.model_name}) multi-step response for: {prompt[:100]}..."
        elif "single-step retrieval" in prompt:
            return f"MockLLM ({self.model_name}) single-step response for: {prompt[:100]}..."
        else:
            return f"MockLLM ({self.model_name}) direct response for: {prompt[:100]}..."

class MockSentenceTransformer:
    def encode(self, texts: List[str]) -> List[List[float]]:
        return [[float(hash(text) % 1000) / 1000] * 384 for text in texts] # Mock 384-dim embeddings

class MockChromaDB:
    def __init__(self, embedding_function):
        self.collection = {}
        self.embedding_function = embedding_function
        self.documents = []
        self.ids = []

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        embeddings = self.embedding_function.encode(documents)
        for i, doc in enumerate(documents):
            doc_id = ids[i] if ids else f"doc_{len(self.collection) + i}"
            self.collection[doc_id] = {
                "document": doc,
                "embedding": embeddings[i],
                "metadata": metadatas[i] if metadatas else {}
            }
            self.documents.append(doc)
            self.ids.append(doc_id)

    def similarity_search(self, query: str, k: int = 1) -> List[str]:
        query_embedding = self.embedding_function.encode([query])[0]
        if not self.documents:
            return []
        return [doc for doc in self.documents[:k]]

def classify_query_complexity(query: str) -> str:
    query = query.lower()
    if any(keyword in query for keyword in ["what is", "define", "who is", "when was"]):
        return "simple"
    elif any(keyword in query for keyword in ["how does", "explain the mechanism", "compare and contrast"]):
        return "medium"
    elif any(keyword in query for keyword in ["long term effects of", "differential diagnosis for", "treatment protocols for complex", "implications of"]):
        return "complex"
    return "medium"

medical_docs = [
    "Influenza, commonly known as the flu, is an infectious disease caused by an influenza virus.",
    "Symptoms of influenza include fever, sore throat, muscle pains, headache, coughing, and fatigue.",
    "Diabetes mellitus, commonly known as diabetes, is a metabolic disease that causes high blood sugar.",
    "There are two main types of diabetes: Type 1 and Type 2. Type 1 is an autoimmune disease where the body does not produce insulin. Type 2 diabetes occurs when the body either doesn't produce enough insulin or doesn't effectively use the insulin it does produce.",
    "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
    "Treatment for hypertension often includes lifestyle changes like diet and exercise, and sometimes medication such as diuretics, ACE inhibitors, or calcium channel blockers.",
    "A headache is pain in any region of the head. Headaches can be a symptom of a wide range of conditions, including stress, dehydration, or more serious underlying issues like a brain tumor or stroke.",
    "To diagnose a headache, a doctor may ask about the patient's medical history, perform a physical exam, and sometimes order imaging tests like an MRI or CT scan.",
    "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention.",
    "Common symptoms of COVID-19 include fever, cough, fatigue, and loss of taste or smell. More severe symptoms can include difficulty breathing, chest pain, and loss of speech or movement."
]

llm = MockLLM()
embedding_model = MockSentenceTransformer()
vectorstore = MockChromaDB(embedding_function=embedding_model)
vectorstore.add_documents(medical_docs)

def no_retrieval_qa(query: str, llm_instance: MockLLM) -> str:
    prompt = f"Answer the following question directly, based on general knowledge: {query}"
    return llm_instance.invoke(prompt)

def single_step_rag(query: str, llm_instance: MockLLM, vectorstore_instance: MockChromaDB) -> str:
    retrieved_docs = vectorstore_instance.similarity_search(query, k=3)
    context = "\n".join(retrieved_docs)
    prompt = f"Based on the following context, answer the question: {query}\n\nContext:\n{context}"
    return llm_instance.invoke(prompt)

def multi_step_rag(query: str, llm_instance: MockLLM, vectorstore_instance: MockChromaDB) -> str:
    initial_retrieval = vectorstore_instance.similarity_search(query, k=2)
    initial_context = "\n".join(initial_retrieval)
    refined_query_prompt = f"Given the query: '{query}' and initial context: '{initial_context}', identify key terms or sub-questions for further search."
    refined_terms = llm_instance.invoke(refined_query_prompt)
    secondary_retrieval = vectorstore_instance.similarity_search(refined_terms, k=3)
    secondary_context = "\n".join(secondary_retrieval)
    final_prompt = f"Synthesize a comprehensive answer for the query: '{query}', using the initial context:\n{initial_context}\n\nAnd additional information from refined search:\n{secondary_context}"
    return llm_instance.invoke(final_prompt)

def adaptive_rag_system(query: str, llm_instance: MockLLM, vectorstore_instance: MockChromaDB) -> Dict[str, str]:
    complexity = classify_query_complexity(query)
    response = ""
    strategy_used = ""

    if complexity == "simple":
        response = no_retrieval_qa(query, llm_instance)
        strategy_used = "No Retrieval (LLM-only)"
    elif complexity == "medium":
        response = single_step_rag(query, llm_instance, vectorstore_instance)
        strategy_used = "Single-step Retrieval-Augmented Generation"
    elif complexity == "complex":
        response = multi_step_rag(query, llm_instance, vectorstore_instance)
        strategy_used = "Multi-step Iterative Retrieval-Augmented Generation"
    else:
        response = "Could not determine query complexity."
        strategy_used = "Unknown"

    return {"response": response, "strategy": strategy_used, "complexity": complexity}

app = FastAPI()

@app.get("/classify")
async def get_complexity_api(query: str):
    complexity = classify_query_complexity(query)
    return {"query": query, "complexity": complexity}

@app.post("/query")
async def process_query_api(query_data: Dict[str, str]):
    query = query_data.get("query", "")
    if not query:
        return {"error": "Query cannot be empty."}
    result = adaptive_rag_system(query, llm, vectorstore)
    return result

def streamlit_app():
    st.set_page_config(page_title="Adaptive Medical Information Assistant")
    st.title("🩺 Adaptive Medical Information Assistant")
    st.write("This assistant dynamically selects the best retrieval strategy based on your query complexity.")

    user_query = st.text_area("Enter your medical query here:", height=100)

    if st.button("Get Answer"):
        if user_query:
            st.info("Processing your query...")
            result = adaptive_rag_system(user_query, llm, vectorstore)
            st.subheader("Answer:")
            st.write(result["response"])
            st.markdown(f"---")
            st.markdown(f"**Detected Complexity:** `{result['complexity'].upper()}`")
            st.markdown(f"**Strategy Used:** `{result['strategy']}`")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    # To run Streamlit UI: `streamlit run adaptive_medical_assistant.py`
    # To run FastAPI backend: `uvicorn adaptive_medical_assistant:app --host 0.0.0.0 --port 8000 --reload`
    # For simplicity in a single file, we directly run Streamlit by default.
    # For a real application, you would typically run FastAPI and Streamlit separately.
    streamlit_app()
