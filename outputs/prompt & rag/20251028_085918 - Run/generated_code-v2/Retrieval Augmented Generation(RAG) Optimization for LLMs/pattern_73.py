import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import gradio as gr

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# 1. Knowledge Base (Simulated Medical Documents)
medical_documents = [
    "A patient presents with high fever, persistent cough, and fatigue. Initial tests show elevated white blood cell count. Consider bacterial pneumonia or severe influenza.",
    "Bacterial pneumonia treatment typically involves broad-spectrum antibiotics like Azithromycin or Amoxicillin. Monitor for antibiotic resistance.",
    "Severe influenza often requires antiviral medications such as Oseltamivir (Tamiflu) if administered within 48 hours of symptom onset. Rest and hydration are crucial.",
    "Recent research on genetic markers for respiratory diseases indicates that individuals with a specific gene variant (SNP-RS123) show increased susceptibility to severe inflammatory responses in viral infections.",
    "A 65-year-old male with a history of hypertension and Type 2 diabetes presents with shortness of breath and chest pain. ECG shows ST depression. Acute Myocardial Infarction is suspected.",
    "Treatment for Acute Myocardial Infarction includes immediate reperfusion therapy (angioplasty or thrombolysis), antiplatelet drugs (aspirin), and beta-blockers. Drug interaction between beta-blockers and certain diabetes medications can lower blood sugar.",
    "Patient records for John Doe: 58 years old, no known allergies, diagnosed with asthma 10 years ago, currently on Salbutamol inhaler. Recent blood work shows slightly elevated creatinine levels.",
    "Elevated creatinine can indicate kidney dysfunction. It's important to adjust dosages of renally excreted drugs, especially in elderly patients or those with pre-existing conditions like diabetes or hypertension."
]

# Initialize Embedding Model and Vector Store
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Convert strings to Document objects for Chroma
formatted_docs = [Document(page_content=doc) for doc in medical_documents]
vectorstore = Chroma.from_documents(formatted_docs, embeddings)
retriever = vectorstore.as_retriever()

# 2. Prompts for Multi-step RAG

# Prompt for initial query rewriting/decomposition
query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical query rewriter. Your goal is to rephrase complex medical questions into concise search queries for a retrieval system, or decompose it into simpler sub-queries."),
    ("human", "Original Query: {query}\nProvide the most effective search query/sub-queries that would retrieve relevant medical information. Focus on keywords and core concepts."),
])
query_rewriter_chain = query_rewrite_prompt | llm | StrOutputParser()

# Prompt for iterative reasoning and follow-up query generation
iterative_reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent medical diagnostic assistant. Your task is to analyze medical information iteratively to arrive at a comprehensive diagnosis and treatment plan. If you have enough information, provide a final answer; otherwise, suggest a follow-up query."),
    ("human", "Original Complex Query: {original_query}\nCurrent Context and Previous Findings: {current_context}\nNewly Retrieved Medical Documents: {retrieved_documents}\n\nBased on the above, perform the following steps:\n1. Synthesize the new documents with the current context.\n2. Provide an intermediate finding or thought process.\n3. If more information is needed to fully answer the original complex query, formulate a precise FOLLOW_UP_QUERY.\n4. If you have enough information to provide a comprehensive answer, state 'FINAL_ANSWER:' followed by the complete diagnostic answer.\n\nFormat your response as:\nIntermediate Thought: [Your thought process and intermediate findings]\n[Optional] FOLLOW_UP_QUERY: [Your next query for the retriever]\n[OR] FINAL_ANSWER: [Your comprehensive answer]"),
])
iterative_reasoning_chain = iterative_reasoning_prompt | llm | StrOutputParser()

# 3. Multi-step RAG Logic
def multi_step_rag_diagnosis(original_query: str, max_iterations: int = 3) -> str:
    current_context = f"User's initial query: {original_query}\n"
    current_query_for_retrieval = original_query
    final_answer = "" 

    for i in range(max_iterations):
        # Step 1: Query Rewriting/Decomposition (if not first iteration, or initial complex query)
        if i == 0:
            # For the very first step, use the original query as the initial retrieval query.
            # The LLM will process the full original query during the reasoning step.
            search_query = original_query
        else:
            # Use the LLM to refine the current_query_for_retrieval
            search_query = query_rewriter_chain.invoke({"query": current_query_for_retrieval})
            current_context += f"\nIteration {i}: LLM refined query for retrieval: {search_query}\n"

        # Step 2: Retrieval
        retrieved_docs = retriever.invoke(search_query)
        retrieved_docs_content = "\n".join([doc.page_content for doc in retrieved_docs])
        current_context += f"Retrieved documents from iteration {i+1}:\n{retrieved_docs_content}\n"

        # Step 3: Iterative Reasoning and Follow-up Query Generation
        llm_response = iterative_reasoning_chain.invoke({
            "original_query": original_query,
            "current_context": current_context,
            "retrieved_documents": retrieved_docs_content
        })

        current_context += f"LLM's thought and response from iteration {i+1}:\n{llm_response}\n"

        if "FINAL_ANSWER:" in llm_response:
            final_answer = llm_response.split("FINAL_ANSWER:", 1)[1].strip()
            break
        elif "FOLLOW_UP_QUERY:" in llm_response:
            parts = llm_response.split("FOLLOW_UP_QUERY:", 1)
            current_query_for_retrieval = parts[1].strip()
        else:
            # If neither FINAL_ANSWER nor FOLLOW_UP_QUERY, assume it's just an intermediate thought
            current_query_for_retrieval = current_query_for_retrieval # Keep the last query or original if no new one

    if not final_answer:
        # If max iterations reached without a final answer tag, synthesize from current context
        final_answer = f"Could not reach a definitive final answer within {max_iterations} iterations. Here is the accumulated information and latest thoughts:\n{current_context}"

    return final_answer

# 4. Gradio Interface
iface = gr.Interface(
    fn=multi_step_rag_diagnosis,
    inputs=gr.Textbox(lines=5, label="Enter your complex medical query"),
    outputs=gr.Textbox(lines=10, label="Diagnostic Assistant Response"),
    title="Intelligent Medical Diagnostic Assistant (Multi-step RAG)",
    description="Ask complex, multi-hop medical questions and receive iteratively reasoned diagnostic assistance.",
    examples=[
        ["A patient presents with high fever, cough, and fatigue. What are the most probable diagnoses and relevant treatments, considering recent research on genetic markers for respiratory infections?"],
        ["A 65-year-old male with a history of hypertension and Type 2 diabetes has shortness of breath and chest pain. What are the suspected conditions, treatment protocols, and potential drug interactions?"],
        ["John Doe, 58, has asthma and elevated creatinine. What are the implications for drug dosages and kidney function, especially considering his age?"]
    ]
)

iface.launch(share=True)
