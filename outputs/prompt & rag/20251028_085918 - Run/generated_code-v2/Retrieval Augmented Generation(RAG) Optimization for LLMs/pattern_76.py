import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import tqdm

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.0)

# Initialize Embedding Model
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Mock Medical Knowledge Base and Retriever Setup
medical_docs = [
    "GLP-1 receptor agonists like semaglutide and liraglutide have shown significant long-term efficacy in lowering HbA1c and reducing cardiovascular events in type 2 diabetes patients, especially those with pre-existing cardiovascular disease. Studies indicate a reduction in major adverse cardiovascular events (MACE).",
    "The safety profile of GLP-1 receptor agonists is generally good, with common side effects including gastrointestinal issues like nausea, vomiting, and diarrhea. Pancreatitis is a rare but serious concern. Long-term use has not been associated with increased risk of thyroid C-cell tumors in humans.",
    "GLP-1 receptor agonists consistently demonstrate a beneficial effect on body weight, leading to moderate to significant weight loss in most patients. This effect is independent of glycemic control.",
    "Regarding renal function, GLP-1 receptor agonists have shown renoprotective effects, including a reduction in albuminuria and a slowed decline in eGFR in patients with type 2 diabetes and chronic kidney disease. This benefit extends to patients with pre-existing cardiovascular disease.",
    "Type 2 diabetes patients with pre-existing cardiovascular disease benefit from therapies that reduce cardiovascular risk. GLP-1 RAs are recommended in this population.",
    "Long-term studies on semaglutide (e.g., SUSTAIN and PIONEER trials) confirm its sustained efficacy and safety, including cardiovascular and renal benefits. Liraglutide (LEADER trial) also showed similar positive outcomes."
]

# Chunking documents (simple for demonstration)
from langchain_core.documents import Document
documents = [Document(page_content=doc) for doc in medical_docs]

# Create Chroma vector store
vectorstore = Chroma.from_documents(documents, embeddings_model)
retriever = vectorstore.as_retriever()

# Query Decomposition Module
decomposition_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at breaking down complex medical queries into simpler, independent sub-questions. List each sub-question on a new line, numbered."),
    ("human", "Decompose the following complex query into simpler sub-questions:\n\n{query}")
])
decomposition_chain = decomposition_prompt | llm | StrOutputParser()

def decompose_query(complex_query: str) -> list[str]:
    raw_sub_queries = decomposition_chain.invoke({"query": complex_query})
    sub_queries = [q.strip() for q in raw_sub_queries.split('\n') if q.strip() and q.strip()[0].isdigit()]
    return sub_queries

# Sub-query Answering Module (RAG - Retrieval Augmented Generation)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the provided context only. Be concise and factual.\nContext: {context}"),
    ("human", "{input}")
])
document_chain = create_stuff_documents_chain(llm, rag_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def answer_sub_query(sub_query: str) -> str:
    response = retrieval_chain.invoke({"input": sub_query})
    return response["answer"]

# Answer Synthesis Module
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that synthesizes information to provide a comprehensive answer to an original complex query. Combine the provided original query and the answers to its sub-questions into a coherent and complete response."),
    ("human", "Original Complex Query: {original_query}\n\nAnswers to Sub-questions:\n{sub_query_answers}\n\nSynthesize a comprehensive answer:")
])
synthesis_chain = synthesis_prompt | llm | StrOutputParser()

def synthesize_answers(original_query: str, sub_query_answers: list[tuple[str, str]]) -> str:
    formatted_sub_answers = "\n".join([f"- {sq}: {ans}" for sq, ans in sub_query_answers])
    final_answer = synthesis_chain.invoke({"original_query": original_query, "sub_query_answers": formatted_sub_answers})
    return final_answer

# Main Application Logic
def medical_research_assistant(complex_query: str) -> str:
    print(f"\nProcessing complex query: {complex_query}")

    # 1. Query Decomposition
    print("\n--- Decomposing Query ---")
    sub_queries = decompose_query(complex_query)
    print("Decomposed into:")
    for i, sq in enumerate(sub_queries):
        print(f"  {i+1}. {sq}")

    # 2. Iterative Sub-query Answering
    print("\n--- Answering Sub-queries ---")
    sub_query_results = []
    for i, sq in enumerate(tqdm.tqdm(sub_queries, desc="Answering sub-queries")):
        print(f"\n  Answering sub-query {i+1}: {sq}")
        answer = answer_sub_query(sq)
        print(f"  Answer {i+1}: {answer}")
        sub_query_results.append((sq, answer))

    # 3. Answer Synthesis
    print("\n--- Synthesizing Final Answer ---")
    final_answer = synthesize_answers(complex_query, sub_query_results)
    print("\n--- Comprehensive Final Answer ---")
    return final_answer

if __name__ == "__main__":
    complex_clinical_query = (
        "What are the long-term efficacy and safety profiles of GLP-1 receptor agonists in patients with type 2 diabetes "
        "and pre-existing cardiovascular disease, considering their impact on renal function and body weight?"
    )

    final_response = medical_research_assistant(complex_clinical_query)
    print(final_response)

    print("\n\n--- Example 2 ---")
    complex_clinical_query_2 = (
        "How do ACE inhibitors compare to ARBs in reducing proteinuria in hypertensive patients with chronic kidney disease, "
        "and what are their common side effects?"
    )
    # Note: This query will likely yield less specific results due to mock data not containing info on ACEi/ARBs.
    # The purpose is to demonstrate the decomposition and synthesis, even with limited context.
    final_response_2 = medical_research_assistant(complex_clinical_query_2)
    print(final_response_2)

    # Cleanup vectorstore (optional, if you want to clear the data after run)
    # vectorstore.delete_collection()
