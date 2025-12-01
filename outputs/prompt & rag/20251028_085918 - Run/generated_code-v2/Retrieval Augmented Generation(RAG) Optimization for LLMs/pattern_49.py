"""
This script implements a Medical Diagnostic Assistant using the Interleaved Retrieval guided by Chain-of-Thought (IRCoT) pattern.
It leverages Streamlit for the UI, LangChain/LangGraph for orchestration, OpenAI for the LLM,
Sentence-Transformers for embeddings, and ChromaDB for the vector store.

To run this script:
1.  Install necessary libraries: `pip install streamlit langchain openai chromadb sentence-transformers pypdf numpy python-dotenv`
2.  Create a `.env` file in the same directory with your OpenAI API key: `OPENAI_API_KEY='your_api_key_here'`
3.  Run the Streamlit app: `streamlit run medical_diagnostic_assistant.py`

Note: For a real-world application, consider a more robust medical knowledge base and potentially a fine-tuned medical LLM.
This example uses a simple, simulated document set.
"""

import os
from dotenv import load_dotenv
import streamlit as st
from typing import List, Dict, Any, Union

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangGraph specific imports (simplified for a basic RAG chain, full IRCoT would involve a more complex graph)
from langgraph.graph import StateGraph, END

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file. Please set it.")
    st.stop()

# --- LLM and Embeddings Initialization ---
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.7, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# --- Simulated Medical Knowledge Base --- 
# In a real scenario, load from actual medical PDFs, databases, etc.
def get_medical_docs() -> List[Document]:
    docs_text = [
        "Migraine is a severe headache often accompanied by throbbing pain on one side of the head, nausea, vomiting, and extreme sensitivity to light and sound. Triggers can include stress, certain foods, and hormonal changes. Treatment often involves pain relievers, triptans, and preventative medications.",
        "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). The body either doesn't produce enough insulin, or it resists the effects of insulin. Symptoms include increased thirst, frequent urination, increased hunger, and blurred vision. Management involves diet, exercise, and medication like metformin.",
        "Hypertension (high blood pressure) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Risk factors include obesity, lack of physical activity, and a high-sodium diet. Treatment often includes lifestyle changes and medication like ACE inhibitors or diuretics.",
        "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon on the lower right side of your abdomen. Symptoms typically include sudden pain that begins around your navel and shifts to your lower right abdomen, nausea, vomiting, and fever. It usually requires surgical removal of the appendix (appendectomy)."]
    return [Document(page_content=text, metadata={"source": f"medical_text_{i+1}"}) for i, text in enumerate(docs_text)]

medical_docs = get_medical_docs()

# --- Vector Store (ChromaDB) Initialization ---
def initialize_vectorstore(docs: List[Document]) -> Chroma:
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    return vectorstore

vectorstore = initialize_vectorstore(medical_docs)
retriever = vectorstore.as_retriever()

# --- IRCoT-inspired Chain Definitions (Simplified LangChain RAG for demonstration) ---

# 1. Contextualize Question Prompt (CoT aspect: generate better search queries)
contextualize_q_system_prompt = (
    "You are a medical assistant. Given a chat history and the latest user question "
    "which might reference context in the chat history, formulate a standalone question "
    "which can be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 2. Answer Generation Prompt (CoT aspect: reason with retrieved docs)
qa_system_prompt = (
    "You are a medical diagnostic assistant. Your goal is to help clinicians by answering "
    "complex medical questions based on the provided patient context and retrieved medical knowledge. "
    "Provide a detailed, evidence-based answer. Clearly state your reasoning steps (Chain-of-Thought) "
    "and cite any retrieved sources. If you don't have enough information, state that you cannot fully "
    "answer the question with the available information, and suggest what additional information might be needed."
    "\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

# --- Streamlit Application --- 

st.set_page_config(page_title="Medical Diagnostic Assistant (IRCoT)", layout="wide")
st.title("🩺 Medical Diagnostic Assistant (IRCoT)")
st.markdown("Ask complex medical questions and get evidence-based answers with Chain-of-Thought reasoning.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a medical question about a patient case..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare chat history for LangChain
    chat_history_lc = []
    for msg in st.session_state.messages[:-1]: # Exclude the current user prompt
        if msg["role"] == "user":
            chat_history_lc.append(HumanMessage(content=msg["content"]))
        else:
            chat_history_lc.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking and retrieving medical knowledge..."):
            # Invoke the RAG chain
            response = rag_chain.invoke({"input": prompt, "chat_history": chat_history_lc})
            
            # Extract response and sources
            answer = response["answer"]
            # LangChain's create_retrieval_chain directly returns documents in 'context' within the answer
            # For a more explicit IRCoT, we would manage intermediate steps in LangGraph
            retrieved_docs_info = ""
            if "context" in response and response["context"]:
                sources = set([doc.metadata.get("source", "Unknown Source") for doc in response["context"]])
                retrieved_docs_info = "\n\n**Retrieved Sources:**\n" + "\n".join([f"- {s}" for s in sources])
            
            full_response = answer + retrieved_docs_info
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Explanation of IRCoT Integration (Commented out LangGraph advanced structure) ---
# The current implementation uses LangChain's `create_history_aware_retriever` and `create_retrieval_chain`
# which embody a simpler form of interleaved reasoning and retrieval:
# 1. The `history_aware_retriever` uses the LLM to 'reason' about the chat history and user input
#    to generate a better search query (a form of CoT guiding retrieval).
# 2. The `create_stuff_documents_chain` then uses the LLM to 'reason' over the retrieved documents
#    and the original question to formulate the final answer (retrieval guiding CoT planning).

# For a more explicit and complex IRCoT with multiple, dynamic retrieval/reasoning loops,
# LangGraph would be used to define a custom state machine. 
# For example, a LangGraph structure might look like this:

# class GraphState(TypedDict):
#     question: str
#     chat_history: List[Union[HumanMessage, AIMessage]]
#     retrieved_documents: List[Document]
#     reasoning_steps: List[str]
#     final_answer: str

# def retrieve(state):
#     # Logic to retrieve documents based on the current `question` or `reasoning_steps`
#     ...
#     return {"retrieved_documents": docs}

# def generate_cot(state):
#     # LLM generates CoT steps and potentially new retrieval queries
#     # based on `question` and `retrieved_documents`
#     ...
#     return {"reasoning_steps": new_steps, "question": potentially_new_query}

# def decide_to_continue(state):
#     # LLM or rule-based logic to decide if more retrieval/reasoning is needed
#     ...
#     return "continue" or "end"

# workflow = StateGraph(GraphState)
# workflow.add_node("retrieve", retrieve)
# workflow.add_node("generate_cot", generate_cot)
# workflow.add_conditional_edges(
#     "generate_cot",
#     decide_to_continue,
#     {
#         "continue": "retrieve", # Loop back to retrieval
#         "end": END
#     }
# )
# workflow.set_entry_point("generate_cot") # Start with initial reasoning
# app = workflow.compile()

# This simplified RAG chain provides a good starting point for demonstrating the core principles.
