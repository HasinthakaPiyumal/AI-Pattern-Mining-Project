import streamlit as st
import os
from dotenv import load_dotenv
from loguru import logger
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

logger.add("medical_research_assistant.log", rotation="500 MB")

# --- Pydantic Models ---
class SubQueries(BaseModel):
    sub_queries: List[str] = Field(description="List of decomposed sub-queries.")


# --- LLM and Embedding Initialization ---
@st.cache_resource
def get_llms():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    decomposition_llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
    return llm, decomposition_llm

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-ada-002")

llm, decomposition_llm = get_llms()
embeddings = get_embeddings()

# --- ChromaDB Setup ---
@st.cache_resource
def get_vectorstore(embeddings_model):
    # For demonstration, we'll use a dummy document. In a real app, load actual medical data.
    if not os.path.exists("medical_docs"): # Create a dummy directory for docs
        os.makedirs("medical_docs")
    with open("medical_docs/dummy_research.txt", "w") as f:
        f.write("Clinical trial results for a new drug 'MediCure' for Stage 2 breast cancer showed a 70% remission rate in patients over 60. Common side effects reported include mild nausea, fatigue, and occasional headaches. Another study on 'CancerX' for Stage 3 lung cancer highlighted a 50% survival rate over 5 years. Side effects include hair loss and severe fatigue. Latest research on 'ImmunoBoost' for autoimmune diseases shows promising results in reducing inflammation in 75% of patients, with side effects limited to skin rashes. New developments in gene therapy for genetic disorders are showing initial success in trials, with no significant adverse effects reported so far.")
    
    loader = TextLoader("medical_docs/dummy_research.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    logger.info(f"Loaded {len(docs)} document chunks for ChromaDB.")
    vectorstore = Chroma.from_documents(docs, embeddings_model, persist_directory="./chroma_db")
    return vectorstore

vectorstore = get_vectorstore(embeddings)
retriever = vectorstore.as_retriever()

# --- LangChain Chains ---

# 1. Query Decomposition Chain
decomposition_parser = PydanticOutputParser(pydantic_object=SubQueries)
decomposition_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at breaking down complex multi-hop questions into a list of simpler, single-hop sub-questions. Each sub-question should be answerable directly using a knowledge base."),
    ("human", "Decompose the following complex query into simple sub-questions, outputting a JSON list of strings:\nQuery: {query}\n{format_instructions}"),
])

decomposition_chain = (
    decomposition_prompt.partial(format_instructions=decomposition_parser.get_format_instructions())
    | decomposition_llm
    | decomposition_parser
)

# 2. Sub-query Answering (RAG) Chain
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical research assistant. Use the following retrieved context to answer the question accurately and concisely. If the information is not in the context, state that you cannot answer based on the provided information.\n\nContext: {context}"),
    ("human", "Question: {question}")
])

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
)

# 3. Answer Synthesis Chain
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert medical researcher. Synthesize the following individual answers to sub-questions into a comprehensive, coherent, and well-structured final answer to the original complex query. Ensure all relevant information is included and presented clearly. Do not invent information."),
    ("human", "Original Query: {original_query}\n\nIndividual Sub-Answers:\n{sub_answers}\n\nSynthesized Answer:")
])

synthesis_chain = synthesis_prompt | llm


# --- Main Orchestration Function ---
def medical_research_assistant(query: str) -> str:
    logger.info(f"Received complex query: {query}")
    st.session_state.intermediate_steps = {}

    # Step 1: Query Decomposition
    st.session_state.status = "Decomposing query..."
    st.rerun()
    decomposed_queries = decomposition_chain.invoke({"query": query}).sub_queries
    logger.info(f"Decomposed queries: {decomposed_queries}")
    st.session_state.intermediate_steps["Decomposed Queries"] = decomposed_queries

    # Step 2: Iterative Sub-query Answering
    sub_answers = []
    for i, sub_q in enumerate(decomposed_queries):
        st.session_state.status = f"Answering sub-query {i+1}/{len(decomposed_queries)}: {sub_q}..."
        st.rerun()
        answer = rag_chain.invoke(sub_q).content
        logger.info(f"Answer for '{sub_q}': {answer}")
        sub_answers.append(f"Sub-question {i+1} ({sub_q}): {answer}")
        st.session_state.intermediate_steps[f"Answer for '{sub_q}'"] = answer
    
    formatted_sub_answers = "\n".join(sub_answers)
    st.session_state.intermediate_steps["Formatted Sub-Answers"] = formatted_sub_answers

    # Step 3: Answer Synthesis
    st.session_state.status = "Synthesizing final answer..."
    st.rerun()
    final_answer = synthesis_chain.invoke({"original_query": query, "sub_answers": formatted_sub_answers}).content
    logger.info(f"Final synthesized answer: {final_answer}")
    st.session_state.status = "Done!"
    return final_answer


# --- Streamlit UI ---
st.set_page_config(page_title="Medical Research Assistant", layout="wide")
st.title("🔬 Medical Research Assistant")
st.markdown("Ask complex medical research questions and get synthesized answers from our AI assistant.")

user_query = st.text_area(
    "Enter your complex medical research query here:",
    placeholder="e.g., What are the latest clinical trial results for a new drug treating Stage 2 breast cancer in patients over 60, and what are the common side effects reported?"
)

if "intermediate_steps" not in st.session_state:
    st.session_state.intermediate_steps = {}
if "status" not in st.session_state:
    st.session_state.status = "Ready"

st.sidebar.header("Application Status")
st.sidebar.info(st.session_state.status)

if st.button("Get Answer", type="primary"):
    if user_query:
        with st.spinner("Processing your query..."): # Spinner outside the main function
            final_response = medical_research_assistant(user_query)
        st.subheader("Final Synthesized Answer")
        st.write(final_response)
        
        st.subheader("Intermediate Steps")
        for step_name, step_output in st.session_state.intermediate_steps.items():
            st.expander(step_name).write(step_output)
    else:
        st.warning("Please enter a query.")

st.markdown("""
--- 
**Note:** This is a demonstration. For a real-world application, `ChromaDB` would be populated with extensive, up-to-date, and curated medical research documents, and LLM calls would be optimized for cost and performance. Ensure your `OPENAI_API_KEY` is set as an environment variable.
""")

