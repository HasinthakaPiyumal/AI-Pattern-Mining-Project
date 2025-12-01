from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

import os

# 1. LLM Serving Layer (vLLM simulation via OpenAI-compatible API)
# Assume vLLM is running at this endpoint, e.g., using `python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.1`
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")

llm = ChatOpenAI(
    openai_api_base=VLLM_API_BASE,
    model=VLLM_MODEL_NAME,
    api_key="sk-no-key-required",
    temperature=0.7
)

# 2. Retrieval Augmented Generation (RAG) Layer
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Dummy product/FAQ documents
documents = [
    "Product X: High-performance laptop with 16GB RAM and 512GB SSD. Ideal for gaming and professional use.",
    "Product X Troubleshooting: If Product X is not turning on, ensure it is fully charged and try holding the power button for 10 seconds.",
    "Product Y: Wireless earbuds with 24-hour battery life and noise cancellation. Perfect for music lovers on the go.",
    "Shipping Policy: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. Free shipping on orders over $50.",
    "Return Policy: Items can be returned within 30 days of purchase with original receipt. Some exclusions apply, check our website for details.",
    "How to reset password: Visit our website, click 'Forgot Password' on the login page, and follow the instructions sent to your email."
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents(text_splitter.split_text("\n".join(documents)))

vector_store = FAISS.from_documents(texts, embeddings_model)
retriever = vector_store.as_retriever()

# 3. Conversation Management
memory = ConversationBufferWindowMemory(k=5, return_messages=True, memory_key="chat_history", output_key="answer")

# 4. Core Application Logic (Customer Assistant)
# Prompt Template for contextualizing conversation history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if necessary and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Prompt Template for RAG
qa_system_prompt = (
    "You are a helpful customer support assistant for an e-commerce platform. "
    "Answer the user's questions based on the provided context and chat history. "
    "If you don't know the answer, politely state that you cannot provide the information. "
    "Keep your answers concise and helpful."
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
retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

# 5. API/Interface (Demonstration)
def get_customer_assistant_response(user_query: str) -> str:
    config = {"configurable": {"session_id": "test_session"}}
    
    # Retrieve chat history from memory
    chat_history = memory.load_memory_variables(config.get("configurable", {}).get("session_id", "default_session"))["chat_history"]

    # Invoke the RAG chain
    response = retrieval_chain.invoke({"input": user_query, "chat_history": chat_history}, config=config)
    
    # Update memory
    memory.save_context(
        config.get("configurable", {}).get("session_id", "default_session"),
        {"input": user_query, "output": response["answer"]}
    )
    
    return response["answer"]

if __name__ == "__main__":
    print("Welcome to the Smart Customer Support Assistant! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for contacting support. Goodbye!")
            break
        
        assistant_response = get_customer_assistant_response(user_input)
        print(f"Assistant: {assistant_response}")