import os
from collections import deque

from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# --- Configuration ---
# Set your OpenAI API key here or as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Check if API key is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")

# --- 1. Core Chatbot Logic (LLM) ---
llm = OpenAI(temperature=0)

# --- 2. Working Memory (Short-Term Memory) ---
# Langchain's ConversationBufferMemory handles the in-context memory for the LLM
short_term_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- 3. Long-Term Memory (Retrieval-Augmented Generation - RAG) ---
# Dummy Product Knowledge Base
long_term_knowledge_base = [
    "Product A is a premium smartphone with a 6.7-inch display and a 108MP camera. It costs $999.",
    "Product B is a budget laptop with a 14-inch screen and 8GB RAM. It costs $499.",
    "To reset your password, visit our website and click 'Forgot Password' on the login page.",
    "Our return policy allows returns within 30 days of purchase for a full refund, provided the item is in its original condition.",
    "Shipping usually takes 3-5 business days for domestic orders and 7-14 business days for international orders.",
    "Our customer support hours are Monday to Friday, 9 AM to 5 PM EST.",
    "Troubleshooting steps for slow internet include restarting your router, checking cable connections, and contacting your ISP.",
    "Product C is a smart home device that monitors air quality and can be controlled via a mobile app. It costs $129."
]

# Initialize Embedding Model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a dummy vector store for long-term memory
# In a real application, this would be loaded from a persistent store
vectorstore = Chroma.from_texts(long_term_knowledge_base, embeddings, collection_name="customer_support_knowledge")
long_term_retriever = vectorstore.as_retriever()

# --- 4. Orchestration and Flow ---
# Combine LLM, short-term memory, and long-term retriever into a conversational chain
chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=long_term_retriever,
    memory=short_term_memory,
    combine_docs_chain_kwargs={"prompt": """You are a helpful customer support assistant. Answer the user's questions based on the provided context and the chat history.
If you don't know the answer, state that you don't know, and suggest contacting a human agent.

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:"""}
)

# --- 5. User Interface (Command-Line) ---
def main():
    print("Welcome to the Intelligent Customer Support Chatbot!\n")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for chatting! Goodbye.")
            break

        try:
            # The chat_chain handles querying the LLM, updating short-term memory, and potentially using long-term retrieval.
            result = chat_chain({"question": user_input})
            bot_response = result["answer"]
            print(f"Bot: {bot_response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or contact a human agent if the issue persists.")

if __name__ == "__main__":
    main()