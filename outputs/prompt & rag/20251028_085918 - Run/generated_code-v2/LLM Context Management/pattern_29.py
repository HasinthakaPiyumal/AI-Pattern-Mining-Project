
import streamlit as st
import os
from langchain_community.llms import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set your OpenAI API key
# It's recommended to set this as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Ensure the OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# --- Memory System Initialization ---
@st.cache_resource
def get_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_db(embeddings):
    # Define a persistent directory for Chroma DB
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # Initialize ChromaDB with a default collection if it doesn't exist
    # Or load an existing one
    # We'll create a new collection for each run for simplicity, or load if exists
    # For a real app, you'd manage collection names more carefully.
    try:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # Check if the collection is empty. If so, add a dummy entry to initialize.
        if len(vectorstore.get(include=['documents'])['documents']) == 0:
             vectorstore.add_texts(["This is an initial memory entry."], metadatas=[{"source": "system"}])
        return vectorstore
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        st.stop()

embeddings = get_embedding_model()
vectorstore = get_chroma_db(embeddings)

# --- LLM Initialization ---
@st.cache_resource
def get_llm():
    return ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

llm = get_llm()

# --- RAG Chain Setup ---
# Define a custom prompt to leverage retrieved context
rag_prompt_template = """
Your are an intelligent customer support chatbot. Use the following context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Previous Conversation History and Customer Preferences:
{context}

Question: {question}
Answer:"""

rag_prompt = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])

def create_qa_chain():
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False, # Set to True if you want to see the retrieved docs
        chain_type_kwargs={"prompt": rag_prompt}
    )
    return qa_chain

qa_chain = create_qa_chain()

# --- Streamlit UI ---
st.title("Intelligent Customer Support Chatbot")
st.write("I remember our past conversations and your preferences!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get response from RAG chain
        response = qa_chain({"query": prompt})
        full_response = response["result"]
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Augment memory with the current interaction
    # We store the user's prompt and the chatbot's response as a single document
    # This helps the chatbot remember the context of questions and answers.
    memory_content = f"User asked: {prompt}\nChatbot replied: {full_response}"
    try:
        vectorstore.add_texts([memory_content], metadatas=[{"source": "conversation_memory"}])
        vectorstore.persist() # Save changes to disk
    except Exception as e:
        st.error(f"Error storing conversation in memory: {e}")

