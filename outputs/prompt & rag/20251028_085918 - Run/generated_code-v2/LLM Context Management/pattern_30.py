import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, create_stuff_documents_chain
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("OpenAI API key not found. Please set it in your .env file or Streamlit secrets.")
        st.stop()

st.set_page_config(page_title="Intelligent Customer Support Chatbot", layout="centered")
st.title("Intelligent Customer Support Chatbot with Long-Term Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "lc_chat_history" not in st.session_state:
    st.session_state.lc_chat_history = []

@st.cache_resource
def get_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embedding_model()

@st.cache_resource
def get_vectorstore():
    chroma_instance = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="customer_support_history_v3")
    chroma_instance.persist()
    return chroma_instance

vectorstore = get_vectorstore()

@st.cache_resource
def get_llm():
    return ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

llm = get_llm()

history_aware_retriever_chain = create_history_aware_retriever(
    llm,
    vectorstore.as_retriever(search_kwargs={"k": 5}),
    ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a concise standalone question that can be used to search for relevant information to answer the user's latest query.")
    ])
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant. Use the following retrieved context and the conversation history to answer the user's question concisely and helpfully. If the context does not contain the answer, state that you don't know or cannot find the information based on the provided context.\n\nRetrieved context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}")
])

document_stuffing_chain = create_stuff_documents_chain(llm, rag_prompt)

full_rag_chain = create_retrieval_chain(history_aware_retriever_chain, document_stuffing_chain)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("How can I help you today?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.lc_chat_history.append(HumanMessage(content=prompt))

    with st.spinner("Thinking..."):
        try:
            response = full_rag_chain.invoke(
                {"input": prompt, "chat_history": st.session_state.lc_chat_history}
            )
            ai_response = response["answer"]
        except Exception as e:
            ai_response = f"An error occurred: {e}. Please ensure your OpenAI API key is set correctly."
            st.error(ai_response)
            st.session_state.lc_chat_history.pop()

    with st.chat_message("assistant"):
        st.markdown(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.session_state.lc_chat_history.append(AIMessage(content=ai_response))

    try:
        combined_turn_text = f"User asked: {prompt}\nAssistant responded: {ai_response}"
        vectorstore.add_documents([
            Document(page_content=combined_turn_text, metadata={"source": "conversation_turn"})
        ])
        vectorstore.persist()
        st.toast("Long-term memory updated!")
    except Exception as e:
        st.warning(f"Failed to update long-term memory: {e}")