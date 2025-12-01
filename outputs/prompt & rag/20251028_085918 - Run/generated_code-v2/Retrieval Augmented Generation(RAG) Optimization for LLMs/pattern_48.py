import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please create a .env file with your key.")
    st.stop()

DATA_DIR = "data"
CHROMA_DB_PATH = "./chroma_db"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    with open(os.path.join(DATA_DIR, "faq_billing.txt"), "w") as f:
        f.write("Q: How can I check my current bill?\nA: You can check your current bill by logging into your online account on our website or by using our mobile app. Navigate to the 'Billing' section.\n\nQ: What payment methods do you accept?\nA: We accept credit/debit cards, direct debit, and PayPal for bill payments.")
    with open(os.path.join(DATA_DIR, "service_plans.txt"), "w") as f:
        f.write("Our standard internet plan offers 100 Mbps download speed and 20 Mbps upload speed for $50/month.\nOur premium plan offers 500 Mbps download speed and 100 Mbps upload speed for $80/month. Both include unlimited data.")
    with open(os.path.join(DATA_DIR, "troubleshooting_internet.txt"), "w") as f:
        f.write("If your internet is not working, first try restarting your router and modem. Unplug them for 30 seconds, then plug them back in. Wait a few minutes for them to reconnect. If the issue persists, check all cable connections. Ensure no cables are loose or damaged. You can also visit our support portal for more detailed troubleshooting guides.")
    st.warning(f"Created dummy data files in '{DATA_DIR}'. Please replace them with your actual telecommunications knowledge base documents.")

st.set_page_config(page_title="Teleco Chatbot", page_icon="🤖")
st.header("🤖 Teleco Customer Support Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you with your telecommunications queries today?"}]

@st.cache_resource(show_spinner=False)
def get_index():
    with st.spinner("Loading and indexing knowledge base..."):
        llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
        )

        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = db.get_or_create_collection("telecom_kb")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        if len(chroma_collection.get()["ids"]) > 0:
            st.info("Loading existing knowledge base index.")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                service_context=service_context
            )
        else:
            st.info("Building new knowledge base index. This may take a moment.")
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context,
                vector_store=vector_store
            )

        return index

index = get_index()

chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about our services..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})