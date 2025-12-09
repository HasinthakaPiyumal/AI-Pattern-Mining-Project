import streamlit as st
import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer
import chromadb

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    MONGO_DB_CONNECTION_STRING = os.getenv("MONGO_DB_CONNECTION_STRING", "mongodb://localhost:27017/")
    CHROMA_COLLECTION_NAME = "customer_support_kb"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class CustomerHistoryManager:
    def __init__(self):
        self.history_db = {}

    def add_interaction(self, customer_id, query, response):
        if customer_id not in self.history_db:
            self.history_db[customer_id] = []
        self.history_db[customer_id].append({"query": query, "response": response, "timestamp": time.time()})

    def get_history(self, customer_id, limit=5):
        return self.history_db.get(customer_id, [])[-limit:]

class CustomerProfileManager:
    def __init__(self):
        self.profile_db = {}

    def get_profile(self, customer_id):
        return self.profile_db.get(customer_id, {"name": "Guest", "products": []})

    def update_profile(self, customer_id, profile_data):
        if customer_id not in self.profile_db:
            self.profile_db[customer_id] = {}
        self.profile_db[customer_id].update(profile_data)

class KnowledgeBaseManager:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=Config.CHROMA_COLLECTION_NAME)
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        self._initialize_kb()

    def _initialize_kb(self):
        docs = [
            "How to reset your password? Go to settings -> security -> reset password.",
            "My internet is not working. Try restarting your router. If it persists, check cable connections.",
            "How do I update my billing information? Log in to your account, navigate to 'Billing', and click 'Edit Payment Method'.",
            "What are the supported payment methods? We accept Visa, MasterCard, American Express, and PayPal.",
            "How can I contact customer support? You can reach us via phone at 1-800-SUPPORT or email at support@example.com."
        ]
        ids = [f"kb_doc_{i}" for i in range(len(docs))]
        embeddings = self.embedding_model.encode(docs).tolist()
        self.collection.add(
            documents=docs,
            embeddings=embeddings,
            metadatas=[{"source": "FAQ"} for _ in docs],
            ids=ids
        )

    def search_knowledge_base(self, query, n_results=3):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'distances']
        )
        return results['documents'][0] if results['documents'] else []

class FeedbackLogManager:
    def __init__(self):
        self.feedback_log = []

    def log_feedback(self, customer_id, query, response, feedback):
        self.feedback_log.append({
            "customer_id": customer_id,
            "query": query,
            "response": response,
            "feedback": feedback,
            "timestamp": time.time()
        })

class LLMAgent:
    def __init__(self, customer_id="default_customer"):
        self.llm = ChatOpenAI(openai_api_key=Config.OPENAI_API_KEY, model_name="gpt-3.5-turbo")
        self.customer_id = customer_id
        self.history_manager = CustomerHistoryManager()
        self.profile_manager = CustomerProfileManager()
        self.kb_manager = KnowledgeBaseManager()
        self.feedback_manager = FeedbackLogManager()

        if self.customer_id == "default_customer" and not self.profile_manager.get_profile(self.customer_id)["products"]:
            self.profile_manager.update_profile(self.customer_id, {"name": "Default User", "products": ["Product A", "Service X"]})

    def _construct_prompt(self, user_query):
        customer_profile = self.profile_manager.get_profile(self.customer_id)
        interaction_history = self.history_manager.get_history(self.customer_id)
        kb_results = self.kb_manager.search_knowledge_base(user_query)

        profile_str = f"Customer Name: {customer_profile.get('name', 'N/A')}, Products: {', '.join(customer_profile.get('products', []))}"

        history_str = ""
        if interaction_history:
            history_str = "Recent Interaction History:\n"
            for i, interaction in enumerate(interaction_history):
                history_str += f"  Q{i+1}: {interaction['query']}\n"
                history_str += f"  A{i+1}: {interaction['response']}\n"

        kb_str = ""
        if kb_results:
            kb_str = "Relevant Knowledge Base Articles:\n"
            for doc in kb_results:
                kb_str += f"  - {doc}\n"

        system_message = f"""You are an intelligent customer support agent.\nYour goal is to provide helpful, accurate, and personalized assistance.\nYou have access to the customer's profile, past interaction history, and a knowledge base.\n\n{profile_str}\n\n{history_str}\n\n{kb_str}\n\nBased on the above information and the current customer query, provide a comprehensive response.\nIf you need more information, ask clarifying questions.\n"""
        return system_message

    def get_response(self, user_query):
        system_prompt = self._construct_prompt(user_query)
        messages = [
            HumanMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        response = self.llm.invoke(messages).content
        self.history_manager.add_interaction(self.customer_id, user_query, response)
        return response

    def provide_feedback(self, query, response, feedback):
        self.feedback_manager.log_feedback(self.customer_id, query, response, feedback)

st.set_page_config(page_title="Intelligent Customer Support")
st.title("Intelligent Customer Support Agent")

if "agent" not in st.session_state:
    st.session_state.agent = LLMAgent(customer_id="test_customer_123")
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.get_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    if len(st.session_state.messages) > 1:
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            if st.button("👍"):
                st.session_state.agent.provide_feedback(prompt, response, "good")
                st.toast("Feedback 'Good' logged!")
        with col2:
            if st.button("👎"):
                st.session_state.agent.provide_feedback(prompt, response, "bad")
                st.toast("Feedback 'Bad' logged!")

st.sidebar.subheader("Memory Contents (for Demo)")
st.sidebar.markdown("**Customer Profile:**")
st.sidebar.json(st.session_state.agent.profile_manager.get_profile(st.session_state.agent.customer_id))
st.sidebar.markdown("**Interaction History:**")
st.sidebar.json(st.session_state.agent.history_manager.get_history(st.session_state.agent.customer_id, limit=3))
st.sidebar.markdown("**Feedback Log (last 3):**")
st.sidebar.json(st.session_state.agent.feedback_manager.feedback_log[-3:])
