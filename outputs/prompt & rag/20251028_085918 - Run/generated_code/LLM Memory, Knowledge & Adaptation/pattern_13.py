
import streamlit as st
import os
import time
from collections import defaultdict

# LangChain and related imports
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# Query Classifier imports (simple scikit-learn for demonstration)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

# --- Configuration --- #
# Set your OpenAI API key as an environment variable or directly here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY") # Replace with your key or ensure env var is set
if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
    st.warning("Please set your OPENAI_API_KEY in the environment variables or replace 'YOUR_OPENAI_API_KEY' in the code.")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_DIR = "./chroma_db"
CLASSIFIER_MODEL_PATH = "./query_classifier.joblib"

# --- Helper Functions and Classes --- #

class CustomerSupportAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        self.vectorstore = self._init_vector_store()
        self.retriever = self.vectorstore.as_retriever()
        
        self.query_classifier_pipeline = self._load_or_train_classifier()
        
        # LangChain conversational memory
        self.chat_history = defaultdict(ChatMessageHistory)
        
        # LangChain RAG chain
        self.rag_chain = self._setup_rag_chain()

    def _init_vector_store(self):
        """Initializes or loads the Chroma vector store."""
        if os.path.exists(CHROMA_DB_DIR):
            st.info("Loading existing knowledge base...")
            return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embedding_model)
        else:
            st.info("Initializing new knowledge base...")
            # For a fresh start, add some dummy documents
            dummy_docs = [
                "Our refund policy states that you can return items within 30 days for a full refund if the item is unused and in its original packaging.",
                "To reset your password, please visit our website and click on 'Forgot Password' link on the login page.",
                "Our customer support hours are Monday to Friday, 9 AM to 5 PM EST.",
                "We offer free standard shipping on all orders over $50. Expedited shipping options are available at an additional cost."
            ]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.create_documents(dummy_docs)
            vectorstore = Chroma.from_documents(docs, self.embedding_model, persist_directory=CHROMA_DB_DIR)
            vectorstore.persist()
            return vectorstore

    def _load_or_train_classifier(self):
        """Loads an existing classifier or trains a new one with dummy data."""
        if os.path.exists(CLASSIFIER_MODEL_PATH):
            st.info("Loading existing query classifier...")
            return joblib.load(CLASSIFIER_MODEL_PATH)
        else:
            st.info("Training new query classifier with dummy data...")
            # Dummy training data for query classification
            queries = [
                "I want to return a product.", "What is your return policy?", "Can I get my money back?",
                "How do I change my password?", "Forgot my login, help!", "Need to reset my account.",
                "What are your operating hours?", "When can I talk to support?", "Contact information.",
                "Where is my order?", "Shipping cost for my purchase.", "Expedited delivery options.",
                "I have a complex issue, can I speak to a human?", "Need to talk to a manager.", "Escalate my case."
            ]
            labels = [
                "refund_policy", "refund_policy", "refund_policy",
                "password_reset", "password_reset", "password_reset",
                "support_hours", "support_hours", "support_hours",
                "shipping_info", "shipping_info", "shipping_info",
                "human_escalation", "human_escalation", "human_escalation"
            ]
            
            classifier_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('svm', SVC())
            ])
            classifier_pipeline.fit(queries, labels)
            joblib.dump(classifier_pipeline, CLASSIFIER_MODEL_PATH)
            return classifier_pipeline

    def _setup_rag_chain(self):
        """Sets up the LangChain RAG conversational chain."""
        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an helpful AI customer support agent. Answer the user's questions based on the provided context and conversation history. If the context does not contain the answer, state that you don't know, and suggest human escalation for complex issues."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create a combined chain for RAG and conversation
        rag_chain = (
            RunnablePassthrough.assign(
                context=(lambda x: self.retriever.invoke(x["input"])) | (lambda docs: "\n\n".join([doc.page_content for doc in docs]))
            ) 
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def classify_query(self, query: str) -> str:
        """Classifies the incoming user query."""
        return self.query_classifier_pipeline.predict([query])[0]

    def process_query(self, user_id: str, query: str):
        """Processes a user query by classifying it and routing to the appropriate strategy."""
        current_chat_history = self.chat_history[user_id].messages
        
        classification = self.classify_query(query)
        st.sidebar.write(f"Query classified as: **{classification}**")

        response = ""
        if classification == "human_escalation":
            response = "I understand this is a complex issue. Let me connect you to a human agent who can provide more in-depth assistance."
        elif classification in ["refund_policy", "password_reset", "support_hours", "shipping_info"]:
            # Use RAG for these specific informational queries
            try:
                # LangChain's ConversationalRetrievalChain or a manual RAG chain
                # For simplicity, using the pre-setup rag_chain directly and managing history outside.
                response = self.rag_chain.invoke({"input": query, "chat_history": current_chat_history})
                if "I don't know" in response or "I cannot answer" in response: # Simple check for non-committal LLM answers
                    response += " If this isn't what you're looking for, please consider escalating to a human agent."
            except Exception as e:
                st.error(f"Error during RAG processing: {e}")
                response = "I'm having trouble retrieving specific information right now. Could you please rephrase, or would you like to speak to a human agent?"
        else:
            # Default to direct LLM answer for general inquiries not covered by specific classifications
            try:
                response = self.llm.invoke(prompt.format(input=query, chat_history=current_chat_history)).content
            except Exception as e:
                st.error(f"Error during direct LLM processing: {e}")
                response = "I'm experiencing an issue and cannot respond right now. Please try again later."
        
        # Update short-term memory (chat history)
        self.chat_history[user_id].add_user_message(query)
        self.chat_history[user_id].add_ai_message(response)
        
        return response

    def update_knowledge_base(self, documents_text: str):
        """Allows dynamic updates to the knowledge base."""
        st.info("Updating knowledge base...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_docs = text_splitter.create_documents([documents_text])
        
        # Add new documents to the existing vector store
        self.vectorstore.add_documents(new_docs)
        self.vectorstore.persist()
        
        # Re-initialize retriever if necessary (Chroma updates in place, but good practice)
        self.retriever = self.vectorstore.as_retriever()
        st.success("Knowledge base updated successfully!")

    def simulate_fine_tuning_update(self, new_policy_data: str):
        """Placeholder for simulating LLM fine-tuning and model update.
        In a real scenario, this would involve training with TRL/PEFT and loading a new model.
        """
        st.info(f"Simulating LLM fine-tuning with new policy data: '{new_policy_data[:50]}...'\n")
        st.write("This would involve using libraries like TRL and PEFT to adapt the base LLM.")
        st.write("Upon completion, a newly fine-tuned model would be loaded for inference.")
        st.success("LLM fine-tuning simulation complete. Model parameters conceptually updated.")


# --- Streamlit UI --- #
def main():
    st.set_page_config(page_title="Adaptive Customer Support Agent", layout="wide")
    st.title("🧠 Adaptive Customer Support Agent")

    # Initialize the agent (cached to avoid re-initializing on every rerun)
    @st.cache_resource
    def get_agent():
        return CustomerSupportAgent(openai_api_key=OPENAI_API_KEY)
    
    agent = get_agent()

    if "user_id" not in st.session_state:
        st.session_state.user_id = str(time.time())

    st.sidebar.header("Agent Controls")
    
    # Knowledge Base Update Section
    st.sidebar.subheader("Dynamic Knowledge Update")
    new_kb_text = st.sidebar.text_area("Enter new policy/FAQ text to add to KB:", height=150)
    if st.sidebar.button("Update Knowledge Base"):
        if new_kb_text:
            agent.update_knowledge_base(new_kb_text)
        else:
            st.sidebar.warning("Please enter some text to update the knowledge base.")
            
    # LLM Fine-tuning Simulation Section
    st.sidebar.subheader("LLM Adaptation (Fine-tuning Simulation)")
    new_ft_data = st.sidebar.text_area("Enter new data for LLM fine-tuning simulation:", "E.g., Updated return conditions for electronics.", height=100)
    if st.sidebar.button("Simulate Fine-tuning"):
        agent.simulate_fine_tuning_update(new_ft_data)

    # Chat Interface
    st.header("Chat with the Agent")

    # Display chat messages from history on app rerun
    for message in agent.chat_history[st.session_state.user_id].messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # React to user input
    if prompt := st.chat_input("How can I help you today?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.process_query(st.session_state.user_id, prompt)
                st.markdown(response)

    st.sidebar.markdown("---")
    st.sidebar.info("This is a demonstration of an adaptive LLM agent. Features include dynamic query classification, RAG-based knowledge retrieval, conversational memory, and simulated knowledge/model updates.")

if __name__ == "__main__":
    main()
