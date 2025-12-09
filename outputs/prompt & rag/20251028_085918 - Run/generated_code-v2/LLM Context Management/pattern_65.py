import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 0. Configuration and Environment Setup ---
# Set your OpenAI API key here, or ensure it's in your environment variables
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

if "OPENAI_API_KEY" not in os.environ:
    st.error("OPENAI_API_KEY environment variable not set. Please set it in your environment or directly in the script.")
    st.stop()

# --- 1. Knowledge Base ---
KNOWLEDGE_BASE = {
    "shipping": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.",
    "returns": "You can return items within 30 days of purchase. Items must be in original condition with tags.",
    "products": "We sell a variety of electronics, apparel, and home goods.",
    "contact": "You can reach customer support at support@example.com or call us at 1-800-123-4567.",
    "payment_methods": "We accept Visa, Mastercard, American Express, and PayPal.",
    "order_status": "To check your order status, please visit the 'My Orders' section on our website.",
    "discount_codes": "Discount codes can be applied at checkout. Only one code can be used per order.",
    "product_warranty": "Most electronics come with a 1-year manufacturer's warranty.",
    "size_chart": "Please refer to the size chart available on each product page for accurate measurements.",
    "international_shipping": "Yes, we offer international shipping to select countries. Shipping costs and times vary.",
}

# --- 2. Embedding Model and ChromaDB Setup ---
@st.cache_resource
def initialize_vector_store():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Prepare documents for ChromaDB
    docs = []
    for key, value in KNOWLEDGE_BASE.items():
        docs.append({"page_content": value, "metadata": {"source": key}})
    
    # Using from_texts for simplicity, providing metadata directly
    # Note: ChromaDB's from_texts expects list of texts and an optional list of metadatas
    texts = [d["page_content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    vectordb = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, persist_directory="./chroma_db")
    vectordb.persist()
    return vectordb, embeddings

vectordb, embeddings = initialize_vector_store()

# --- 3. Query Complexity Classifier (QCC) Module ---
class QueryComplexityClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)
        self._train_classifier()

    def _train_classifier(self):
        # Simulated training data
        queries = [
            "What is the shipping time?", "How long for delivery?", "Where is my order?", # Straightforward
            "What are your return policies?", "Can I return a damaged item?", "How do I process a refund?", # Moderate
            "Compare the features of product X vs product Y.", "Provide a detailed explanation of the warranty terms for high-end electronics, including conditions for voiding the warranty.", "I need help troubleshooting a complex issue with my recently purchased smart home device that isn't connecting to my network and showing error code E-234. What steps should I take and what are the potential causes?" # Complex
        ]
        labels = [
            "straightforward", "straightforward", "straightforward",
            "moderate", "moderate", "moderate",
            "complex", "complex", "complex"
        ]
        self.X_train = self.vectorizer.fit_transform(queries)
        self.model.fit(self.X_train, labels)

    def classify(self, query: str) -> str:
        query_vec = self.vectorizer.transform([query])
        prediction = self.model.predict(query_vec)
        return prediction[0]

# --- 4. LLM Orchestration and Strategy Selection Module ---
class LLMOrchestrator:
    def __init__(self, vectordb, embeddings):
        self.llm = OpenAI(temperature=0.7, openai_api_key=os.environ["OPENAI_API_KEY"])
        self.vectordb = vectordb
        self.embeddings = embeddings

    def _get_rag_chain(self, k_retrievals: int = 2):
        qa_template = """You are an AI assistant for an e-commerce platform. Use the following context to answer the user's question accurately and helpfully. If the answer is not in the context, state that you don't know, but try to guide the user to relevant information or actions. If it's a troubleshooting query, provide step-by-step guidance. If it requires information not in the context, suggest contacting human support.

Context: {context}

Question: {question}

Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)
        
        retriever = self.vectordb.as_retriever(search_kwargs={"k": k_retrievals})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False, # Set to True if you want to see sources
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

    def generate_response(self, query: str, complexity_label: str) -> str:
        if complexity_label == "straightforward":
            # Direct LLM call for straightforward queries
            prompt = f"Answer the following e-commerce customer query concisely: {query}"
            response = self.llm.invoke(prompt)
            return response.strip()
        elif complexity_label == "moderate":
            # Single-step RAG
            rag_chain = self._get_rag_chain(k_retrievals=2)
            response = rag_chain.invoke({"query": query})
            return response["result"].strip()
        elif complexity_label == "complex":
            # Multi-step RAG (simulated: more retrievals and a more detailed prompt)
            # In a real scenario, this might involve re-ranking, multiple retrieval steps,
            # or using a more sophisticated LangChain agent.
            rag_chain = self._get_rag_chain(k_retrievals=4) # More aggressive retrieval
            response = rag_chain.invoke({"query": query})
            
            # Simulate potential human hand-off or more detailed LLM instruction
            if "troubleshooting" in query.lower() or "issue" in query.lower() or "not working" in query.lower():
                return f"Your query is complex and might require detailed troubleshooting. I have performed an extensive search.\n\n{response['result'].strip()}\n\nIf this doesn't fully resolve your issue, I recommend contacting our advanced technical support team." 
            else:
                return response["result"].strip()
        else:
            return "I'm sorry, I couldn't understand the complexity of your query."

# --- Streamlit UI --- 
st.set_page_config(page_title="E-commerce Chatbot", page_icon=":shopping_trolley:")
st.title(":shopping_trolley: E-commerce Customer Support Chatbot")
st.markdown("This chatbot dynamically adjusts its strategy based on your query complexity.")

# Initialize components
qcc = QueryComplexityClassifier()
llm_orchestrator = LLMOrchestrator(vectordb, embeddings)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing query and generating response..."):
            # 1. Classify query complexity
            complexity_label = qcc.classify(prompt)
            st.info(f"Query classified as: **{complexity_label.capitalize()}**") # Show complexity for debugging/demo
            
            # 2. Generate response based on complexity
            response = llm_orchestrator.generate_response(prompt, complexity_label)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
