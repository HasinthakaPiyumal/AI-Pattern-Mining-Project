
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms.fake import FakeListLLM
import json

# 1. InCLT Example Management Module
class InCLTExampleManager:
    def __init__(self):
        # Storing examples as (source_lang_text, target_lang_text, source_lang_question, target_lang_question)
        self.examples = [
            {
                "source_text": "The product warranty covers manufacturing defects.",
                "target_text": "La garantía del producto cubre defectos de fabricación.",
                "source_question": "What does the warranty cover?",
                "target_question": "¿Qué cubre la garantía?"
            },
            {
                "source_text": "Our return policy allows for returns within 30 days of purchase.",
                "target_text": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra.",
                "source_question": "What is the return period?",
                "target_question": "¿Cuál es el período de devolución?"
            },
            {
                "source_text": "To reset your password, click on 'Forgot Password'.",
                "target_text": "Para restablecer su contraseña, haga clic en 'Olvidó su contraseña'.",
                "source_question": "How do I reset my password?",
                "target_question": "¿Cómo reinicio mi contraseña?"
            }
        ]

    def get_examples_for_prompt(self, num_examples=2, source_lang="en", target_lang="es"):
        formatted_examples = []
        for i, example in enumerate(self.examples[:num_examples]):
            formatted_examples.append(f"Example {i+1} (Source: {source_lang}):\nQuestion: {example['source_question']}\nAnswer: {example['source_text']}\nExample {i+1} (Target: {target_lang}):\nQuestion: {example['target_question']}\nAnswer: {example['target_text']}")
        return "\n\n".join(formatted_examples)

# 2. Chatbot Service (Integrates all core logic)
class ChatbotService:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.inclt_example_manager = InCLTExampleManager()
        self.qa_collection = self._init_chroma()
        self.llm_chain = self._get_llm_chain()

    def _init_chroma(self):
        collection_name = "faq_collection"
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except:
            pass # Collection might not exist
        
        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        faqs = [
            {"id": "faq1", "text": "Our company provides 24/7 customer support via chat and email.", "lang": "en", "question": "How can I contact customer support?"},
            {"id": "faq2", "text": "Nuestra empresa ofrece soporte al cliente 24 horas al día, 7 días a la semana a través de chat y correo electrónico.", "lang": "es", "question": "¿Cómo puedo contactar al soporte al cliente?"},
            {"id": "faq3", "text": "Shipping usually takes 3-5 business days for domestic orders.", "lang": "en", "question": "What is the shipping time?"},
            {"id": "faq4", "text": "El envío suele tardar de 3 a 5 días hábiles para pedidos nacionales.", "lang": "es", "question": "¿Cuál es el tiempo de envío?"},
            {"id": "faq5", "text": "We accept major credit cards and PayPal.", "lang": "en", "question": "What payment methods are accepted?"},
            {"id": "faq6", "text": "Aceptamos las principales tarjetas de crédito y PayPal.", "lang": "es", "question": "¿Qué métodos de pago se aceptan?"}
        ]

        documents = [faq['text'] for faq in faqs]
        metadatas = [{'lang': faq['lang'], 'question': faq['question']} for faq in faqs]
        ids = [faq['id'] for faq in faqs]
        embeddings = self.embedding_model.encode(documents).tolist()
        
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        return collection

    def _get_llm_chain(self):
        # A fake LLM to demonstrate the InCLT pattern without needing a real large model
        # In a real scenario, replace with a HuggingFacePipeline or OpenAI/Google LLM
        responses = [
            "Based on the provided context and examples, I can provide a relevant answer.",
            "I have processed your query using the cross-lingual examples."
        ]
        llm = FakeListLLM(responses=responses)

        template = """
        You are a multilingual customer support assistant. Answer the user's question based on the provided FAQs.
        Also, learn from the following in-context examples which demonstrate cross-lingual transfer.

        {in_context_examples}

        --- Relevant FAQs ---
        {context}

        --- User Query ({target_lang}) ---
        Question: {question}
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # Define the RAG chain
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._retrieve_faqs(x["question"], x["target_lang"])
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    def _retrieve_faqs(self, query: str, target_lang: str, k=2) -> str:
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.qa_collection.query(
            query_embeddings=query_embedding,
            n_results=k
            # You might want to filter by language here if you only want target language FAQs
            # or retrieve relevant FAQs regardless of language and let the LLM handle it.
            # For this demo, we'll retrieve and then select relevant language if available.
        )
        
        relevant_docs = []
        if results['documents'] and results['metadatas']:
            for i in range(len(results['documents'][0])):
                doc_text = results['documents'][0][i]
                doc_lang = results['metadatas'][0][i]['lang']
                if doc_lang == target_lang or target_lang == "en": # Simple language preference
                    relevant_docs.append(doc_text)
        
        return "\n".join(relevant_docs) if relevant_docs else "No relevant FAQs found."

    def answer_query(self, query: str, target_lang: str) -> str:
        in_context_examples = self.inclt_example_manager.get_examples_for_prompt(target_lang=target_lang)
        
        # The `llm_chain` expects 'question', 'in_context_examples', 'target_lang'
        response = self.llm_chain.invoke({
            "question": query,
            "in_context_examples": in_context_examples,
            "target_lang": target_lang
        })
        return response

# 3. Streamlit UI
st.set_page_config(page_title="Multilingual Chatbot with InCLT")
st.title("🌍 Multilingual Customer Support Chatbot")
st.subheader("InCLT Enhanced FAQ Answering")

@st.cache_resource
def get_chatbot_service():
    return ChatbotService()

chatbot_service = get_chatbot_service()

# Language selection
target_lang = st.selectbox(
    "Select your preferred response language:",
    options=["en", "es"], # Add more languages as supported by your LLM and data
    format_func=lambda x: {"en": "English", "es": "Español"}.get(x, x)
)

user_query = st.text_input("Ask a question in any language:")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Thinking..."):
            response = chatbot_service.answer_query(user_query, target_lang)
            st.write(f"**Chatbot ({target_lang}):** {response}")
    else:
        st.warning("Please enter a question.")

st.markdown("""
--- 
This demo showcases the **InCLT Crosslingual Transfer Prompting** pattern.
It uses a `FakeListLLM` for demonstration purposes. In a real application,
this would be replaced with a robust multilingual LLM (e.g., from Hugging Face Transformers or other providers).
""")

