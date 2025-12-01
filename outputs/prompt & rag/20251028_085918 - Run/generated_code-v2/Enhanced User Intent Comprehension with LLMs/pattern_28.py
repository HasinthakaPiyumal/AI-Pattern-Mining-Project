import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class MockLLM:
    def invoke(self, prompt_value: dict) -> str:
        context = prompt_value.get("context", "")
        question = prompt_value.get("question", "")

        if context:
            if isinstance(context, dict) and "answer" in context:
                return f"Answer based on our knowledge base: {context['answer']}"
            elif isinstance(context, str) and context:
                if "track" in question.lower() and "order" in question.lower() and "track my order" in context.lower():
                    return "You can track your order by logging into your account and visiting the 'My Orders' section."
                elif "return" in question.lower() and "policy" in question.lower() and "return policy" in context.lower():
                    return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition."
                elif "contact" in question.lower() and "support" in question.lower() and "contact customer support" in context.lower():
                    return "You can contact customer support via email at support@example.com or call us at 1-800-123-4567."
                elif "shipping" in question.lower() and "international" in question.lower() and "international shipping" in context.lower():
                    return "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary by destination."
                elif "product x" in question.lower() and "product x" in context.lower():
                    return "Product X is a high-performance laptop featuring 16GB RAM and a 512GB SSD, ideal for gaming and professional use."
                elif "product y" in question.lower() and "product y" in context.lower():
                    return "Product Y is an ergonomic office chair designed for maximum comfort, featuring adjustable lumbar support and breathable mesh material."
                return f"Based on the information I found ('{context}'), regarding your question '{question}', I can provide a general answer. Please check our FAQ or contact support for more details."
        
        return f"I'm not sure how to answer '{question}' without specific information. Could you please provide more context or rephrase your question?"


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection_name = "ecommerce_knowledge_base"
kb_collection = client.get_or_create_collection(name=collection_name)

faq_data = [
    {"id": "faq1", "content": "How do I track my order?", "answer": "You can track your order by logging into your account and visiting the 'My Orders' section."},
    {"id": "faq2", "content": "What is your return policy?", "answer": "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition."},
    {"id": "faq3", "content": "How can I contact customer support?", "answer": "You can contact customer support via email at support@example.com or call us at 1-800-123-4567."},
    {"id": "faq4", "content": "Do you offer international shipping?", "answer": "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary by destination."},
    {"id": "product1_info", "content": "Information about Product X: A high-performance laptop with 16GB RAM and 512GB SSD.", "answer": "Product X is a high-performance laptop featuring 16GB RAM and a 512GB SSD, ideal for gaming and professional use."},
    {"id": "product2_info", "content": "Details on Product Y: A comfortable ergonomic office chair with lumbar support.", "answer": "Product Y is an ergonomic office chair designed for maximum comfort, featuring adjustable lumbar support and breathable mesh material."},
]

if kb_collection.count() == 0:
    kb_collection.add(
        documents=[d["content"] for d in faq_data],
        metadatas=[{"answer": d["answer"]} for d in faq_data],
        ids=[d["id"] for d in faq_data]
    )

def get_relevant_docs(query: str, n_results: int = 1):
    query_embedding = model.encode(query).tolist()
    results = kb_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas']
    )
    docs = results['documents'][0] if results['documents'] else []
    metas = results['metadatas'][0] if results['metadatas'] else []
    return docs, metas

llm = MockLLM()

def process_query_chain(query: str):
    docs, metadatas = get_relevant_docs(query)
    
    context_for_llm = {}
    if metadatas and "answer" in metadatas:
        context_for_llm = metadatas
    elif docs:
        context_for_llm = {"content": docs}
    
    response = llm.invoke({"context": context_for_llm, "question": query})
    return response

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = process_query_chain(request.query)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def streamlit_app():
    st.title("🛍️ E-commerce Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What can I help you with?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = process_query_chain(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    streamlit_app()
