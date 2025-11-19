
import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

# Ensure OpenAI API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in environment variables. Please set it.")
    st.stop()

# Initialize LLM and Embeddings
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
embeddings = OpenAIEmbeddings()

# Sample Product Data
product_data = [
    {"name": "Wireless Bluetooth Headphones", "category": "Electronics", "price": "$99.99", "description": "High-quality sound, comfortable design, 20 hours battery life."}, 
    {"name": "Smartwatch Series 7", "category": "Wearable Tech", "price": "$349.00", "description": "Fitness tracking, heart rate monitor, notifications, water-resistant."}, 
    {"name": "Ergonomic Office Chair", "category": "Home Office", "price": "$249.50", "description": "Adjustable lumbar support, breathable mesh, suitable for long working hours."}, 
    {"name": "4K Ultra HD Smart TV", "category": "Electronics", "price": "$799.00", "description": "Stunning picture quality, built-in streaming apps, voice control."}, 
    {"name": "Stainless Steel Coffee Maker", "category": "Kitchen Appliances", "price": "$75.00", "description": "12-cup capacity, programmable timer, sleek design."}
]

# Sample FAQ Data
faq_data = [
    {"question": "What is your return policy?", "answer": "You can return most items within 30 days of purchase for a full refund or exchange."}, 
    {"question": "How can I track my order?", "answer": "Once your order ships, you will receive a tracking number via email to monitor its delivery status."}, 
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship to over 100 countries worldwide. Shipping fees and delivery times vary by destination."}, 
    {"question": "How do I contact customer support?", "answer": "You can reach our customer support team via live chat on our website or by emailing support@example.com."}
]

@st.cache_resource
def create_vector_stores(products, faqs, embeddings_model):
    product_docs = [
        Document(
            page_content=f"Product: {p['name']}\nCategory: {p['category']}\nPrice: {p['price']}\nDescription: {p['description']}",
            metadata=p
        )
        for p in products
    ]
    faq_docs = [
        Document(
            page_content=f"Question: {f['question']}\nAnswer: {f['answer']}",
            metadata=f
        )
        for f in faqs
    ]

    product_vectorstore = FAISS.from_documents(product_docs, embeddings_model)
    faq_vectorstore = FAISS.from_documents(faq_docs, embeddings_model)

    return product_vectorstore, faq_vectorstore

product_vectorstore, faq_vectorstore = create_vector_stores(product_data, faq_data, embeddings)

# Recommendation System Chain
recommendation_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are an intelligent e-commerce recommendation system. Based on the user's preferences and the provided product context, suggest relevant products and explain why they are a good fit. If no relevant products are found, suggest broadening the search."),
     ("human", "User preferences: {query}\n\nProducts found:\n{context}")]
)

recommendation_chain = (
    RunnableParallel(
        context=RunnablePassthrough.assign(retrieved_docs=lambda x: product_vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(x["query"])),
        query=RunnablePassthrough()
    ) |
    {
        "context": lambda x: "\n".join([doc.page_content for doc in x["context"]["retrieved_docs"]]),
        "query": lambda x: x["query"]
    } |
    recommendation_prompt_template |
    llm |
    StrOutputParser()
)

# Customer Support Chatbot Chain
support_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful and polite e-commerce customer support assistant. Answer the user's questions based on the provided product and FAQ information. If you cannot find a direct answer, kindly state that you don't have enough information and suggest alternative ways to find help."),
     ("human", "User question: {query}\n\nContext:\n{context}")]
)

def get_combined_retriever(question, product_vs, faq_vs):
    product_results = product_vs.as_retriever(search_kwargs={"k": 2}).invoke(question)
    faq_results = faq_vs.as_retriever(search_kwargs={"k": 2}).invoke(question)
    return product_results + faq_results

support_chain = (
    RunnableParallel(
        context=lambda x: get_combined_retriever(x["query"], product_vectorstore, faq_vectorstore),
        query=RunnablePassthrough()
    ) |
    {
        "context": lambda x: "\n".join([doc.page_content for doc in x["context"]]),
        "query": lambda x: x["query"]
    } |
    support_prompt_template |
    llm |
    StrOutputParser()
)

# Streamlit UI
st.set_page_config(page_title="E-commerce LLM System", layout="wide")
st.title("🛍️ Intelligent E-commerce Assistant")

tabs = st.tabs(["Product Recommendations", "Customer Support Chat"])

with tabs[0]:
    st.header("Find Your Perfect Product")
    user_preference = st.text_input("Tell us what you're looking for (e.g., 'headphones for clear calls', 'chair for home office', 'smartwatch with long battery')")

    if st.button("Get Recommendations"):
        if user_preference:
            with st.spinner("Generating recommendations..."):
                recommendations = recommendation_chain.invoke({"query": user_preference})
                st.markdown(recommendations)
        else:
            st.warning("Please enter your preferences to get recommendations.")

with tabs[1]:
    st.header("Chat with Our Support Team")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about products, orders, or policies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = support_chain.invoke({"query": prompt})
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

