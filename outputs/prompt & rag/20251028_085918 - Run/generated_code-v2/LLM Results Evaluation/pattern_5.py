import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

def rephrase_query(original_query: str) -> str:
    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that rephrases and expands customer questions for an e-commerce chatbot. Your goal is to make the query clearer and more comprehensive to ensure a better understanding before generating a final answer. Only provide the rephrased and expanded query, nothing else."),
            HumanMessage(content=original_query),
        ]
    )
    rephrase_chain = rephrase_prompt | llm
    response = rephrase_chain.invoke({"content": original_query})
    return response.content

def generate_response(rephrased_query: str) -> str:
    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an e-commerce customer support agent. Provide a helpful and concise answer to the customer\'s rephrased question. Keep your answer under 100 words."),
            HumanMessage(content=rephrased_query),
        ]
    )
    response_chain = response_prompt | llm
    response = response_chain.invoke({"content": rephrased_query})
    return response.content

st.set_page_config(page_title="E-commerce Chatbot (RaR)", layout="centered")
st.title("E-commerce Customer Support Chatbot (Rephrase and Respond)")

st.write("Ask me anything about our products or services!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        rephrased_q = rephrase_query(prompt)
        final_response = generate_response(rephrased_q)

        st.session_state.messages.append({"role": "assistant", "content": f"**Original Query:** {prompt}\n\n**Rephrased Query:** {rephrased_q}\n\n**Chatbot Response:** {final_response}"})
        
        with st.chat_message("assistant"):
            st.markdown(f"**Original Query:** {prompt}")
            st.markdown(f"**Rephrased Query:** {rephrased_q}")
            st.markdown(f"**Chatbot Response:** {final_response}")
