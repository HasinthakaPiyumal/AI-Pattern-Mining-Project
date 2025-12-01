import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

# Initialize LLM
# Ensure OPENAI_API_KEY is set in your environment or .env file
llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

def select_persona(query: str) -> str:
    """Determines the chatbot's persona based on keywords in the user query."""
    query_lower = query.lower()
    if "technical" in query_lower or "error" in query_lower or "bug" in query_lower:
        return "technical expert"
    elif "refund" in query_lower or "billing" in query_lower or "payment" in query_lower:
        return "financial advisor"
    elif "hello" in query_lower or "hi" in query_lower or "how are you" in query_lower:
        return "friendly guide"
    elif "problem" in query_lower or "issue" in query_lower:
        return "empathetic listener"
    else:
        return "customer support agent"

def get_chatbot_response(user_query: str) -> str:
    """Generates a chatbot response using the selected persona and LLM."""
    persona = select_persona(user_query)
    
    messages = [
        SystemMessage(content=f"Pretend you are a {persona}. Provide helpful and relevant information."),
        HumanMessage(content=user_query),
    ]
    
    response = llm.invoke(messages)
    return response.content

# Streamlit UI
st.set_page_config(page_title="Persona-Driven Customer Support Chatbot")
st.title("Customer Support Chatbot with Persona Role-Playing")
st.markdown("This chatbot adapts its persona based on your query to provide tailored support.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # Generate response
        response = get_chatbot_response(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"): 
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
