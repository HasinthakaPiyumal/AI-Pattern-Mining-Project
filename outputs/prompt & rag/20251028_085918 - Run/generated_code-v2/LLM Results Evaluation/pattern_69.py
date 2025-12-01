import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# --- Rephrase and Respond (RaR) Mechanism ---

# Prompt Template for Rephrasing
rephrase_template = """You are an AI assistant designed to help customers on an e-commerce platform. Your first task is to rephrase and expand the user's question to ensure complete understanding before providing an answer. Do not answer the question yet. Just rephrase and clarify. If the question is already clear, just state that you understand.

Chat History:
{chat_history}

Customer Question: {question}

Rephrased and Expanded Question:"""

rephrase_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=rephrase_template
)

# Chain for Rephrasing
rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt, output_key="rephrased_question", verbose=True)

# Prompt Template for Responding
respond_template = """You are an AI assistant for an e-commerce platform. Provide a concise and helpful answer based on the following clarified question. Use the chat history for context.

Chat History:
{chat_history}

Clarified Question: {rephrased_question}

Answer:"""

respond_prompt = PromptTemplate(
    input_variables=["chat_history", "rephrased_question"],
    template=respond_template
)

# Chain for Responding
respond_chain = LLMChain(llm=llm, prompt=respond_prompt, output_key="answer", verbose=True)

# Combined Sequential Chain for RaR
overall_chain = SequentialChain(
    chains=[rephrase_chain, respond_chain],
    input_variables=["chat_history", "question"],
    output_variables=["rephrased_question", "answer"],
    verbose=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="E-commerce Chatbot (RaR)", layout="centered")
st.title("🛒 E-commerce Customer Support Chatbot")
st.subheader("Powered by Rephrase and Respond (RaR) Pattern")

# Initialize chat history and memory in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=False)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            # Get chat history from memory
            chat_history_str = st.session_state.memory.load_memory_variables({})

            # Run the RaR chain
            response = overall_chain.invoke({"question": prompt, "chat_history": chat_history_str['chat_history']})
            
            rephrased_q = response["rephrased_question"]
            final_answer = response["answer"]

            # Store interactions in memory
            st.session_state.memory.save_context({"inputs": prompt}, {"outputs": final_answer})
            
            # Display the rephrased question and final answer
            with st.chat_message("assistant"):
                st.markdown(f"**Rephrased and Clarified:** {rephrased_q}")
                st.markdown(f"**Answer:** {final_answer}")
                
            st.session_state.messages.append({"role": "assistant", "content": f"**Rephrased and Clarified:** {rephrased_q}\n\n**Answer:** {final_answer}"})

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})
