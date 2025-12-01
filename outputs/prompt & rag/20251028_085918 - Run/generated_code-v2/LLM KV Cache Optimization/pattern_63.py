import streamlit as st
from fastapi import FastAPI, Request
import requests
import json

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- FastAPI Chatbot API ---
app = FastAPI()

_docs = [
    "The customer support chatbot can answer questions about product features.",
    "For billing inquiries, please visit our support portal.",
    "Our hours of operation are Monday to Friday, 9 AM to 5 PM EST.",
    "You can reset your password by clicking on the 'Forgot Password' link on the login page.",
    "We offer a 30-day money-back guarantee on all products."
]
_embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_vectorstore = Chroma.from_texts(texts=_docs, embedding=_embeddings_model)
_retriever = _vectorstore.as_retriever()

_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

VLLM_API_URL = "http://localhost:8000/generate"

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_query = data.get("query")

    if not user_query:
        return {"response": "Please provide a query."}

    rag_chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | _prompt
    )
    
    formatted_prompt = rag_chain.invoke({"question": user_query}).messages[0].content

    try:
        response = requests.post(
            VLLM_API_URL,
            json={
                "prompt": formatted_prompt,
                "max_tokens": 512,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        vllm_output = response.json()
        
        llm_response_text = vllm_output["text"][0]["generated_text"].strip() # Adjusted based on common vLLM output format
        return {"response": llm_response_text}
    except requests.exceptions.ConnectionError:
        return {"response": "Error: Could not connect to the vLLM server. Make sure vLLM is running at " + VLLM_API_URL}
    except requests.exceptions.RequestException as e:
        return {"response": f"Error from vLLM server: {e}"}
    except KeyError:
        return {"response": f"Error: Unexpected response format from vLLM. Make sure 'text' and 'generated_text' keys exist: {vllm_output}"}
    except Exception as e:
        return {"response": f"An unexpected error occurred: {e}"}


# --- Streamlit Frontend ---
def run_streamlit_app():
    st.set_page_config(page_title="AI Customer Support Chatbot")
    st.title("AI Customer Support Chatbot 💬")

    FASTAPI_CHAT_URL = "http://localhost:8001/chat"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What can I help you with?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            try:
                response = requests.post(FASTAPI_CHAT_URL, json={"query": prompt})
                response.raise_for_status()
                data = response.json()
                assistant_response = data.get("response", "An error occurred during processing.")
            except requests.exceptions.ConnectionError:
                assistant_response = "Error: Could not connect to the Chatbot API. Make sure the FastAPI server is running at " + FASTAPI_CHAT_URL
            except requests.exceptions.RequestException as e:
                assistant_response = f"Error from Chatbot API: {e}"
            except Exception as e:
                assistant_response = f"An unexpected error occurred: {e}"

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

if __name__ == "__main__":
    # Check if the script is run directly (intended for FastAPI) or via streamlit (intended for Streamlit)
    import sys
    if "streamlit" not in sys.modules:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
    else:
        run_streamlit_app()