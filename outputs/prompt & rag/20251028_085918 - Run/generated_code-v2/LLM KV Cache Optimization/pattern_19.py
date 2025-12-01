from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import streamlit as st
import requests

# --- FastAPI Backend (api.py content) ---
fastapi_app = FastAPI()

# Initialize vLLM engine
# Using a small model for demonstration. In a real scenario, you'd use a larger, fine-tuned model.
# PagedAttention is handled internally by vLLM for efficient KV cache management.
llm = LLM(model="facebook/opt-125m") 

# Sampling parameters for LLM inference
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@fastapi_app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # In a real application, you'd manage session history here.
    # For this example, we'll just pass the current message.
    prompt = f"User: {request.message}\nAssistant:"

    # Generate response using vLLM
    outputs = llm.generate([prompt], sampling_params)

    generated_text = outputs[0].outputs[0].text.strip()
    return {"response": generated_text}

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# --- Streamlit Frontend (app.py content) ---

# FastAPI backend URL
API_URL = "http://localhost:8000/chat"

def run_streamlit_app():
    st.title("AI Customer Support Chatbot (PagedAttention Powered)")

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

        # Send message to backend API
        try:
            response = requests.post(API_URL, json={"message": prompt}).json()
            assistant_response = response.get("response", "Sorry, I couldn't get a response.")
        except requests.exceptions.ConnectionError:
            assistant_response = "Error: Could not connect to the chatbot backend. Is it running?"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    # This combined file contains both the FastAPI backend and Streamlit frontend code.
    # To run this application, you need to execute them as separate processes.
    # 1. Run the FastAPI backend:
    #    Save the FastAPI part (from `from fastapi import FastAPI` to `uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)`) 
    #    into a file named `api.py` and run it using `python api.py` or `uvicorn api:fastapi_app --host 0.0.0.0 --port 8000`.
    #
    # 2. Run the Streamlit frontend:
    #    Save the Streamlit part (from `import streamlit as st` to the end of `run_streamlit_app()` function)
    #    into a file named `app.py` and run it using `streamlit run app.py`.
    #
    # For this single file demonstration, direct execution will only run the `if __name__ == "__main__":` block. 
    # To run both, you would typically use separate terminal windows or a process manager.
    print("This file contains code for both FastAPI backend and Streamlit frontend. \n")
    print("Please refer to the explanation on how to run these components separately.")
    print("To run the FastAPI backend, execute: `uvicorn chatbot_system:fastapi_app --host 0.0.0.0 --port 8000`")
    print("To run the Streamlit frontend, execute: `streamlit run chatbot_system.py`")
    print("Note: For Streamlit to work, you may need to move the `run_streamlit_app()` call into the top level ")
    print("or ensure `streamlit run` is pointed to a file that directly executes the Streamlit code.")
    print("A more practical approach is to keep them in separate files (api.py and app.py) as suggested in the architecture.")
    # If you want to try running them within the same script via multiprocessing (advanced, may have issues):
    # import multiprocessing
    # import time
    # fastapi_process = multiprocessing.Process(target=run_fastapi)
    # fastapi_process.start()
    # time.sleep(5) # Give FastAPI time to start
    # # Streamlit generally needs to be run from the CLI, direct programmatic execution can be tricky.
    # # For this example, it's better to guide the user to run them separately.
