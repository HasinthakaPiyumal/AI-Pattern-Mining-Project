import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import requests

NUM_SAMPLES = 3
LLM_MODEL_NAME = "gpt2"
REWARD_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
llm_pipeline = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=llm_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)
reward_pipeline = pipeline(
    "sentiment-analysis",
    model=reward_model,
    tokenizer=reward_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def get_best_response(query: str, num_samples: int = NUM_SAMPLES) -> str:
    candidate_responses = []
    for _ in range(num_samples):
        generated_output = llm_pipeline(
            query,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )[0]["generated_text"]
        
        response_only = generated_output[len(query):].strip()
        candidate_responses.append(response_only)

    if not candidate_responses:
        return "No responses could be generated."

    scores = []
    for response in candidate_responses:
        reward_output = reward_pipeline(response)[0]
        
        if reward_output['label'] == 'LABEL_1':
            scores.append(reward_output['score'])
        elif reward_output['label'] == 'LABEL_0':
            scores.append(1 - reward_output['score'])
        else:
            scores.append(0.0)

    best_response_index = scores.index(max(scores))
    return candidate_responses[best_response_index]

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

@app.post("/generate_response", response_model=ChatResponse)
async def generate_chat_response(request: QueryRequest):
    best_response = get_best_response(request.query)
    return ChatResponse(response=best_response)

def streamlit_app():
    st.set_page_config(page_title="Customer Support Chatbot (Rejection Sampling)")
    st.title("Customer Support Chatbot")
    st.write("Ask a question and get an optimized response!")

    user_query = st.text_area("Your Question:", height=100)

    if st.button("Get Response"):
        if user_query:
            with st.spinner("Generating best response..."):
                try:
                    fastapi_url = "http://localhost:8000/generate_response"
                    response = requests.post(fastapi_url, json={"query": user_query})
                    
                    if response.status_code == 200:
                        chat_response = response.json()
                        st.success("Response generated!")
                        st.markdown(f"**Chatbot:** {chat_response['response']}")
                    else:
                        st.error(f"Error from backend: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please ensure the backend is running.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter a question.")

streamlit_app()