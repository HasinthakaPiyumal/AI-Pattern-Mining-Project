
import streamlit as st
from fastapi import FastAPI, Request
from langdetect import detect, DetectorFactory, LangDetectException
from google.cloud import translate_v2 as translate
import openai
import os
import uvicorn
import requests
import json

# --- Configuration --- #
# Set your API keys and project ID as environment variables or replace placeholders
# For Google Cloud Translation, ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set
# pointing to your service account key file, e.g., os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/google-cloud-key.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "YOUR_GOOGLE_CLOUD_PROJECT_ID")

openai.api_key = OPENAI_API_KEY

try:
    # Initialize Google Cloud Translation client
    # The client will use GOOGLE_APPLICATION_CREDENTIALS environment variable for authentication
    translate_client = translate.Client(project=GOOGLE_CLOUD_PROJECT_ID)
except Exception as e:
    st.error(f"Failed to initialize Google Cloud Translation client. Ensure GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT_ID are set correctly: {e}")
    translate_client = None # Set to None if initialization fails

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

# --- Helper Functions (Core CLET Logic) --- #

def detect_language(text: str) -> str:
    """Detects the language of the given text."""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        # Fallback if language detection fails (e.g., very short text)
        return "en"
    except Exception as e:
        print(f"Error during language detection: {e}")
        return "en"

def translate_text(text: str, target_language: str, source_language: str = None) -> str:
    """Translates text using Google Cloud Translation API."""
    if not translate_client:
        return f"Translation service not available. Original text: {text}"
    try:
        result = translate_client.translate(
            text, target_language=target_language, source_language=source_language
        )
        return result["translatedText"]
    except Exception as e:
        print(f"Google Cloud Translation failed: {e}")
        return text # Return original text on failure

def get_generative_ai_response(prompt: str) -> str:
    """Gets a response from the Generative AI model (OpenAI)."""
    if not openai.api_key or openai.api_key == "YOUR_OPENAI_API_KEY":
        return "Generative AI service not configured. Please set OPENAI_API_KEY."
    try:
        # Using ChatCompletion for more advanced models like gpt-3.5-turbo
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant for an e-commerce platform."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Generative AI response failed: {e}")
        return "I am sorry, I could not process your request at this moment. Please try again later."

# --- FastAPI Backend API Gateway & Orchestrator --- #
fastapi_app = FastAPI()

@fastapi_app.post("/chat")
async def chat_endpoint(request: Request):
    """Handles incoming customer queries, orchestrates CLET steps, and returns responses."""
    try:
        data = await request.json()
        customer_query = data.get("query", "")

        if not customer_query:
            return {"response": "Please provide a query.", "original_lang": "unknown"}

        # 1. Language Detection
        detected_lang = detect_language(customer_query)
        print(f"[FastAPI] Detected Language: {detected_lang}")

        english_query = customer_query
        # 2. Tools Integration: Translate non-English inputs to English
        if detected_lang != "en":
            english_query = translate_text(customer_query, target_language="en", source_language=detected_lang)
            print(f"[FastAPI] Translated to English: {english_query}")

        # 3. Strategic Planning & Decomposition (Implicit for this basic implementation)
        # For more complex scenarios, this is where you'd break down a query into sub-tasks,
        # perform knowledge mining, or generate multi-step prompts for the GenAI.
        # Here, the GenAI is expected to handle the query directly.

        # 4. Generative AI Model: Get response in English
        ai_response_english = get_generative_ai_response(f"User query: {english_query}")
        print(f"[FastAPI] AI Response (English): {ai_response_english}")

        final_response = ai_response_english
        # 5. Tools Integration: Translate English response back to the customer's native language
        if detected_lang != "en":
            final_response = translate_text(ai_response_english, target_language=detected_lang, source_language="en")
            print(f"[FastAPI] Translated back to {detected_lang}: {final_response}")

        # 6. AI-Human Iteration and Refinement (Conceptual)
        # This phase would typically involve a separate interface for human agents to review
        # and refine ambiguous translations, or an offline process for model fine-tuning
        # based on feedback. For a real-time chatbot, this might be a "human handover" trigger.

        return {"response": final_response, "original_lang": detected_lang}

    except json.JSONDecodeError:
        return {"response": "Invalid JSON in request body.", "original_lang": "unknown"}
    except Exception as e:
        print(f"[FastAPI] An error occurred in chat_endpoint: {e}")
        return {"response": "An internal server error occurred. Please try again later.", "original_lang": "unknown"}

# --- Streamlit Frontend (Customer Facing) --- #
def run_streamlit_app():
    st.set_page_config(page_title="CLET Multi-lingual Chatbot", layout="centered")
    st.title("🌎 CLET Multi-lingual Customer Support Chatbot")
    st.write("Ask your questions in any language, and I'll do my best to assist you!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Translating and generating response..."):
                try:
                    # Call FastAPI backend
                    fastapi_url = "http://localhost:8000/chat"
                    response = requests.post(fastapi_url, json={"query": prompt})
                    response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                    response_data = response.json()
                    assistant_response = response_data.get("response", "Error getting response from backend.")

                except requests.exceptions.ConnectionError:
                    assistant_response = "Could not connect to the backend server. Please ensure the FastAPI server is running at http://localhost:8000."
                except requests.exceptions.RequestException as e:
                    assistant_response = f"Error communicating with backend: {e}"
                except Exception as e:
                    assistant_response = f"An unexpected error occurred: {e}"

            st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# --- Main Entry Point --- #
if __name__ == "__main__":
    # This script contains both the FastAPI backend and the Streamlit frontend.
    # For a real-world application, these would typically be deployed separately.
    # To run this application:
    # 1. Save this code as `clet_chatbot_app.py`.
    # 2. Install dependencies: `pip install streamlit fastapi uvicorn python-langdetect google-cloud-translate openai requests`
    # 3. Set your environment variables for API keys:
    #    export OPENAI_API_KEY="your_openai_api_key_here"
    #    export GOOGLE_CLOUD_PROJECT_ID="your_gcp_project_id_here"
    #    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google_cloud_service_account_key.json"
    #    (Replace placeholders with your actual keys/paths)
    # 4. Open two separate terminal windows.
    #
    #    In the FIRST terminal, run the FastAPI backend:
    #    `uvicorn clet_chatbot_app:fastapi_app --reload --port 8000`
    #    (The `--reload` flag is for development; remove for production)
    #
    #    In the SECOND terminal, run the Streamlit frontend:
    #    `streamlit run clet_chatbot_app.py`
    #
    #    The Streamlit app will open in your browser, and it will communicate
    #    with the FastAPI backend for translation and AI response generation.

    # If you try to run `python clet_chatbot_app.py` directly, it will only execute the Streamlit app.
    # The Streamlit app expects the FastAPI backend to be running on http://localhost:8000.
    run_streamlit_app()

