
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import streamlit as st

# --- Configuration and Environment Variables ---
load_dotenv()

class LLMConfig(BaseModel):
    provider: str
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 150

# --- LLM Abstraction Layer ---

class LLMProvider(ABC):
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        pass

class OpenAIGPTProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        if not api_key:
            raise ValueError("OpenAI API key is not set.")
        self.api_key = api_key
        self.model_name = model_name
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def invoke(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 150),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

class GoogleGeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        if not api_key:
            raise ValueError("Google Gemini API key is not set.")
        self.api_key = api_key
        self.model_name = model_name
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")

    def invoke(self, prompt: str, **kwargs) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt,
                                              generation_config=genai.types.GenerationConfig(
                                                  temperature=kwargs.get("temperature", 0.7),
                                                  max_output_tokens=kwargs.get("max_tokens", 150),
                                              ))
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Google Gemini API error: {e}")

class HuggingFaceLlamaProvider(LLMProvider):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", api_token: Optional[str] = None):
        # For simplicity, this example will just return a mock response.
        # A real implementation would involve loading the model via transformers or using an inference API.
        self.model_name = model_name
        self.api_token = api_token

    def invoke(self, prompt: str, **kwargs) -> str:
        # In a real scenario, you'd integrate with Hugging Face transformers or their inference API.
        # For this demonstration, we'll return a simple mock response.
        print(f"Using HuggingFace Llama (mock) with model: {self.model_name}")
        return f"[Mock Llama Response for '{prompt[:50]}...'] This is a simulated response from {self.model_name}."

class LLMFactory:
    _providers: Dict[str, type[LLMProvider]] = {
        "openai": OpenAIGPTProvider,
        "gemini": GoogleGeminiProvider,
        "llama": HuggingFaceLlamaProvider, # This is a mock implementation
    }

    @staticmethod
    def get_provider(config: LLMConfig) -> LLMProvider:
        provider_class = LLMFactory._providers.get(config.provider.lower())
        if not provider_class:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        if config.provider.lower() == "openai":
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            return provider_class(api_key=api_key, model_name=config.model_name)
        elif config.provider.lower() == "gemini":
            api_key = config.api_key or os.getenv("GEMINI_API_KEY")
            return provider_class(api_key=api_key, model_name=config.model_name)
        elif config.provider.lower() == "llama":
            # For Llama, you might need a Hugging Face API token for inference endpoints
            api_token = config.api_key or os.getenv("HF_API_TOKEN")
            return provider_class(model_name=config.model_name, api_token=api_token)
        else:
            raise ValueError(f"Configuration for {config.provider} is incomplete.")

# --- RAG System (Simplified with Chroma) ---

class RAGSystem:
    def __init__(self):
        try:
            import chromadb
            from langchain_community.document_loaders import TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores import Chroma
            from langchain_core.documents import Document

            # Initialize ChromaDB in-memory for simplicity
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection("customer_support_docs")

            # For demonstration, add some dummy documents
            self.add_documents([Document(page_content="Our refund policy allows full refunds within 30 days of purchase."),
                                Document(page_content="To reset your password, visit the 'Forgot Password' link on the login page."),
                                Document(page_content="Our customer support hours are Monday to Friday, 9 AM to 5 PM EST.")])
        except ImportError:
            print("ChromaDB and LangChain components not installed. RAG will be disabled.")
            self.client = None
            self.collection = None
            self.add_documents = lambda x: None # Mock the method

    def add_documents(self, documents: List[Any]):
        if not self.collection:
            print("RAG system not initialized. Cannot add documents.")
            return
        try:
            # Dummy embedding function for basic Chroma usage if no LLM for embeddings is configured
            # In a real app, you'd use a proper embedding model.
            # For Langchain integration with OpenAIEmbeddings:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
                for i, doc in enumerate(documents):
                    self.collection.add(documents=[doc.page_content], metadatas=[{"source": "dummy_doc"}], ids=[f"doc_{i}"], embeddings=[embeddings_model.embed_query(doc.page_content)])
            else:
                print("OPENAI_API_KEY not found. Cannot use OpenAIEmbeddings for RAG.")
                for i, doc in enumerate(documents):
                    self.collection.add(documents=[doc.page_content], metadatas=[{"source": "dummy_doc"}], ids=[f"doc_{i}"])


        except Exception as e:
            print(f"Error adding documents to Chroma: {e}")

    def retrieve(self, query: str, k: int = 2) -> List[str]:
        if not self.collection:
            return []
        try:
            # In a real RAG setup, you'd embed the query first.
            # For simplicity, if no embedding model is available, this might just do a direct text search (less effective).
            results = self.collection.query(query_texts=[query], n_results=k)
            return [doc for doc in results['documents'][0]] if results['documents'] else []
        except Exception as e:
            print(f"Error retrieving from Chroma: {e}")
            return []

# --- Chatbot Backend (FastAPI) ---

app = FastAPI()

class ChatRequest(BaseModel):
    user_message: str
    model_provider: str = "gemini" # Default provider
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

# Initialize RAG system globally
rag_system = RAGSystem()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Retrieve relevant context from RAG
        context_docs = rag_system.retrieve(request.user_message)
        context_str = "\n".join(context_docs) if context_docs else "No additional context available."

        # Construct prompt for LLM
        system_prompt = "You are a helpful customer support assistant.\n"
        if context_docs:
            system_prompt += f"Answer the user's question based on the following context:\n{context_str}\n"
        full_prompt = f"{system_prompt}\nUser: {request.user_message}\nAssistant:"

        # Prepare LLM configuration
        llm_config_data = {
            "provider": request.model_provider,
            "model_name": request.model_name or (os.getenv("OPENAI_DEFAULT_MODEL") if request.model_provider == "openai" else os.getenv("GEMINI_DEFAULT_MODEL")),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            # API keys are loaded from .env or passed implicitly by the factory
        }
        llm_config = LLMConfig(**{k: v for k, v in llm_config_data.items() if v is not None})

        # Get LLM provider and invoke
        llm_provider = LLMFactory.get_provider(llm_config)
        response_content = llm_provider.invoke(full_prompt,
                                              temperature=llm_config.temperature,
                                              max_tokens=llm_config.max_tokens)

        return {"response": response_content}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Streamlit Frontend ---

def streamlit_frontend():
    st.title("🤖 AI Customer Support Chatbot")

    # Sidebar for LLM configuration
    st.sidebar.header("LLM Configuration")
    selected_provider = st.sidebar.selectbox(
        "Select LLM Provider",
        ("gemini", "openai", "llama"),
        index=0 # Default to Gemini
    )

    default_openai_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")
    default_gemini_model = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-pro")
    default_llama_model = os.getenv("HF_DEFAULT_MODEL", "meta-llama/Llama-2-7b-chat-hf")

    model_options = {
        "openai": [default_openai_model, "gpt-4-turbo", "gpt-4o"],
        "gemini": [default_gemini_model, "gemini-1.5-pro", "gemini-1.5-flash"],
        "llama": [default_llama_model, "meta-llama/Llama-3-8b-chat-hf"],
    }
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        model_options.get(selected_provider, [])
    )

    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=500, value=150, step=10)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                import requests
                # Make request to FastAPI backend
                fastapi_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
                response = requests.post(f"{fastapi_url}/chat", json={
                    "user_message": prompt,
                    "model_provider": selected_provider,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                })
                response.raise_for_status() # Raise an exception for HTTP errors
                chatbot_response = response.json().get("response", "Error: No response from chatbot.")
                full_response += chatbot_response
            except requests.exceptions.ConnectionError:
                full_response = "Error: Could not connect to the FastAPI backend. Make sure it's running at {fastapi_url}."
            except requests.exceptions.RequestException as e:
                full_response = f"Error calling FastAPI: {e}"
            except Exception as e:
                full_response = f"An unexpected error occurred: {e}"

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Main execution block ---
if __name__ == "__main__":
    # This block allows you to run either FastAPI or Streamlit for demonstration.
    # In a real deployment, you would typically run them as separate processes.
    
    # To run FastAPI: uvicorn chatbot_app:app --reload
    # To run Streamlit: streamlit run chatbot_app.py
    
    # For this single file output, we'll demonstrate the Streamlit frontend. 
    # The FastAPI backend needs to be started separately.
    
    print("\n--- Instructions ---")
    print("To run the FastAPI backend: open a terminal and run `uvicorn chatbot_app:app --reload`")
    print(f"Ensure your .env file has OPENAI_API_KEY and/or GEMINI_API_KEY set.")
    print(f"If using OpenAIEmbeddings for RAG, OPENAI_API_KEY is required.")
    print(f"For HuggingFace Llama (mocked), HF_API_TOKEN is optional.")
    print("Once FastAPI is running, open another terminal and run `streamlit run chatbot_app.py` for the frontend.")
    print("--------------------\n")

    # You can comment out the streamlit_frontend() call below if you only want to focus on running FastAPI externally.
    # For a combined demonstration within a single script context (not typical for prod but for quick demo):
    # You'd typically run FastAPI in one process and Streamlit in another.
    # As per the instructions above, the user should launch them separately.
    streamlit_frontend()

