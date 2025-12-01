
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
import uuid
import httpx

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration --- 

VLLM_API_URL = "http://localhost:8000/generate" # Assuming vLLM server runs on localhost:8000
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"

app = FastAPI()

# --- RAG Setup (Conceptual Data) ---

# In a real application, this would be loaded from a proper knowledge base
KNOWLEDGE_BASE_DOCUMENTS = [
    "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "To check your order status, please visit the 'My Orders' section on our website and enter your order number.",
    "The new 'Quantum Leap' smartphone features a 6.8-inch OLED display and a 108MP camera.",
    "Shipping usually takes 3-5 business days for standard delivery. Expedited shipping options are available at checkout.",
    "For technical support, please contact our support team at support@ecommerce.com or call 1-800-555-0123."
]

embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = Chroma.from_texts(KNOWLEDGE_BASE_DOCUMENTS, embeddings, persist_directory=CHROMA_DB_PATH)
vectorstore.persist()

# --- ChatService Components ---

class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

# In-memory storage for conversation history (for demonstration)
conversation_memories: Dict[str, ConversationBufferWindowMemory] = {}

async def get_llm_response_from_vllm(prompt: str, session_id: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            # vLLM expects a 'prompt' and can optionally take 'sampling_params'
            # For KV cache reuse, vLLM handles it internally if prefixes match
            # We just send the full context/prompt.
            response = await client.post(
                VLLM_API_URL,
                json={
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 256,
                    "stop": ["\nUser:", "\nCustomer:"] # Example stop sequences
                },
                timeout=60.0 # Increase timeout for potentially longer LLM responses
            )
            response.raise_for_status()
            result = response.json()
            # vLLM returns a list of generations, take the first one
            if result and "text" in result and len(result["text"]) > 0:
                # The actual response from vLLM's /generate endpoint might vary slightly
                # This assumes it returns a list of dictionaries with a 'text' key.
                generated_text = result["text"][0].replace(prompt, "").strip()
                return generated_text
            return "Sorry, I couldn't get a response from the LLM."
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"vLLM request failed: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"vLLM returned an error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred with vLLM: {e}")

# Custom LLM for LangChain that interacts with vLLM
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, Generation, AIMessage as LC_AIMessage

class VLLMChatModel(BaseChatModel):
    vllm_api_url: str
    session_id: str # Pass session_id to allow potential logging/tracking if vLLM supported it

    async def _agenerate(self, messages: List[List[HumanMessage | AIMessage]], **kwargs) -> ChatResult:
        # Convert LangChain messages to a single prompt string for vLLM
        full_prompt = ""
        for msg in messages[0]: # messages is a list of lists here for some reason
            if isinstance(msg, HumanMessage):
                full_prompt += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                full_prompt += f"AI: {msg.content}\n"
        full_prompt += "AI:"

        llm_output = await get_llm_response_from_vllm(full_prompt, self.session_id)
        return ChatResult(generations=[Generation(text=llm_output, message=LC_AIMessage(content=llm_output))])

    def _generate(self, messages: List[List[HumanMessage | AIMessage]], **kwargs) -> ChatResult:
        # Synchronous version, not used in FastAPI async path
        raise NotImplementedError("VLLMChatModel does not support synchronous _generate")

    @property
    def _llm_type(self) -> str:
        return "vllm_chat_model"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_query = request.query

    if session_id not in conversation_memories:
        conversation_memories[session_id] = ConversationBufferWindowMemory(
            memory_key="chat_history", return_messages=True, k=5 # Keep last 5 turns
        )

    memory = conversation_memories[session_id]

    # Initialize VLLMChatModel with the current session_id
    llm = VLLMChatModel(vllm_api_url=VLLM_API_URL, session_id=session_id)

    # RAG Chain setup
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False # Set to True if you want to see retrieved docs
    )

    try:
        # LangChain's ConversationalRetrievalChain will construct the prompt
        # including history and retrieved context, which vLLM can then parse
        # and potentially apply KV cache reuse.
        result = await qa_chain.ainvoke({"question": user_query})
        ai_response = result["answer"]
        return ChatResponse(session_id=session_id, response=ai_response)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# To run the FastAPI app:
# uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# To simulate vLLM server (run in a separate terminal):
# docker run --gpus all -it --rm -p 8000:8000 -v ~/.cache:/root/.cache vllm/vllm-openai --model facebook/opt-125m
# Replace 'facebook/opt-125m' with your desired LLM, e.g., 'mistralai/Mistral-7B-Instruct-v0.2'
