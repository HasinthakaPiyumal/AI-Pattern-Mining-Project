
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import hashlib
import json
import httpx  # For making HTTP requests to vLLM server

app = FastAPI(
    title="KV Cache Reuse Chatbot",
    description="Intelligent Customer Support Chatbot leveraging KV Cache Reuse with vLLM."
)

# --- Configuration ---
VLLM_API_URL = "http://localhost:8000/generate"
# In a real scenario, this would be more robust (e.g., Redis, persistent storage)
# For demonstration, a simple in-memory dict
# Stores {hashed_prefix_text: "prefilled_system_prompt_id" or similar placeholder}
# This cache is to *conceptually* track if the system prompt has been processed once,
# relying on vLLM's internal caching for actual KV tensor reuse.
kv_prefix_cache: Dict[str, str] = {}
# Stores conversation history per session: {session_id: List[str]}
conversation_history: Dict[str, List[str]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "lmsys/vicuna-7b-v1.5"  # Default model for vLLM


class ChatResponse(BaseModel):
    session_id: str
    response: str
    cached_prefix_used: bool = False


# Helper to hash text prefixes for our application-level cache.
# In a real system, you'd tokenize and hash token IDs for greater accuracy.
def get_prefix_hash(prefix_text: str) -> str:
    return hashlib.sha256(prefix_text.encode('utf-8')).hexdigest()


async def call_vllm_generate(
    prompt: str,
    model: str,
) -> str:
    """
    Calls the vLLM generate API endpoint.
    vLLM internally handles KV cache reuse (e.g., via PagedAttention) when it detects
    common prefixes in incoming prompts. This application layer relies on that behavior.
    If the vLLM server is not running or encounters an error, a mock response is returned.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "model": model,
        "max_tokens": 512,  # Example generation length
        "temperature": 0.7,  # Example sampling temperature
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(VLLM_API_URL, headers=headers, json=payload, timeout=600)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            response_data = response.json()

            # vLLM's actual output format is a list of choices, each with a "text" field.
            # We extract the generated text from the first choice.
            if response_data and isinstance(response_data, dict) and "text" in response_data:
                # This path for a simplified mock vLLM response format
                return response_data["text"][0].strip()
            elif response_data and isinstance(response_data, list) and response_data[0] and "text" in response_data[0]:
                # This path for a more accurate vLLM response format
                return response_data[0]["text"][0].strip()
            else:
                # Fallback if vLLM response format is unexpected
                print(f"Warning: Unexpected vLLM response format. Mocking response for prompt: {prompt[:80]}...")
                return f"Mocked LLM response (unexpected format) to '{prompt[:80]}...': This is an automated reply from the chatbot."

    except httpx.RequestError as e:
        print(f"Error calling vLLM at {VLLM_API_URL}: {e}")
        # Fallback to a mock response if vLLM is unreachable or connection error
        return f"Service unavailable. Could not connect to LLM. Mocked response to '{prompt[:80]}...': How can I help you today?"
    except httpx.HTTPStatusError as e:
        print(f"vLLM HTTP error: {e.response.status_code} - {e.response.text}")
        # Fallback to a mock response for HTTP errors from vLLM
        return f"Error from LLM service. Status: {e.response.status_code}. Mocked response to '{prompt[:80]}...': Please try again later."
    except json.JSONDecodeError as e:
        print(f"JSON decode error from vLLM response: {e}")
        # Fallback to a mock response for JSON parsing errors
        return f"Error processing LLM response. Mocked response to '{prompt[:80]}...': There was an issue processing your request."
    except Exception as e:
        print(f"An unexpected error occurred while calling vLLM: {e}")
        # Generic fallback for any other unexpected errors
        return f"An unexpected error occurred. Mocked response to '{prompt[:80]}...': Please try again."


@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message
    model = request.model

    # Initialize conversation history for a new session
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    current_history = conversation_history[session_id]
    cached_prefix_used = False

    # --- KV Cache Reuse Logic (Application-Level Concept) ---
    # The 'KV Cache Reuse' pattern is primarily managed by vLLM internally via PagedAttention
    # when it processes prompts with shared prefixes. At the application level, we facilitate this
    # by consistently structuring our prompts and tracking if a base prefix (e.g., system prompt)
    # has been processed, implying vLLM will reuse its KV cache for that part.

    system_prompt = "You are a helpful customer support assistant for a tech company. Answer questions concisely and professionally."
    system_prompt_hash = get_prefix_hash(system_prompt)

    full_prompt_for_llm: str

    if not current_history:  # First turn of the conversation
        # Construct the initial prompt including system prompt and the first user message.
        full_prompt_for_llm = f"{system_prompt}\nCustomer: {user_message}\nAgent:"

        # Mark the system prompt as "processed" in our application's conceptual cache.
        # This signifies that vLLM has likely computed and cached its KV states for the system prompt.
        if system_prompt_hash not in kv_prefix_cache:
            kv_prefix_cache[system_prompt_hash] = "initial_system_context_seeded" # Placeholder ID
            print(f"Session {session_id}: First turn. System prompt (prefix) concept stored.")
        else:
            # If the system prompt has been seen before (across sessions or a reset), we acknowledge conceptual reuse.
            print(f"Session {session_id}: First turn. System prompt (prefix) already concept stored, implying vLLM reuse.")
            cached_prefix_used = True

        conversation_history[session_id].append(f"Customer: {user_message}")

    else:  # Subsequent turns in the conversation
        # For subsequent turns, we build the prompt by concatenating the system prompt,
        # the entire conversation history, and the new user message. vLLM's internal
        # mechanisms (like PagedAttention) are expected to reuse the KV cache for the
        # common prefix (system_prompt + previous history).
        history_str = "\n".join(current_history)
        full_prompt_for_llm = f"{system_prompt}\n{history_str}\nCustomer: {user_message}\nAgent:"

        # We consider the KV cache "reused" if the session has previous history,
        # as the system prompt and prior turns form a consistent prefix for vLLM.
        if system_prompt_hash in kv_prefix_cache:
            cached_prefix_used = True
            print(f"Session {session_id}: Subsequent turn. Relying on vLLM's internal KV cache reuse for long prefix.")

        conversation_history[session_id].append(f"Customer: {user_message}")

    # Call the LLM (vLLM) with the constructed full prompt
    llm_response_text = await call_vllm_generate(full_prompt_for_llm, model)

    # Extract the agent's reply from the LLM's raw output.
    # This is a simple heuristic; more robust parsing might be needed depending on LLM output format.
    agent_reply_start_tag = "Agent:"
    if agent_reply_start_tag in llm_response_text:
        agent_reply = llm_response_text.split(agent_reply_start_tag, 1)[-1].strip()
    else:
        agent_reply = llm_response_text.strip() # If "Agent:" not found, take the whole response

    # Update conversation history with the agent's response
    conversation_history[session_id].append(f"Agent: {agent_reply}")

    return ChatResponse(
        session_id=session_id,
        response=agent_reply,
        cached_prefix_used=cached_prefix_used
    )

