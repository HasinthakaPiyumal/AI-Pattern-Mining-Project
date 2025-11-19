import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
import requests

################################################################################
# Backend (FastAPI) Application Code - backend_app.py
################################################################################

app = FastAPI()

# Load a small LLM for demonstration
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure pad_token is set for generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

class TrieNode:
    def __init__(self):
        self.children = {}
        self.kv_cache = None  # Stores (key_states, value_states)
        self.token_id = None

class KVCacheManager:
    def __init__(self):
        self._prefix_trie = TrieNode()

    def _get_trie_node(self, tokens, create_if_not_exists=False):
        current_node = self._prefix_trie
        for token_id in tokens:
            if token_id not in current_node.children:
                if create_if_not_exists:
                    current_node.children[token_id] = TrieNode()
                else:
                    return None
            current_node = current_node.children[token_id]
            current_node.token_id = token_id # Store token_id for debugging or future use
        return current_node

    def get_cached_kv(self, input_ids: torch.Tensor):
        current_node = self._prefix_trie
        longest_prefix_tokens = []
        cached_kv = None

        for token_id in input_ids.squeeze().tolist():
            if token_id in current_node.children:
                current_node = current_node.children[token_id]
                longest_prefix_tokens.append(token_id)
                if current_node.kv_cache is not None:
                    cached_kv = current_node.kv_cache
            else:
                break

        if cached_kv is not None:
            # Determine the index where the new tokens start
            start_index = len(longest_prefix_tokens)
            remaining_input_ids = input_ids[:, start_index:]
            return cached_kv, start_index, remaining_input_ids
        else:
            return None, 0, input_ids

    def update_cache(self, input_ids: torch.Tensor, new_kv_cache):
        current_node = self._prefix_trie
        for i, token_id in enumerate(input_ids.squeeze().tolist()):
            if token_id not in current_node.children:
                current_node.children[token_id] = TrieNode()
            current_node = current_node.children[token_id]
            current_node.token_id = token_id # Store token_id
            # Store the KV cache up to this point
            if i < len(new_kv_cache):
                current_node.kv_cache = tuple(t.clone() for t in new_kv_cache[i])
            # If new_kv_cache is shorter than input_ids (e.g., due to attention mechanisms),
            # we only cache up to the available KV states.


kv_cache_manager = KVCacheManager()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message

    input_ids = tokenizer.encode(user_message, return_tensors="pt")
    
    # Get cached KV states
    cached_kv, start_index, remaining_input_ids = kv_cache_manager.get_cached_kv(input_ids)

    # Prepare attention mask for the full sequence
    if cached_kv:
        past_length = cached_kv[0][0].shape[-2] # Assuming kv_cache is a tuple of (key, value) pairs, first element is layer 0, first element is key, second last dimension is sequence length
        full_input_ids = torch.cat([input_ids[:, :start_index], remaining_input_ids], dim=-1)
        attention_mask = torch.ones(full_input_ids.shape, dtype=torch.long, device=input_ids.device)
    else:
        past_length = 0
        full_input_ids = input_ids
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=remaining_input_ids,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            past_key_values=cached_kv,
            attention_mask=attention_mask # Pass the full attention mask for correct generation
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Update KV cache with the new full input (user message + generated response)
    # For this simple example, we'll cache the user input part. A more robust solution
    # would recompute KV for the full generated sequence or store intermediate states.
    # Here, we'll simulate updating the cache for the original input + first few generated tokens.
    full_response_ids = outputs[0] # This now contains the original input + generated response
    
    # We need to re-run the model once more to get the KV cache for the full sequence to be stored.
    # This is a simplification for demonstration. In a real-world scenario, you'd get the KV cache
    # directly from the `generate` call if it supported returning intermediate KV states easily for caching.
    with torch.no_grad():
        # Process the full input to get the KV states for caching
        model_output = model(input_ids=full_response_ids.unsqueeze(0), use_cache=True)
        updated_kv_cache = model_output.past_key_values

    # Only update the cache with the prefix that corresponds to the original user input for reuse
    # For simplicity, we are taking the KV cache up to the length of the original user_message tokens
    user_message_token_length = input_ids.shape[1]
    
    # updated_kv_cache is a tuple of tuples. Each inner tuple is (key, value) for a layer.
    # key/value tensors have shape (batch_size, num_heads, sequence_length, head_dim).
    # We want to slice along the sequence_length dimension.
    
    sliced_kv_cache = []
    for layer_kv in updated_kv_cache:
        key_states = layer_kv[0][:, :, :user_message_token_length, :]
        value_states = layer_kv[1][:, :, :user_message_token_length, :]
        sliced_kv_cache.append((key_states, value_states))
    
    kv_cache_manager.update_cache(input_ids, sliced_kv_cache)

    return {"response": generated_text}

################################################################################
# Frontend (Streamlit) Application Code - frontend_app.py
################################################################################

if __name__ == "__main__":
    # The following block will run the FastAPI app if this script is executed directly
    # For Streamlit, you would run 'streamlit run ecommerce_chatbot.py'
    # For FastAPI, you would run 'python ecommerce_chatbot.py' and then access the API.
    # This setup is for demonstration to contain everything in one file. 
    # In a real deployment, these would typically be separate processes.

    # Check if a specific argument is passed to run the backend or frontend
    import sys
    if "run_backend" in sys.argv:
        print("Starting FastAPI Backend...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        st.title("E-commerce Chatbot with KV Cache Reuse")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about our products..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                try:
                    response = requests.post("http://localhost:8000/chat", json={"message": prompt})
                    response.raise_for_status() # Raise an exception for HTTP errors
                    bot_response = response.json()["response"]
                except requests.exceptions.ConnectionError:
                    bot_response = "Error: Could not connect to the backend. Please ensure the backend is running at http://localhost:8000."
                except requests.exceptions.RequestException as e:
                    bot_response = f"Error from backend: {e}"

            with st.chat_message("assistant"):
                st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
