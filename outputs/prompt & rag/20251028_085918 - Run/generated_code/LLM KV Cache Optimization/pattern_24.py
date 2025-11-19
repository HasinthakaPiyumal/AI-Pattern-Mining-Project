from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import collections

# --- Constants for Mock LLM --- #
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = 64

# --- KVCacheManager Class --- #
class KVCacheManager:
    def __init__(self):
        self.cache = {}

    def get_kv_cache(self, prefix_tokens):
        longest_match = []
        cached_kv_data = None

        for i in range(1, len(prefix_tokens) + 1):
            current_prefix = tuple(prefix_tokens[:i])
            if current_prefix in self.cache:
                longest_match = current_prefix
                cached_kv_data = self.cache[current_prefix]
            else:
                break
        
        return longest_match, cached_kv_data

    def store_kv_cache(self, prefix_tokens, kv_data):
        # Ensure prefix_tokens is a tuple for hashing
        self.cache[tuple(prefix_tokens)] = kv_data

# --- LLMMock/Simulator Class --- #
class LLMMock:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.name = model_name

    def _simulate_kv_tensors(self, sequence_length):
        # Simulates generating KV tensors for a given sequence length
        # Returns a list of (key_tensor, value_tensor) for each layer
        kv_tensors = []
        for _ in range(NUM_LAYERS):
            key = torch.rand(1, NUM_HEADS, sequence_length, HEAD_DIM)
            value = torch.rand(1, NUM_HEADS, sequence_length, HEAD_DIM)
            kv_tensors.append((key, value))
        return kv_tensors

    def inference(self, input_ids: list[int], past_key_values=None):
        current_sequence_length = len(input_ids)

        simulated_output_text = f"Mock LLM response for tokens: {self.tokenizer.decode(input_ids)}."

        if past_key_values:
            # Simulate KV tensor concatenation for decoding phase
            total_kv_tensors = []
            for i in range(NUM_LAYERS):
                past_key, past_value = past_key_values[i]
                current_key = torch.rand(1, NUM_HEADS, current_sequence_length, HEAD_DIM)
                current_value = torch.rand(1, NUM_HEADS, current_sequence_length, HEAD_DIM)
                
                new_key = torch.cat((past_key, current_key), dim=2)
                new_value = torch.cat((past_value, current_value), dim=2)
                total_kv_tensors.append((new_key, new_value))
            simulated_output_text = f"[Cached Prefixes Used] " + simulated_output_text
        else:
            # Simulate new KV tensor generation for prefill phase
            total_kv_tensors = self._simulate_kv_tensors(current_sequence_length)
            simulated_output_text = f"[Fresh Prefill] " + simulated_output_text

        return simulated_output_text, total_kv_tensors

# --- FastAPI Application --- #
app = FastAPI()

# Global instances
kv_cache_manager = KVCacheManager()
llm_mock = LLMMock()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_query = request.query
    input_ids = llm_mock.tokenizer.encode(user_query, add_special_tokens=True)

    # Try to find the longest matching prefix in the KV cache
    longest_match_prefix_tokens, cached_kv_data = kv_cache_manager.get_kv_cache(input_ids)

    final_response = ""
    new_kv_tensors_for_store = None

    if cached_kv_data and longest_match_prefix_tokens:
        # Cache hit: Use cached KV for the prefix and process remaining tokens
        remaining_tokens_ids = input_ids[len(longest_match_prefix_tokens):]
        
        # If there are no remaining tokens, it means the entire query was a hit
        if not remaining_tokens_ids:
            final_response = f"[Full Cache Hit] Response for '{user_query}'."
            new_kv_tensors_for_store = cached_kv_data # Reuse the cached data as is
        else:
            # Simulate inference for the remaining part, using past_key_values
            response, new_kv_tensors_for_store = llm_mock.inference(
                remaining_tokens_ids, past_key_values=cached_kv_data
            )
            final_response = response
            
            # For storing, we need the KV for the *entire* input, so we use the combined ones
            # (new_kv_tensors_for_store already contains the combined ones from LLMMock.inference)
            
    else:
        # Cache miss: Process the entire query from scratch (prefill)
        response, new_kv_tensors_for_store = llm_mock.inference(input_ids, past_key_values=None)
        final_response = response

    # Store/update the KV cache for the full input_ids
    # Only store if new_kv_tensors_for_store was actually generated or updated
    if new_kv_tensors_for_store is not None:
        kv_cache_manager.store_kv_cache(input_ids, new_kv_tensors_for_store)

    return {"response": final_response}
