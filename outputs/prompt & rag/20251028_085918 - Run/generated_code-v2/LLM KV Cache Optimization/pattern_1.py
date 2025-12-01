import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List
from fastapi import FastAPI
from pydantic import BaseModel

# kv_cache_chatbot.py content
class LLMCacheManager:
    def __init__(self):
        self._cache = {}

    def get_cached_kv(self, prefix_tokens: Tuple[int, ...]) -> Optional[Tuple[torch.Tensor, ...]]:
        return self._cache.get(prefix_tokens)

    def store_kv_cache(self, prefix_tokens: Tuple[int, ...], kv_values: Tuple[torch.Tensor, ...]):
        self._cache[prefix_tokens] = kv_values

class EcommerceChatbot:
    def __init__(self, model_name: str = "gpt2"):
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._kv_cache_manager = LLMCacheManager()
        self._conversation_history = []

    def _find_longest_common_prefix(self, current_input_tokens: List[int]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[torch.Tensor, ...]], int]:
        longest_prefix_tokens = None
        cached_kv_values = None
        prefix_length = 0

        for length in range(len(current_input_tokens), 0, -1):
            prefix = tuple(current_input_tokens[:length])
            cached = self._kv_cache_manager.get_cached_kv(prefix)
            if cached:
                longest_prefix_tokens = prefix
                cached_kv_values = cached
                prefix_length = length
                break
        return longest_prefix_tokens, cached_kv_values, prefix_length

    def generate_response(self, user_query: str) -> str:
        self._conversation_history.append(f"User: {user_query}")
        full_prompt = "\n".join(self._conversation_history) + "\nAssistant:"

        input_ids = self._tokenizer.encode(full_prompt, return_tensors="pt")
        current_input_tokens = input_ids[0].tolist()

        longest_prefix_tokens, cached_kv_values, prefix_length = self._find_longest_common_prefix(current_input_tokens)

        if cached_kv_values:
            print(f"Cache hit! Reusing KV cache for prefix of length {prefix_length}.")
            suffix_input_ids = self._tokenizer.encode(self._tokenizer.decode(current_input_tokens[prefix_length:]), return_tensors="pt", add_special_tokens=False)
            
            if suffix_input_ids.numel() == 0:
                output = self._model.generate(
                    input_ids=input_ids[:, :prefix_length],
                    past_key_values=cached_kv_values,
                    max_new_tokens=50,
                    pad_token_id=self._tokenizer.eos_token_id,
                    do_sample=True, 
                    top_k=50, 
                    top_p=0.95,
                    num_return_sequences=1
                )
            else:
                output = self._model.generate(
                    input_ids=suffix_input_ids,
                    past_key_values=cached_kv_values,
                    max_new_tokens=50,
                    pad_token_id=self._tokenizer.eos_token_id,
                    do_sample=True, 
                    top_k=50, 
                    top_p=0.95,
                    num_return_sequences=1
                )
        else:
            print("No cache hit. Performing full inference.")
            output = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                pad_token_id=self._tokenizer.eos_token_id,
                do_sample=True, 
                top_k=50, 
                top_p=0.95,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False
            )
            with torch.no_grad():
                outputs = self._model(input_ids, return_dict=True, output_attentions=False, output_hidden_states=False)
                past_key_values_to_cache = outputs.past_key_values
                self._kv_cache_manager.store_kv_cache(tuple(current_input_tokens), past_key_values_to_cache)

        response_tokens = output.sequences[0]
        decoded_response = self._tokenizer.decode(response_tokens[input_ids.shape[-1]:], skip_special_tokens=True)

        self._conversation_history.append(f"Assistant: {decoded_response}")
        return decoded_response.strip()

# fastapi_app.py content
app = FastAPI()

chatbot = EcommerceChatbot(model_name="gpt2")

class ChatRequest(BaseModel):
    user_query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = chatbot.generate_response(request.user_query)
    return {"response": response}

# To run this application, save it as `chatbot_application.py` and run:
# uvicorn chatbot_application:app --reload