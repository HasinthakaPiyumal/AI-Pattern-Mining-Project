from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

app = FastAPI()

class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    answer: str

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
llm = LLM(model=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

class KVCacheManager:
    def __init__(self):
        self.prefix_token_ids: Dict[str, List[int]] = {}

    def get_prefix_tokens(self, prefix_text: str) -> Optional[List[int]]:
        return self.prefix_token_ids.get(prefix_text)

    def store_prefix_tokens(self, prefix_text: str, token_ids: List[int]):
        self.prefix_token_ids[prefix_text] = token_ids
        print(f"Stored KV cache for prefix: '{prefix_text}' with {len(token_ids)} tokens.")

kv_cache_manager = KVCacheManager()

def get_prefix(text: str, num_words: int = 5) -> str:
    words = text.split()
    return " ".join(words[:num_words]) if len(words) >= num_words else text

@app.post("/ask", response_model=QueryResponse)
async def ask_chatbot(request: QueryRequest):
    full_text = request.text
    prefix_text = get_prefix(full_text)
    
    # Attempt to retrieve prefix token IDs from our simulated cache
    cached_prefix_token_ids = kv_cache_manager.get_prefix_tokens(prefix_text)
    
    prompt_token_ids = tokenizer.encode(full_text)
    
    if cached_prefix_token_ids:
        print(f"Application identified a cached prefix: '{prefix_text}'. "
              "vLLM will internally leverage KV cache reuse if applicable.")
        # When vLLM receives prompt_token_ids that match a previously processed prefix,
        # it intelligently reuses the precomputed KV tensors. Our application's role
        # here is to identify this reuse opportunity.
        outputs = await llm.agenerate(
            prompt_token_ids=[prompt_token_ids],
            sampling_params=sampling_params,
        )
    else:
        print(f"Prefix '{prefix_text}' not in application's cache. Performing full inference.")
        outputs = await llm.agenerate(
            prompts=[full_text],
            sampling_params=sampling_params,
        )
        
        # After a full inference, store the prefix tokens for future reuse opportunities.
        prefix_tokens_for_storage = tokenizer.encode(prefix_text)
        kv_cache_manager.store_prefix_tokens(prefix_text, prefix_tokens_for_storage)

    generated_text = ""
    for output in outputs:
        generated_text = output.outputs[0].text
        break

    return QueryResponse(answer=generated_text)