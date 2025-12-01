import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# In-memory KV Cache and Conversation History
# kv_cache: {'session_id': {'past_key_values': tuple_of_tensors, 'cached_input_ids': list_of_ints}}
kv_cache = {}
# conversation_history: {'session_id': [{'role': 'user'|'assistant', 'content': 'message'}, ...]}}
conversation_history = {}

# Load LLM and Tokenizer
model_name = "gpt2"  # Using gpt2 for demonstration, can be swapped
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure tokenizer has a pad_token, especially for `generate`
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message

    # 1. Update conversation_history
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({"role": "user", "content": user_message})

    # 2. Construct the full_conversation_text
    full_conversation_text = ""
    for entry in conversation_history[session_id]:
        if entry["role"] == "user":
            full_conversation_text += f"User: {entry['content']}\n"
        elif entry["role"] == "assistant":
            full_conversation_text += f"Assistant: {entry['content']}\n"
    full_conversation_text += "Assistant:" # Prompt the assistant for its turn

    # 3. Tokenize the full conversation text
    # This will be used to check prefix match and for final cache update
    full_conversation_tokenized_ids = tokenizer.encode(full_conversation_text, return_tensors="pt")[0]
    full_conversation_tokenized_ids = full_conversation_tokenized_ids.to(device)

    input_ids_for_generation = full_conversation_tokenized_ids.unsqueeze(0)
    past_kv_for_generate = None

    # 4. KV Cache Logic: Check for prefix match
    if session_id in kv_cache:
        cached_kv_data = kv_cache[session_id]
        cached_input_ids_tensor = torch.tensor(cached_kv_data["cached_input_ids"], device=device)

        # Check if cached_input_ids is a prefix of the current full conversation tokens
        if (len(full_conversation_tokenized_ids) >= len(cached_input_ids_tensor) and
                torch.equal(full_conversation_tokenized_ids[:len(cached_input_ids_tensor)], cached_input_ids_tensor)):
            past_kv_for_generate = cached_kv_data["past_key_values"]
            # Only pass the new tokens (suffix) for generation
            input_ids_for_generation = full_conversation_tokenized_ids[len(cached_input_ids_tensor):].unsqueeze(0)

    # 5. LLM Inference
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids_for_generation,
            past_key_values=past_kv_for_generate,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )

    # 6. Extract the assistant's response
    # The generated sequence contains the input_ids_for_generation + newly generated tokens
    generated_sequence_ids = generation_output.sequences[0]
    
    # The assistant's response starts after the input tokens provided to generate
    assistant_response_tokens = generated_sequence_ids[input_ids_for_generation.shape[-1]:]
    assistant_response = tokenizer.decode(assistant_response_tokens, skip_special_tokens=True).strip()

    # Clean up potential partial sentences (optional)
    if assistant_response.endswith(tokenizer.eos_token):
        assistant_response = assistant_response[:-len(tokenizer.eos_token)].strip()
    
    # 7. Update conversation_history with assistant's response
    conversation_history[session_id].append({"role": "assistant", "content": assistant_response})

    # 8. Update KV Cache for the next turn
    # The new cached_input_ids should represent the full conversation up to this point
    updated_cached_input_ids = tokenizer.encode(full_conversation_text + assistant_response, return_tensors="pt")[0].tolist()

    kv_cache[session_id] = {
        "past_key_values": generation_output.past_key_values,
        "cached_input_ids": updated_cached_input_ids,
    }

    return {"session_id": session_id, "response": assistant_response}