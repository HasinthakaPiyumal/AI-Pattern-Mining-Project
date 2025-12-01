import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.generation import GenerateOutput

class KVCacheManager:
    def __init__(self):
        self.cache = {}

    def get_cache(self, prefix_token_ids_tuple):
        return self.cache.get(prefix_token_ids_tuple)

    def set_cache(self, prefix_token_ids_tuple, past_key_values, attention_mask_length):
        self.cache[prefix_token_ids_tuple] = (past_key_values, attention_mask_length)

class IntelligentChatbot:
    def __init__(self, model_name="distilgpt2", system_prompt="You are a helpful customer support assistant."):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.kv_cache_manager = KVCacheManager()

        self.system_prompt = system_prompt
        self.conversation_history_tokens = []
        self.current_past_key_values = None
        self.current_attention_mask_length = 0

        self._prefill_system_prompt()

    def _prefill_system_prompt(self):
        system_prompt_tokens = self.tokenizer.encode(self.system_prompt, return_tensors="pt")
        system_prompt_tuple = tuple(system_prompt_tokens[0].tolist())

        cached_data = self.kv_cache_manager.get_cache(system_prompt_tuple)
        if cached_data:
            self.current_past_key_values, self.current_attention_mask_length = cached_data
            self.conversation_history_tokens.extend(system_prompt_tokens[0].tolist())
            return

        with torch.no_grad():
            outputs = self.model(
                input_ids=system_prompt_tokens,
                attention_mask=torch.ones_like(system_prompt_tokens),
                use_cache=True,
                return_dict=True
            )
        
        self.current_past_key_values = outputs.past_key_values
        self.current_attention_mask_length = system_prompt_tokens.shape[1]
        self.conversation_history_tokens.extend(system_prompt_tokens[0].tolist())
        
        self.kv_cache_manager.set_cache(
            system_prompt_tuple,
            self.current_past_key_values,
            self.current_attention_mask_length
        )

    def chat(self, user_input_text, max_new_tokens=100):
        user_input_ids = self.tokenizer.encode(user_input_text, return_tensors="pt")
        new_input_len = user_input_ids.shape[1]
        
        attention_mask = torch.ones(
            1, self.current_attention_mask_length + new_input_len, dtype=torch.long
        )

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            return_dict_in_generate=True,
            use_cache=True
        )

        with torch.no_grad():
            output_sequence = self.model.generate(
                input_ids=user_input_ids,
                attention_mask=attention_mask,
                past_key_values=self.current_past_key_values,
                generation_config=generation_config
            )
        
        full_output_ids = output_sequence.sequences[0].tolist()
        
        start_of_generation_idx = self.current_attention_mask_length + new_input_len
        
        newly_generated_tokens = full_output_ids[start_of_generation_idx:]
        response_text = self.tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()

        self.conversation_history_tokens.extend(user_input_ids[0].tolist())
        self.conversation_history_tokens.extend(newly_generated_tokens)
        
        self.current_past_key_values = output_sequence.past_key_values
        self.current_attention_mask_length = len(full_output_ids)

        return response_text

if __name__ == "__main__":
    chatbot = IntelligentChatbot(model_name="distilgpt2")

    print("--- Intelligent Customer Support Chatbot Ready (Type 'exit' to end) ---")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = chatbot.chat(user_input)
        print(f"Bot: {response}")
        print(f"Full context length for next turn: {chatbot.current_attention_mask_length} tokens.")
        print("-" * 30)