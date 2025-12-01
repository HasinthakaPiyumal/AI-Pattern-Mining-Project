import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import random

# 1. Base LLM Selection and Loading
# Using a small model for demonstration purposes (e.g., facebook/opt-125m)
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# Ensure tokenizer has a pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Or use eos_token if suitable
    model.resize_token_embeddings(len(tokenizer))

# 2. Efficient Fine-tuning Module (LoRA/QLoRA) - Setup for demonstration
# For actual QLoRA, you'd load the model with load_in_4bit=True and bnb_4bit_quant_type="nf4"
# and pass quantization_config to AutoModelForCausalLM.from_pretrained

# Prepare model for K-bit training (important for QLoRA, even if not fully QLoRA here)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# print(model.print_trainable_parameters()) # Uncomment to see trainable parameters

# 3. E-commerce Domain Dataset Preparation (Dummy Data)
# In a real scenario, this would be loaded from files and preprocessed
dummy_data = {
    "conversation": [
        "I am looking for a new smartphone.",
        "Do you have any recommendations for a gaming laptop?",
        "What kind of headphones are good for working out?",
        "Show me some affordable smartwatches.",
        "I need a new pair of running shoes."
    ],
    "product_recommendation": [
        "Consider the 'Galaxy S23' for its camera, or the 'iPhone 15' for its ecosystem.",
        "The 'Alienware m18' offers top-tier performance, or the 'ROG Zephyrus G14' for portability.",
        "The 'JBL Reflect Flow PRO' are waterproof and have great sound, or 'Apple AirPods Pro' for seamless integration.",
        "The 'Amazfit Bip 3 Pro' is very affordable, or the 'Xiaomi Smart Band 7' for basic tracking.",
        "The 'Brooks Ghost 15' are known for comfort, or the 'Hoka Clifton 9' for maximum cushioning."
    ]
}

df = pd.DataFrame(dummy_data)
dataset = Dataset.from_pandas(df)

def format_prompt(sample):
    return f"User: {sample['conversation']}\nChatbot: {sample['product_recommendation']}{tokenizer.eos_token}"

dataset = dataset.map(format_prompt, batched=False)

# In a real fine-tuning, you would tokenize the formatted_text and train the model
# using a Trainer from transformers or a custom training loop.
# For this demo, we skip actual training and simulate inference with the LoRA-enabled model.

# 4. Recommendation Generation Engine
class RecommendationEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device=0 if torch.cuda.is_available() else -1
        )

    def get_recommendation(self, user_query, max_new_tokens=100, temperature=0.7):
        prompt = f"User: {user_query}\nChatbot:"
        
        # Manually set pad_token_id if it's not set (for some models or custom tokens)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id
        )
        
        generated_text = outputs[0]["generated_text"]
        
        # Extract only the chatbot's response
        if "Chatbot:" in generated_text:
            response = generated_text.split("Chatbot:", 1)[1].strip()
            # Remove any trailing user input from the generated response
            if "User:" in response:
                response = response.split("User:", 1)[0].strip()
            # Remove EOS token if it's present at the end
            if response.endswith(self.tokenizer.eos_token):
                response = response[:-len(self.tokenizer.eos_token)].strip()
            return response
        return generated_text.strip() # Fallback

# 5. Chatbot Interface
def run_chatbot():
    print("Initializing recommendation engine...")
    engine = RecommendationEngine(model, tokenizer)
    print("Chatbot ready! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        recommendation = engine.get_recommendation(user_input)
        print(f"Chatbot: {recommendation}")

if __name__ == "__main__":
    run_chatbot()