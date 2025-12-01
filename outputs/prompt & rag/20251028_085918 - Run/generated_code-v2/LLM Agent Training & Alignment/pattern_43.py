import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import random

# 1. Data Models and Simulation
class Product(BaseModel):
    product_id: str
    name: str
    description: str
    category: str

class UserInteraction(BaseModel):
    user_id: str
    product_id: str
    interaction_type: str  # e.g., "view", "add_to_cart", "purchase"
    timestamp: str

def load_dummy_data():
    products = [
        Product(product_id="P001", name="Laptop Pro", description="High performance laptop", category="Electronics"),
        Product(product_id="P002", name="Smartphone X", description="Latest model smartphone", category="Electronics"),
        Product(product_id="P003", name="Bluetooth Headphones", description="Noise-cancelling headphones", category="Audio"),
        Product(product_id="P004", name="Gaming Mouse", description="Ergonomic gaming mouse", category="Accessories"),
        Product(product_id="P005", name="Smart Watch Sport", description="Fitness tracker with smart features", category="Wearables"),
    ]
    interactions = [
        UserInteraction(user_id="U001", product_id="P001", interaction_type="view", timestamp="2023-01-01T10:00:00"),
        UserInteraction(user_id="U001", product_id="P002", interaction_type="add_to_cart", timestamp="2023-01-01T10:10:00"),
        UserInteraction(user_id="U001", product_id="P001", interaction_type="purchase", timestamp="2023-01-01T11:00:00"),
        UserInteraction(user_id="U002", product_id="P003", interaction_type="view", timestamp="2023-01-02T12:00:00"),
        UserInteraction(user_id="U002", product_id="P005", interaction_type="add_to_cart", timestamp="2023-01-02T12:30:00"),
        UserInteraction(user_id="U003", product_id="P004", interaction_type="purchase", timestamp="2023-01-03T14:00:00"),
    ]
    return products, interactions

# 2. Data Preprocessing (Simplified)
class PreparedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def prepare_dataset(products, interactions, tokenizer):
    texts = []
    for interaction in interactions:
        product = next((p for p in products if p.product_id == interaction.product_id), None)
        if product:
            texts.append(f"User {interaction.user_id} {interaction.interaction_type} product {product.name} ({product.category}).")
            texts.append(f"Product description: {product.description}")

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    return PreparedDataset(encodings)

# 3. Efficient LLM Fine-tuning Module (Setup only, no actual training loop)
class LLMFineTuner:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.peft_model = None

    def configure_lora(self, lora_config):
        self.peft_model = get_peft_model(self.model, lora_config)

    def train(self, dataset, training_args):
        print(f"Mock training LLM with LoRA for {training_args.num_train_epochs} epochs...")
        # In a real scenario, this would involve Trainer or custom training loop
        # For demonstration, we'll simulate a trained model.
        # self.peft_model.train_model(dataset, training_args) 
        print("Mock training complete. Model is now 'fine-tuned'.")

    def get_fine_tuned_model(self):
        return self.peft_model if self.peft_model else self.model

# 4. Recommendation Inference Engine
class RecommendationEngine:
    def __init__(self, model, tokenizer, products):
        self.model = model
        self.tokenizer = tokenizer
        self.products = {p.product_id: p for p in products}

    def generate_recommendations(self, user_id: str, num_recommendations: int = 3):
        user_prompt = f"Based on user {user_id}'s past interactions, what products should be recommended? Products: {[p.name for p in self.products.values()]}. Recommend:"
        inputs = self.tokenizer(user_prompt, return_tensors="pt", padding=True, truncation=True)

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Simple parsing of recommendations (very basic and often requires more sophisticated NLP)
        recommended_product_names = []
        for product_name in self.products.values():
            if product_name.name in generated_text and len(recommended_product_names) < num_recommendations:
                recommended_product_names.append(product_name.name)
        
        # Fallback to random if LLM doesn't generate valid product names
        if not recommended_product_names:
            all_product_names = [p.name for p in self.products.values()]
            recommended_product_names = random.sample(all_product_names, min(num_recommendations, len(all_product_names)))

        return {"user_id": user_id, "recommendations": recommended_product_names}

# 5. FastAPI Application
app = FastAPI()

tuner: LLMFineTuner = None
engine: RecommendationEngine = None
dummy_products = []

@app.on_event("startup")
async def startup_event():
    global tuner, engine, dummy_products
    dummy_products, dummy_interactions = load_dummy_data()

    tuner = LLMFineTuner()
    
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"], # Common for GPT2/causal LMs
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    tuner.configure_lora(lora_config)
    
    # Prepare dataset for mock training
    dataset = prepare_dataset(dummy_products, dummy_interactions, tuner.tokenizer)
    training_args = TrainingArguments(
        output_dir="./lora_results",
        num_train_epochs=1, 
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        logging_dir="./lora_logs",
        logging_steps=10,
        disable_tqdm=True,
        no_cuda=True, # For CPU only demo
    )
    tuner.train(dataset, training_args)

    engine = RecommendationEngine(tuner.get_fine_tuned_model(), tuner.tokenizer, dummy_products)


class RecommendRequest(BaseModel):
    user_id: str
    num_recommendations: int = 3

@app.post("/recommend")
async def get_product_recommendations(request: RecommendRequest):
    if engine is None:
        return {"error": "Recommendation engine not initialized"}, 500
    
    recommendations = engine.generate_recommendations(request.user_id, request.num_recommendations)
    return recommendations

if __name__ == "__main__":
    # To run this script:
    # 1. Install necessary libraries: pip install torch transformers peft fastapi uvicorn pydantic
    # 2. Run the script: python recommendation_system.py
    # 3. Access the API (e.g., with curl or Postman) at http://127.0.0.1:8000/recommend
    # Example curl request:
    # curl -X POST -H "Content-Type: application/json" -d '{"user_id": "U001", "num_recommendations": 2}' http://127.0.0.1:8000/recommend
    uvicorn.run(app, host="127.0.0.1", port=8000)