import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import uvicorn
import random

# --- 1. Mock Data Generation ---
def generate_mock_product_data(num_products=100):
    products = []
    for i in range(num_products):
        products.append({
            "product_id": f"P{i:03d}",
            "name": f"Product {i}",
            "description": f"A high-quality product {i} with various features and benefits. It is popular among users interested in category {random.choice(['electronics', 'apparel', 'home_goods', 'books'])}.",
            "category": random.choice(["Electronics", "Apparel", "Home & Kitchen", "Books", "Sports"]),
            "price": round(random.uniform(10.0, 500.0), 2)
        })
    return pd.DataFrame(products)

def generate_mock_user_interaction_data(num_users=20, num_interactions=200, products_df=None):
    if products_df is None:
        products_df = generate_mock_product_data()
    
    interactions = []
    product_ids = products_df["product_id"].tolist()
    
    for i in range(num_users):
        user_id = f"U{i:02d}"
        for _ in range(random.randint(5, 15)): # Each user has 5-15 interactions
            product_id = random.choice(product_ids)
            event_type = random.choice(["view", "add_to_cart", "purchase"])
            interactions.append({
                "user_id": user_id,
                "product_id": product_id,
                "event_type": event_type,
                "timestamp": pd.Timestamp.now() - pd.Timedelta(days=random.randint(1, 30))
            })
    return pd.DataFrame(interactions)

# --- 2. Data Preprocessing for LLM Fine-tuning ---
def preprocess_data_for_llm_finetuning(user_interactions_df, products_df):
    # This function creates a dataset suitable for instruct-style fine-tuning
    # Example: "User U01 showed interest in Product P005 (description). Recommend something similar." -> "Product P010 (description)."
    
    training_examples = []
    for user_id in user_interactions_df["user_id"].unique():
        user_history = user_interactions_df[user_interactions_df["user_id"] == user_id]
        
        # Get unique products the user interacted with
        interacted_product_ids = user_history["product_id"].unique()
        if len(interacted_product_ids) < 2: # Need at least two for a simple recommendation scenario
            continue
            
        # Create a prompt: user's interacted products as context, next product as target
        for i in range(1, len(interacted_product_ids)): # Iterate to create sequences
            input_products = interacted_product_ids[:i]
            target_product_id = interacted_product_ids[i]
            
            input_text = f"Based on the user's past interactions with: "
            for pid in input_products:
                prod_desc = products_df[products_df["product_id"] == pid]["description"].iloc[0]
                input_text += f"'{prod_desc}', "
            input_text = input_text.rstrip(', ') + ". What product might they be interested in next?"
            
            target_desc = products_df[products_df["product_id"] == target_product_id]["description"].iloc[0]
            output_text = f"User might like: '{target_desc}'."
            
            training_examples.append({"input": input_text, "output": output_text})
            
    return Dataset.from_pandas(pd.DataFrame(training_examples))

# --- 3. Base LLM Selection & Loading ---
def load_base_llm_and_tokenizer(model_name="facebook/opt-125m"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

# --- 4. Efficient Fine-tuning Module (LoRA Setup - placeholder for actual training) ---
def setup_lora_for_finetuning(model, tokenizer, training_dataset):
    # Configure LoRA
    lora_config = LoraConfig(
        r=8, # LoRA attention dimension
        lora_alpha=16, # Alpha parameter for LoRA scaling
        target_modules=["q_proj", "v_proj"], # Modules to apply LoRA to
        lora_dropout=0.05, # Dropout probability for LoRA layers
        bias="none", # Bias type for LoRA layers
        task_type=TaskType.CAUSAL_LM # Task type for causal language modeling
    )

    # Apply LoRA to the base model
    model = get_peft_model(model, lora_config)
    print("LoRA model prepared:")
    model.print_trainable_parameters()

    # Prepare dataset for training
    def tokenize_function(examples):
        # Combine input and output for Causal LM training
        full_text = [inp + " " + out for inp, out in zip(examples["input"], examples["output"])]
        tokenized_output = tokenizer(full_text, truncation=True, padding="max_length", max_length=128)
        return tokenized_output
    
    tokenized_dataset = training_dataset.map(tokenize_function, batched=True)
    
    # For Causal LM, labels are typically the input_ids shifted
    tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])

    # Placeholder for actual training setup
    # In a real scenario, you would use TrainingArguments and Trainer here.
    # For this example, we will just return the LoRA-adapted model and tokenized dataset
    # as a conceptual step for fine-tuning.
    
    # training_args = TrainingArguments(
    #     output_dir="./lora_finetuned_model",
    #     learning_rate=2e-4,
    #     per_device_train_batch_size=4,
    #     num_train_epochs=3,
    #     logging_dir="./logs",
    #     logging_steps=10,
    # )
    # 
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset,
    #     tokenizer=tokenizer,
    # )
    # trainer.train()
    # model.save_pretrained("./lora_finetuned_model")

    print("LoRA fine-tuning setup complete. (Actual training skipped in this example due to resource constraints.)")
    return model, tokenized_dataset # Return the LoRA-adapted model (conceptually fine-tuned)

# --- 5. Product Embedding & Vector Store (Simplified In-Memory) ---
class ProductVectorStore:
    def __init__(self, products_df):
        self.products_df = products_df
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # A good general purpose embedding model
        self.product_embeddings = self._generate_embeddings()
        self.product_id_to_idx = {pid: i for i, pid in enumerate(products_df["product_id"])}
        self.idx_to_product_id = {i: pid for i, pid in enumerate(products_df["product_id"])}

    def _generate_embeddings(self):
        print("Generating product embeddings...")
        descriptions = self.products_df["description"].tolist()
        embeddings = self.embedding_model.encode(descriptions, convert_to_tensor=True)
        print("Product embeddings generated.")
        return embeddings.cpu().numpy() # Move to CPU for numpy operations

    def retrieve_similar_products(self, query_text, top_k=5):
        query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.product_embeddings)[0]
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        
        recommended_products = []
        for idx in top_k_indices:
            product_id = self.idx_to_product_id[idx]
            product_info = self.products_df[self.products_df["product_id"] == product_id].iloc[0].to_dict()
            recommended_products.append(product_info)
        return recommended_products

    def get_product_description(self, product_id):
        return self.products_df[self.products_df["product_id"] == product_id]["description"].iloc[0]

# --- 6. Recommendation Generation Engine ---
def generate_recommendations_from_llm(llm_model, tokenizer, user_id, user_history_product_ids, product_vector_store, num_recommendations=3):
    # Build context from user history and similar products
    user_history_descriptions = [product_vector_store.get_product_description(pid) for pid in user_history_product_ids]
    history_text = ", ".join([f"'{desc}'" for desc in user_history_descriptions])
    
    # Retrieve similar products based on the user's latest interaction or overall history
    if user_history_descriptions:
        query_for_retrieval = user_history_descriptions[-1] # Use the most recent interaction for retrieval
    else:
        query_for_retrieval = "general product recommendations" # Fallback
    
    context_products = product_vector_store.retrieve_similar_products(query_for_retrieval, top_k=5)
    context_product_descriptions = [f"'{p['description']}'" for p in context_products]
    context_text = ", ".join(context_product_descriptions)

    prompt = f"Based on the user's past interest in: {history_text}. And similar products like: {context_text}. What 3 new products would you recommend for this user? Only list product descriptions, separated by commas."

    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    
    # Generate recommendations
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=100, # Limit generation length
            num_return_sequences=1,
            do_sample=True, # Enable sampling for more diverse recommendations
            top_k=50, 
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process to extract product descriptions (simple heuristic)
    recommendations_raw = generated_text.replace(prompt, "").strip()
    # Simple split, more robust parsing might be needed for production
    recommended_items = [item.strip() for item in recommendations_raw.split(',') if item.strip()]

    # Optionally, map back to product IDs if descriptions are unique enough or use another retrieval pass
    # For this example, we'll return the generated descriptions.
    return recommended_items[:num_recommendations]

# --- 7. API & Serving Layer (FastAPI) ---
app = FastAPI()

# Global variables to hold models and data
products_df = None
user_interactions_df = None
fine_tuned_llm_model = None
llm_tokenizer = None
product_vector_store = None

@app.on_event("startup")
async def startup_event():
    global products_df, user_interactions_df, fine_tuned_llm_model, llm_tokenizer, product_vector_store
    
    print("Initializing recommender system...")
    
    # Generate mock data
    products_df = generate_mock_product_data()
    user_interactions_df = generate_mock_user_interaction_data(products_df=products_df)
    
    # Load base LLM and tokenizer
    llm_model_name = "facebook/opt-125m"
    base_llm_model, llm_tokenizer = load_base_llm_and_tokenizer(llm_model_name)
    
    # Setup LoRA (conceptual fine-tuning)
    # Note: Actual training is resource-intensive and skipped here. 
    # The model returned is the LoRA-adapted base model, ready to *be* fine-tuned.
    # For a real application, you'd load a *trained* LoRA adapter here.
    
    # Create a dummy training dataset to demonstrate the setup
    dummy_training_data = preprocess_data_for_llm_finetuning(user_interactions_df, products_df)
    if len(dummy_training_data) > 0:
        fine_tuned_llm_model, _ = setup_lora_for_finetuning(base_llm_model, llm_tokenizer, dummy_training_data)
        # For this example, we'll just use the base model with LoRA layers initialized
        # as if it were fine-tuned, without actual training steps.
    else:
        print("Not enough data for dummy LoRA setup, using base model directly.")
        fine_tuned_llm_model = base_llm_model

    # Move model to GPU if available
    if torch.cuda.is_available():
        fine_tuned_llm_model.to("cuda")
        print("LLM moved to CUDA.")
    else:
        print("CUDA not available, LLM running on CPU. Performance may be slower.")

    # Initialize product vector store
    product_vector_store = ProductVectorStore(products_df)
    
    print("Recommender system initialized.")

@app.get("/recommend/{user_id}")
async def get_product_recommendations(user_id: str, num_recommendations: int = 3):
    if user_id not in user_interactions_df["user_id"].unique():
        return {"error": "User not found", "user_id": user_id}

    # Get user's interaction history
    user_history_df = user_interactions_df[user_interactions_df["user_id"] == user_id].sort_values(by="timestamp")
    user_history_product_ids = user_history_df["product_id"].tolist()
    
    # Generate recommendations using the (conceptually) fine-tuned LLM
    recommendations = generate_recommendations_from_llm(
        fine_tuned_llm_model,
        llm_tokenizer,
        user_id,
        user_history_product_ids,
        product_vector_store,
        num_recommendations
    )
    
    return {"user_id": user_id, "recommendations": recommendations}

# --- 8. Monitoring & Evaluation (Placeholder) ---
# In a real system, this would involve integrating with tools like WandB or MLflow
# and implementing metrics calculation (e.g., click-through rate, conversion rate).
# For this example, we'll just have a conceptual function.

def monitor_recommendation_quality():
    print("Monitoring recommendation quality... (e.g., A/B testing, user feedback analysis)")
    # Placeholder for actual monitoring logic
    pass

if __name__ == "__main__":
    # To run the FastAPI application:
    # 1. Save this code as recommender_system.py
    # 2. Install dependencies: pip install pandas torch transformers peft datasets sentence-transformers scikit-learn fastapi uvicorn
    # 3. Run from terminal: uvicorn recommender_system:app --host 0.0.0.0 --port 8000 --reload
    # 4. Access recommendations via: http://localhost:8000/recommend/U01?num_recommendations=3
    
    print("Starting FastAPI server...")
    # uvicorn.run(app, host="0.0.0.0", port=8000) # This line is typically called by the uvicorn command directly
    # For demonstration purposes, if you want to run it directly from this script
    # you'd uncomment the above line, but usually, uvicorn is invoked as a CLI tool.
    
    # As an alternative to running uvicorn directly in script for testing:
    # FastAPI is designed to be run with the `uvicorn` command line tool.
    # The __main__ block serves as a guide for how to launch it externally.
    print("Please run the application using 'uvicorn recommender_system:app --host 0.0.0.0 --port 8000 --reload'")

    # Example of how to call components directly for testing without FastAPI
    # products_df_test = generate_mock_product_data()
    # user_interactions_df_test = generate_mock_user_interaction_data(products_df=products_df_test)
    # 
    # base_llm, tok = load_base_llm_and_tokenizer()
    # dummy_train_data = preprocess_data_for_llm_finetuning(user_interactions_df_test, products_df_test)
    # if len(dummy_train_data) > 0:
    #    lora_model, _ = setup_lora_for_finetuning(base_llm, tok, dummy_train_data)
    # else:
    #    lora_model = base_llm
    # 
    # if torch.cuda.is_available():
    #    lora_model.to("cuda")
    # 
    # prod_vec_store = ProductVectorStore(products_df_test)
    # 
    # test_user_id = user_interactions_df_test["user_id"].iloc[0] # Get first user ID
    # test_user_history_pids = user_interactions_df_test[user_interactions_df_test["user_id"] == test_user_id]["product_id"].tolist()
    # 
    # if test_user_history_pids:
    #     print(f"\nGenerating recommendations for user {test_user_id}:")
    #     recs = generate_recommendations_from_llm(lora_model, tok, test_user_id, test_user_history_pids, prod_vec_store)
    #     print(recs)
    # else:
    #     print(f"No history for user {test_user_id}, skipping direct recommendation test.")

    monitor_recommendation_quality() # Conceptual call