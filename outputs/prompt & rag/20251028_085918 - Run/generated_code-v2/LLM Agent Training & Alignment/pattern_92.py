import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd

class PersonalizedProductRecommender:
    def __init__(self, model_name="gpt2", lora_r=8, lora_alpha=16, lora_dropout=0.05):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model, self.peft_config)
        self.model.print_trainable_parameters()

    def prepare_data(self, user_data_df, product_catalog_df):
        # Simulate creating recommendation prompts
        recommendation_data = []
        for index, row in user_data_df.iterrows():
            user_id = row["user_id"]
            browsing_history = row["browsing_history"]
            purchases = row["purchases"]
            preferences = row["preferences"]

            # Simple prompt for demonstration
            # In a real system, this would be more sophisticated and dynamic
            prompt_text = (
                f"User {user_id} has browsed: {', '.join(browsing_history)}. "
                f"Purchased: {', '.join(purchases)}. Preferences: {', '.join(preferences)}. "
                f"Recommend 3 relevant products from the catalog: "
            )
            # For fine-tuning, we'd typically have an expected output product list
            # Here, we'll just append some dummy product names for the target
            target_products = [f"product_{i}" for i in range(1, 4)] # Dummy target
            full_text = prompt_text + ", ".join(target_products) + self.tokenizer.eos_token
            recommendation_data.append({"text": full_text})

        return Dataset.from_pandas(pd.DataFrame(recommendation_data))

    def fine_tune(self, training_data, output_dir="./lora_finetuned_model"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2, # Small batch size for demo
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=1, # Single epoch for quick demo
            logging_steps=10,
            save_steps=10,
            fp16=torch.cuda.is_available(),
            optim="paged_adamw_8bit",
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=training_data,
            peft_config=self.peft_config,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=training_args,
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def generate_recommendations(self, user_context_prompt, num_recommendations=3, max_new_tokens=50):
        inputs = self.tokenizer(user_context_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id
        )
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process to extract recommendations (simple heuristic for demo)
        # In a real application, the LLM would be prompted to output a structured format (e.g., JSON)
        # and a more robust parsing mechanism would be used.
        recommendations = decoded_output.split("Recommend ")[-1]
        if "from the catalog:" in recommendations:
            recommendations = recommendations.split("from the catalog:")[-1].strip()
        recommendation_list = [rec.strip() for rec in recommendations.split(",") if rec.strip()]

        return recommendation_list[:num_recommendations]

if __name__ == "__main__":
    # 1. Dummy Data Ingestion and Preprocessing
    user_data = {
        "user_id": ["user_A", "user_B"],
        "browsing_history": [["product_X", "product_Y"], ["product_Z"]],
        "purchases": [["product_A"], ["product_B", "product_C"]],
        "preferences": [["electronics", "gadgets"], ["books", "fiction"]],
    }
    user_df = pd.DataFrame(user_data)

    product_catalog = {
        "product_id": ["product_1", "product_2", "product_3", "product_4", "product_5", "product_X", "product_Y", "product_Z", "product_A", "product_B", "product_C"],
        "description": [
            "High-end smartphone with advanced camera",
            "Noise-cancelling over-ear headphones",
            "Bestselling fantasy novel series",
            "Smartwatch with fitness tracking",
            "Portable Bluetooth speaker",
            "Gaming laptop", "4K monitor", "Ergonomic keyboard",
            "Wireless earbuds", "E-reader", "Coffee maker"
        ],
        "category": [
            "electronics", "electronics", "books", "electronics", "electronics",
            "electronics", "electronics", "electronics", "electronics", "books", "home_appliances"
        ]
    }
    product_df = pd.DataFrame(product_catalog)

    print("Initializing Recommender...")
    recommender = PersonalizedProductRecommender(model_name="gpt2") # Using a small model for faster demo
    print("Preparing training data...")
    training_dataset = recommender.prepare_data(user_df, product_df)
    print(f"Generated {len(training_dataset)} training samples.")
    print("Example training sample:")
    print(training_dataset[0]["text"])

    print("Starting LoRA Fine-tuning...")
    recommender.fine_tune(training_dataset)
    print("Fine-tuning complete. Model saved to ./lora_finetuned_model/")

    # 4. Recommendation Generation Module (Inference)
    print("\nGenerating Recommendations...")
    user_context_A = (
        "User user_A has browsed: Gaming laptop, 4K monitor. "
        "Purchased: Wireless earbuds. Preferences: electronics, gadgets. "
        "Recommend 3 relevant products from the catalog: "
    )
    recommendations_A = recommender.generate_recommendations(user_context_A)
    print(f"Recommendations for user A: {recommendations_A}")

    user_context_B = (
        "User user_B has browsed: Ergonomic keyboard. "
        "Purchased: E-reader, Coffee maker. Preferences: books, fiction. "
        "Recommend 3 relevant products from the catalog: "
    )
    recommendations_B = recommender.generate_recommendations(user_context_B)
    print(f"Recommendations for user B: {recommendations_B}")

    print("\nDemonstration complete.")
