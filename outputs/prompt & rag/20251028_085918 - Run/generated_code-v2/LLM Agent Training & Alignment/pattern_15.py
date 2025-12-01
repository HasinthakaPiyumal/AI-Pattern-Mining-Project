import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

def generate_dummy_data():
    users = [f"user_{i}" for i in range(50)]
    products = [f"product_{i}" for i in range(100)]
    categories = ["electronics", "clothing", "home", "books", "sports"]

    product_data = []
    for i in range(100):
        product_data.append({
            "item_id": f"product_{i}",
            "product_name": f"Awesome Product {i}",
            "category": categories[i % len(categories)],
            "description": f"This is an amazing product number {i}. It belongs to the {categories[i % len(categories)]} category and offers great value. Users love its features.",
            "features": f"Feature A, Feature B, Feature C for product {i}"
        })
    product_df = pd.DataFrame(product_data)

    interaction_data = []
    for u_id in users:
        liked_products = [products[j] for j in torch.randint(0, 100, (torch.randint(5, 15, (1,)).item(),)).tolist()]
        for p_id in liked_products:
            interaction_data.append({
                "user_id": u_id,
                "item_id": p_id,
                "interaction_type": "purchase",
                "label": 1
            })
        all_products_set = set(products)
        liked_products_set = set(liked_products)
        disliked_products = list(all_products_set - liked_products_set)
        num_disliked = torch.randint(5, 10, (1,)).item()
        for i in range(min(num_disliked, len(disliked_products))):
             interaction_data.append({
                "user_id": u_id,
                "item_id": disliked_products[i],
                "interaction_type": "browsed_not_bought",
                "label": 0
            })

    interaction_df = pd.DataFrame(interaction_data)

    df = pd.merge(interaction_df, product_df[['item_id', 'description']], on='item_id', how='left')

    user_profiles = {}
    for user_id in df['user_id'].unique():
        user_liked_items_desc = " ".join(df[(df['user_id'] == user_id) & (df['label'] == 1)]['description'].tolist())
        user_profiles[user_id] = f"User has previously shown interest in: {user_liked_items_desc[:500]}..."

    return df, user_profiles, product_df

class RecommendationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def setup_lora_model(model_name="bert-base-uncased", num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return tokenizer, peft_model

def train_model(tokenizer, model, train_dataset, val_dataset=None):
    training_args = TrainingArguments(
        output_dir="./lora_recommendation_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="accuracy" if val_dataset else None,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    return model, trainer

def generate_recommendations(user_id, num_recommendations, fine_tuned_model, tokenizer, user_profiles, product_df):
    user_profile_text = user_profiles.get(user_id, "No specific preferences found.")
    candidate_products = product_df['item_id'].tolist()
    product_descriptions = product_df.set_index('item_id')['description'].to_dict()

    recommendation_scores = []

    fine_tuned_model.eval()
    with torch.no_grad():
        for item_id in candidate_products:
            product_desc = product_descriptions.get(item_id, "")
            input_text = f"User preference: {user_profile_text}. Product description: {product_desc}"

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(fine_tuned_model.device) for k, v in inputs.items()}

            outputs = fine_tuned_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            like_probability = probabilities[:, 1].item()

            recommendation_scores.append({"item_id": item_id, "score": like_probability})

    recommendation_scores.sort(key=lambda x: x['score'], reverse=True)
    return recommendation_scores[:num_recommendations]

if __name__ == "__main__":
    print("Generating dummy data...")
    interaction_df, user_profiles, product_df = generate_dummy_data()

    training_texts = []
    training_labels = []
    for idx, row in interaction_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        label = row['label']

        user_profile_text = user_profiles.get(user_id, "No specific preferences found.")
        product_desc = product_df[product_df['item_id'] == item_id]['description'].iloc[0]

        input_text = f"User preference: {user_profile_text}. Product description: {product_desc}"
        training_texts.append(input_text)
        training_labels.append(label)

    print(f"Generated {len(training_texts)} training examples.")

    print("Setting up LoRA model...")
    tokenizer, lora_model = setup_lora_model()

    print("Tokenizing training data...")
    train_encodings = tokenizer(training_texts, truncation=True, padding=True, max_length=512)
    train_dataset = RecommendationDataset(train_encodings, training_labels)

    print("Starting LoRA fine-tuning...")
    fine_tuned_model, trainer = train_model(tokenizer, lora_model, train_dataset)
    print("LoRA fine-tuning complete.")

    target_user = "user_10"
    num_recs = 5
    print(f"\nGenerating top {num_recs} recommendations for {target_user}...")
    recommendations = generate_recommendations(target_user, num_recs, fine_tuned_model, tokenizer, user_profiles, product_df)

    print(f"Top {num_recs} recommendations for {target_user}:")
    for rec in recommendations:
        product_name = product_df[product_df['item_id'] == rec['item_id']]['product_name'].iloc[0]
        print(f"- {product_name} (ID: {rec['item_id']}) with score: {rec['score']:.4f}")

    print("Demonstration complete.")