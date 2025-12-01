import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score

# 1. Simulate Data
# For simplicity, we create a dummy dataset where the LLM learns to classify
# a user's product interest into broad categories based on a query.
# In a real scenario, this would involve richer user interaction data, product descriptions, etc.

data = {
    "text": [
        "user wants a new smartphone, recommend best one",
        "looking for a durable laptop for coding",
        "cheap headphones for everyday use",
        "high-end gaming pc with rgb lights",
        "budget friendly smartwatch with fitness tracking",
        "i need a professional camera for photography",
        "comfortable office chair for long hours",
        "wireless mouse for productivity",
        "portable bluetooth speaker with good bass",
        "fast ssd drive for my desktop"
    ],
    "labels": [
        0, # consumer_electronics
        1, # computer_accessories
        0, # consumer_electronics
        1, # computer_accessories
        0, # consumer_electronics
        1, # computer_accessories
        0, # consumer_electronics
        1, # computer_accessories
        0, # consumer_electronics
        1  # computer_accessories
    ]
}
id2label = {0: "consumer_electronics", 1: "computer_accessories"}
label2id = {"consumer_electronics": 0, "computer_accessories": 1}

dataset = Dataset.from_dict(data)

# 2. Load a Pre-trained LLM and Tokenizer
# We use a small, pre-trained BERT-like model (DistilBERT) for demonstration.
# A larger LLM would be used in a production system.
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
)

# Add a padding token if it doesn't exist, and resize embeddings if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    base_model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Split into very small train and test sets for quick demonstration
train_dataset = tokenized_dataset.shuffle(seed=42).select(range(8))
eval_dataset = tokenized_dataset.shuffle(seed=42).select(range(8, 10))

# 3. Implement LoRA (Low-Rank Adaptation)
# Configure LoRA parameters to update only a small subset of the model's parameters.
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # Sequence Classification task
    r=8,  # LoRA attention dimension (rank)
    lora_alpha=16, # Alpha parameter for LoRA scaling
    lora_dropout=0.1, # Dropout probability for LoRA layers
    target_modules=["q_lin", "v_lin"], # Target specific attention layers for LoRA injection
)

# Get PEFT model, which wraps the base model with LoRA layers
model = get_peft_model(base_model, lora_config)
print("\nLoRA model trainable parameters:")
model.print_trainable_parameters() # Observe the significantly reduced number of trainable parameters

# 4. Define Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2, # Small batch size for demo purposes
    per_device_eval_batch_size=2,
    num_train_epochs=3, # Few epochs for a quick demonstration
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 5. Train the model using LoRA fine-tuning
print("\nStarting LoRA fine-tuning...")
trainer.train()
print("\nLoRA fine-tuning complete.")

# 6. Demonstrate Inference with the fine-tuned LoRA model
print("\nDemonstrating inference with the fine-tuned LoRA model:")

# Example 1: Query for an office chair
eval_text_1 = "I need a comfortable chair for my study room."
inputs_1 = tokenizer(eval_text_1, return_tensors="pt", truncation=True, padding="max_length")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}

with torch.no_grad():
    outputs_1 = model(**inputs_1)
    logits_1 = outputs_1.logits
    predicted_class_id_1 = torch.argmax(logits_1, dim=-1).item()

predicted_label_1 = model.config.id2label[predicted_class_id_1]
print(f"Query: '{eval_text_1}'")
print(f"Predicted Category: {predicted_label_1}")

# Example 2: Query for a new phone
eval_text_2 = "recommend the latest iphone model"
inputs_2 = tokenizer(eval_text_2, return_tensors="pt", truncation=True, padding="max_length")
inputs_2 = {k: v.to(device) for k, v in inputs_2.items()}
with torch.no_grad():
    outputs_2 = model(**inputs_2)
    logits_2 = outputs_2.logits
    predicted_class_id_2 = torch.argmax(logits_2, dim=-1).item()
predicted_label_2 = model.config.id2label[predicted_class_id_2]
print(f"Query: '{eval_text_2}'")
print(f"Predicted Category: {predicted_label_2}")