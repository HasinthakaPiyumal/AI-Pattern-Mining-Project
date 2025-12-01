import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Suppress some Hugging Face output for cleaner execution
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# sys.stdout = open(os.devnull, "w") # Redirect stdout

# 1. Data Collection and Preparation (Mock Data)
mock_data = [
    {"query": "Please create a new task named 'Prepare Q3 report' for next Monday.", "command": "create task Prepare Q3 report due next Monday"},
    {"query": "Assign the 'Bug fix for login' task to John.", "command": "assign Bug fix for login to John"},
    {"query": "What's the status of 'Website redesign' project?", "command": "get status Website redesign"},
    {"query": "Create 'Review marketing strategy' due end of month.", "command": "create task Review marketing strategy due end of month"},
    {"query": "Assign 'Client meeting' to Alice today.", "command": "assign Client meeting to Alice"},
    {"query": "Complete 'User onboarding flow' task.", "command": "complete User onboarding flow"},
    {"query": "Show me all tasks due this week.", "command": "list tasks due this week"},
    {"query": "Change priority of 'Database migration' to high.", "command": "set priority Database migration to high"},
    {"query": "Can you make 'Team retrospective' for Friday?", "command": "create task Team retrospective due Friday"},
    {"query": "Who is working on 'Mobile app development' project?", "command": "get assignee Mobile app development"},
    {"query": "Mark 'User story writing' as done.", "command": "complete User story writing"},
    {"query": "Set 'Sprint planning' for next Tuesday.", "command": "create task Sprint planning due next Tuesday"},
    {"query": "Tell me about 'Frontend refactor' status.", "command": "get status Frontend refactor"},
    {"query": "Add a new task 'API documentation' due in two weeks.", "command": "create task API documentation due in two weeks"},
    {"query": "Move 'Marketing campaign' to Sarah.", "command": "assign Marketing campaign to Sarah"}
]

# Create mappings for commands to numerical labels
unique_commands = sorted(list(set([item["command"] for item in mock_data])))
command_to_id = {cmd: i for i, cmd in enumerate(unique_commands)}
id_to_command = {i: cmd for cmd, i in command_to_id.items()}
num_labels = len(unique_commands)

# 4. Tokenization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["query"], truncation=True, padding="max_length", max_length=128)
    tokenized_inputs["labels"] = [command_to_id[cmd] for cmd in examples["command"]]
    return tokenized_inputs

# Convert mock data to a Hugging Face Dataset
raw_datasets = Dataset.from_list(mock_data)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["query", "command"])

# Split for training and evaluation (simple split for demonstration)
train_size = int(0.8 * len(tokenized_datasets))
# Ensure eval_size is at least 1, handle cases where data is too small
if len(tokenized_datasets) - train_size < 1:
    train_size = len(tokenized_datasets) - 1 if len(tokenized_datasets) > 1 else len(tokenized_datasets)
eval_size = len(tokenized_datasets) - train_size

train_dataset = tokenized_datasets.select(range(train_size))
eval_dataset = tokenized_datasets.select(range(train_size, train_size + eval_size))

# 2. Pre-trained Language Model
# 3. Behavior Cloning (Fine-tuning)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_chatbot_model")
tokenizer.save_pretrained("./fine_tuned_chatbot_model")

# sys.stdout = sys.__stdout__ # Restore stdout

# 5. Inference
loaded_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_chatbot_model")
loaded_model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_chatbot_model")

def predict_command(query):
    inputs = loaded_tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return id_to_command[predicted_class_id]

print("\n--- Inference Examples ---")
test_queries = [
    "Create a new task for 'Team building event' due next month.",
    "Assign the task 'Database migration' to Bob.",
    "What is the status of 'Website redesign project'?",
    "I need to create a task 'Review design mockups' by Wednesday.",
    "Complete 'User story writing' now.",
    "Set 'Sprint planning' for next week."
]

for query in test_queries:
    predicted_command = predict_command(query)
    print(f"User Query: \"{query}\"")
    print(f"Predicted Command: \"{predicted_command}\"\n")
