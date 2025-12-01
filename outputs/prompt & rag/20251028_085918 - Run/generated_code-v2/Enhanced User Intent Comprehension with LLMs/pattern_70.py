import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Synthetic Dataset
queries = [
    "book me a flight to New York", "I need a flight from London to Paris", "Find flights for next week",
    "What's the weather like in Berlin?", "Give me the forecast for tomorrow", "Is it sunny in Rome?",
    "Play some jazz music", "Next song, please", "Turn up the volume",
    "Set a timer for 10 minutes", "Remind me to call John at 3 PM", "Add milk to my shopping list"
]
intents = [
    "flight_booking", "flight_booking", "flight_booking",
    "weather_inquiry", "weather_inquiry", "weather_inquiry",
    "music_control", "music_control", "music_control",
    "task_management", "task_management", "task_management"
]

labels = sorted(list(set(intents)))
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for i, label in enumerate(labels)}

encoded_intents = [label_to_id[intent] for intent in intents]

# Split data
train_queries, test_queries, train_encoded_intents, test_encoded_intents = train_test_split(
    queries, encoded_intents, test_size=0.2, random_state=42
)

# 2. Data Preprocessing Module
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings = tokenizer(train_queries, truncation=True, padding=True)
test_encodings = tokenizer(test_queries, truncation=True, padding=True)

train_dataset = IntentDataset(train_encodings, train_encoded_intents)
test_dataset = IntentDataset(test_encodings, test_encoded_intents)

# 3. Intent Classification Model Module
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(labels)
)

# 4. Training and Evaluation Module
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 5. Inference Module
def predict_intent(query, model, tokenizer, label_mapping):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return label_mapping[predicted_class_id]

# Example Usage of Inference Module
print("\n--- Inference Examples ---")
new_queries = [
    "I want to book a ticket to Sydney",
    "What will the weather be like next Tuesday?",
    "Can you play some rock music?",
    "Remind me about the meeting at 10 AM"
]

for query in new_queries:
    predicted_intent = predict_intent(query, model, tokenizer, id_to_label)
    print(f"Query: '{query}' -> Predicted Intent: '{predicted_intent}'")
