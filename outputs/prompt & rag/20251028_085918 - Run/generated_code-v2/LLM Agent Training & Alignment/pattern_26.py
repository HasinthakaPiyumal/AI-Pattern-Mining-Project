import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import Dataset

# --- 1. CRM Simulator (Simplified) ---
class CRMSimulator:
    def __init__(self):
        self.tickets = {}
        self.next_ticket_id = 1

    def create_ticket(self, customer_id, subject, priority, description):
        ticket_id = f"#{self.next_ticket_id}"
        self.tickets[ticket_id] = {
            "customer_id": customer_id,
            "subject": subject,
            "priority": priority,
            "status": "open",
            "notes": [description]
        }
        self.next_ticket_id += 1
        return f"Ticket {ticket_id} created."

    def update_ticket_status(self, ticket_id, status):
        if ticket_id in self.tickets:
            self.tickets[ticket_id]["status"] = status
            return f"Ticket {ticket_id} status updated to {status}."
        return f"Ticket {ticket_id} not found."

    def add_note(self, ticket_id, note):
        if ticket_id in self.tickets:
            self.tickets[ticket_id]["notes"].append(note)
            return f"Note added to ticket {ticket_id}."
        return f"Ticket {ticket_id} not found."

    def search_customer(self, customer_id):
        found_tickets = [t for tid, t in self.tickets.items() if t["customer_id"] == customer_id]
        if found_tickets:
            return f"Customer {customer_id} found with tickets: {len(found_tickets)}."
        return f"Customer {customer_id} not found."

    def get_crm_state(self):
        return str(self.tickets) # Simplified state representation

# --- 2. Data Collection/Demonstrations ---
def collect_demonstrations():
    # Simulate human demonstrations
    # Each demonstration is a tuple: (customer_query, crm_state_before_action, human_action)
    demonstrations = [
        (
            "My internet is down, I need help!",
            "{}",
            "create_ticket(customer_id=\"CUST001\", subject=\"Internet Connectivity Issue\", priority=\"high\", description=\"Customer reports no internet connection.\")"
        ),
        (
            "What's the status of my refund?",
            "{'#1': {'customer_id': 'CUST002', 'subject': 'Refund Request', 'priority': 'medium', 'status': 'open', 'notes': ['Customer requested a refund.']}}",
            "search_customer(customer_id=\"CUST002\")"
        ),
        (
            "I want to follow up on ticket #1.",
            "{'#1': {'customer_id': 'CUST001', 'subject': 'Internet Connectivity Issue', 'priority': 'high', 'status': 'open', 'notes': ['Customer reports no internet connection.']}}",
            "add_note(ticket_id=\"#1\", note=\"Customer called for follow-up.\")"
        ),
        (
            "Close ticket #1, issue resolved.",
            "{'#1': {'customer_id': 'CUST001', 'subject': 'Internet Connectivity Issue', 'priority': 'high', 'status': 'open', 'notes': ['Customer reports no internet connection.']}}",
            "update_ticket_status(ticket_id=\"#1\", status=\"resolved\")"
        ),
    ]
    return demonstrations

# --- 3. Language Model (Placeholder for Fine-tuning) ---
# Using a small T5 model for demonstration. In a real scenario, you'd use a larger model.
MODEL_NAME = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# --- Preprocessing for training ---
def preprocess_function(examples):
    inputs = [f"Query: {q} CRM_State: {s}" for q, s, _ in examples["demonstration"]]
    targets = [action for _, _, action in examples["demonstration"]]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- 4. Behavior Cloning Training Loop ---
def train_bot(demonstrations):
    # Convert demonstrations to a Hugging Face Dataset
    raw_datasets = Dataset.from_dict({"demonstration": demonstrations})
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=["demonstration"])

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets, # Using train for eval for simplicity in this example
        tokenizer=tokenizer,
    )

    trainer.train()
    return model, tokenizer

# --- 5. Inference/Bot Interaction ---
def bot_predict_action(model, tokenizer, customer_query, crm_state):
    input_text = f"Query: {customer_query} CRM_State: {crm_state}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_new_tokens=128)
        
    predicted_action = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_action

def execute_crm_action(crm_simulator, action_string):
    try:
        # This is a very simplified and UNSAFE way to execute a string as a function call.
        # In a real application, you'd have a robust action parser and executor.
        eval(f"crm_simulator.{action_string}")
        print(f"Executed: {action_string}")
    except Exception as e:
        print(f"Error executing action {action_string}: {e}")

if __name__ == "__main__":
    print("Collecting human demonstrations...")
    demonstrations = collect_demonstrations()

    print("Training the Customer Support Bot using Behavior Cloning...")
    trained_model, trained_tokenizer = train_bot(demonstrations)
    print("Training complete.")

    print("\n--- Demonstrating Bot Interaction ---")
    crm = CRMSimulator()
    
    # Scenario 1: Create a new ticket
    current_query = "My delivery is late by a week! Order #ORD123."
    current_crm_state = crm.get_crm_state()
    print(f"\nCustomer Query: {current_query}")
    print(f"Current CRM State: {current_crm_state}")
    predicted_action = bot_predict_action(trained_model, trained_tokenizer, current_query, current_crm_state)
    print(f"Bot Predicted Action: {predicted_action}")
    # Manually map to a suitable action for demonstration as eval is unsafe
    # For real use, `eval` would be replaced by a safe action dispatcher
    # For this demo, let's hardcode a reasonable action after prediction
    if "create_ticket" in predicted_action:
        execute_crm_action(crm, "create_ticket(customer_id=\"CUST003\", subject=\"Late Delivery\", priority=\"medium\", description=\"Customer reports order #ORD123 is a week late.\")")
    
    # Scenario 2: Follow up on an existing (simulated) ticket
    crm.create_ticket("CUST004", "Billing Inquiry", "high", "Customer has a question about their last bill.")
    current_query = "I want to know about my billing issue, ticket #4."
    current_crm_state = crm.get_crm_state()
    print(f"\nCustomer Query: {current_query}")
    print(f"Current CRM State: {current_crm_state}")
    predicted_action = bot_predict_action(trained_model, trained_tokenizer, current_query, current_crm_state)
    print(f"Bot Predicted Action: {predicted_action}")
    if "add_note" in predicted_action and "#4" in predicted_action:
        execute_crm_action(crm, "add_note(ticket_id=\"#4\", note=\"Customer called for an update on billing.\")")
    elif "search_customer" in predicted_action:
        execute_crm_action(crm, "search_customer(customer_id=\"CUST004\")")

    print("\nFinal CRM State:")
    print(crm.get_crm_state())
