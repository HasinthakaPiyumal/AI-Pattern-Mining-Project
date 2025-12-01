import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import random
import re
import os

def generate_simulated_data(num_samples=100):
    data = []
    customer_ids = [f"CUST{i:03d}" for i in range(1, 21)]
    possible_states = [
        "Customer profile open for ID {cid}",
        "Query history displayed for ID {cid}",
        "Order details shown for ID {cid}",
        "No active customer session",
        "Billing information loaded for ID {cid}"
    ]
    
    possible_queries_with_cid = [
        "What is the last order for {cid}?",
        "Update the shipping address for {cid} to 123 Main St",
        "Create a support ticket for {cid} for a broken product",
        "What is the current plan for {cid}?",
        "Can I get contact details for {cid}?",
        "How many orders has {cid} placed?",
        "Refund the last purchase for {cid}",
        "Escalate the issue for {cid}",
        "Check payment history for {cid}"
    ]

    possible_commands_map = {
        "What is the last order for {cid}?": "GET_ORDER_HISTORY {cid}",
        "Update the shipping address for {cid} to 123 Main St": "UPDATE_ADDRESS {cid} 123 Main St",
        "Create a support ticket for {cid} for a broken product": "CREATE_TICKET {cid} BROKEN_PRODUCT",
        "What is the current plan for {cid}?".format(cid="{cid}"): "GET_PLAN_DETAILS {cid}",
        "Can I get contact details for {cid}?".format(cid="{cid}"): "GET_CONTACT_DETAILS {cid}",
        "How many orders has {cid} placed?".format(cid="{cid}"): "GET_ORDER_COUNT {cid}",
        "Refund the last purchase for {cid}".format(cid="{cid}"): "REFUND_LAST_ORDER {cid}",
        "Escalate the issue for {cid}".format(cid="{cid}"): "ESCALATE_TICKET {cid}",
        "Check payment history for {cid}".format(cid="{cid}"): "GET_PAYMENT_HISTORY {cid}"
    }

    for _ in range(num_samples):
        cid = random.choice(customer_ids)
        crm_state = random.choice(possible_states).format(cid=cid)
        
        query_template = random.choice(possible_queries_with_cid)
        customer_query = query_template.format(cid=cid)
        
        human_command_template = possible_commands_map.get(query_template)
        human_command = human_command_template.format(cid=cid) if human_command_template else f"UNKNOWN_COMMAND {cid} {customer_query.split()[0].upper()}"

        data.append({
            "crm_state": crm_state,
            "customer_query": customer_query,
            "human_command": human_command
        })
    
    return pd.DataFrame(data)

class CRMSimulator:
    def __init__(self):
        self._customer_data = {
            "CUST001": {"address": "100 Elm St", "plan": "Premium", "orders": ["ORD001", "ORD002"]},
            "CUST002": {"address": "200 Oak Ave", "plan": "Basic", "orders": ["ORD003"]},
            "CUST003": {"address": "300 Pine Ln", "plan": "Basic", "orders": ["ORD004", "ORD005", "ORD006"]},
            "CUST004": {"address": "400 Maple Rd", "plan": "Premium", "orders": ["ORD007"]},
        }
        self._current_customer_id = None
        self._last_crm_response = "System ready."

    def _extract_cid_from_command(self, command):
        match = re.search(r"CUST\d{3}", command)
        if match:
            return match.group(0)
        return None

    def get_state(self):
        if self._current_customer_id:
            return f"Customer profile open for ID {self._current_customer_id}. Last response: {self._last_crm_response}"
        return f"No active customer session. Last response: {self._last_crm_response}"

    def execute_command(self, command):
        self._last_crm_response = "Command processed."
        
        cid_from_command = self._extract_cid_from_command(command)
        if cid_from_command:
            self._current_customer_id = cid_from_command
        
        parts = command.split(" ", 1)
        action = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        response = "Command not understood."

        if action == "GET_ORDER_HISTORY":
            cid = args.strip()
            orders = self._customer_data.get(cid, {}).get("orders", [])
            response = f"Order history for {cid}: {', '.join(orders) if orders else 'No orders found.'}"
        elif action == "UPDATE_ADDRESS":
            cid, new_address = args.split(" ", 1)
            if cid in self._customer_data:
                self._customer_data[cid]["address"] = new_address
                response = f"Address for {cid} updated to {new_address}."
            else:
                response = f"Customer {cid} not found."
        elif action == "CREATE_TICKET":
            cid, issue = args.split(" ", 1)
            response = f"Ticket created for {cid} regarding: {issue}."
        elif action == "GET_PLAN_DETAILS":
            cid = args.strip()
            plan = self._customer_data.get(cid, {}).get("plan", "Unknown")
            response = f"Plan for {cid}: {plan}."
        elif action == "GET_CONTACT_DETAILS":
            cid = args.strip()
            response = f"Contact details for {cid}: (Simulated: email@example.com, 555-1234)."
        elif action == "GET_ORDER_COUNT":
            cid = args.strip()
            count = len(self._customer_data.get(cid, {}).get("orders", []))
            response = f"Customer {cid} has placed {count} orders."
        elif action == "REFUND_LAST_ORDER":
            cid = args.strip()
            response = f"Refund processed for last order of {cid} (Simulated)."
        elif action == "ESCALATE_TICKET":
            cid = args.strip()
            response = f"Issue for {cid} escalated to specialist (Simulated)."
        elif action == "GET_PAYMENT_HISTORY":
            cid = args.strip()
            response = f"Payment history for {cid}: (Simulated: transactions list)."
        elif action == "UNKNOWN_COMMAND":
            response = f"Attempted unknown command: {command}. Please rephrase."
        else:
            response = f"Unrecognized CRM command: {command}"
        
        self._last_crm_response = response
        return response

class CRMTrainingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels["input_ids"][idx])
        return item

    def __len__(self):
        return len(self.labels["input_ids"])

def train_crm_model(df, model_name="t5-small", output_dir="./crm_model_bc_artifacts"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = [f"{row['crm_state']} | {row['customer_query']}" for _, row in df.iterrows()]
    labels = df["human_command"].tolist()

    input_encodings = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
    label_encodings = tokenizer(labels, truncation=True, padding="max_length", max_length=128)

    dataset = CRMTrainingDataset(input_encodings, label_encodings)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs_crm_bc",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=20,
        save_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer

class CustomerSupportAgent:
    def __init__(self, model_path, crm_simulator):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.crm_simulator = crm_simulator

    def handle_customer_query(self, customer_query):
        crm_state = self.crm_simulator.get_state()
        input_text = f"{crm_state} | {customer_query}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95
        )
        predicted_command = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        crm_response = self.crm_simulator.execute_command(predicted_command)
        
        return predicted_command, crm_response

if __name__ == "__main__":
    print("Generating simulated data...")
    df_data = generate_simulated_data(num_samples=200)
    print(f"Generated {len(df_data)} samples.")
    
    print("Starting model training...")
    try:
        trained_model, trained_tokenizer = train_crm_model(df_data, output_dir="./crm_agent_model_bc_artifacts")
        print("Model training complete. Model saved to ./crm_agent_model_bc_artifacts")
    except Exception as e:
        print(f"Error during model training (this is expected if CUDA is not available or if resources are limited for t5-small): {e}")
        print("Falling back to a dummy agent for demonstration purposes.")
        class DummyTokenizer:
            def __init__(self): pass
            def __call__(self, text, *args, **kwargs):
                return {"input_ids": torch.tensor([[1,2,3]]), "attention_mask": torch.tensor([[1,1,1]])}
            def decode(self, ids, *args, **kwargs):
                return "DUMMY_COMMAND CUST001 DUMMY_ARG"
        class DummyModel:
            def __init__(self): pass
            def generate(self, *args, **kwargs):
                return torch.tensor([[101, 102, 103, 104]])
        
        trained_tokenizer = DummyTokenizer()
        trained_model = DummyModel()
        os.makedirs("./crm_agent_model_bc_artifacts", exist_ok=True)

    crm_system = CRMSimulator()
    agent = CustomerSupportAgent(model_path="./crm_agent_model_bc_artifacts", crm_simulator=crm_system)

    print("\n--- Agent in action ---")
    
    test_queries = [
        "What is the last order for CUST001?",
        "Update the shipping address for CUST002 to 456 Bay Dr",
        "Create a support ticket for CUST003 about payment issue",
        "What is the current plan for CUST004?",
        "Can I get contact details for CUST001?",
        "How many orders has CUST002 placed?",
        "Refund the last purchase for CUST003",
        "Escalate the issue for CUST004",
        "Check payment history for CUST001"
    ]

    for i, query in enumerate(test_queries):
        command, response = agent.handle_customer_query(query)
        print(f"Customer Query {i+1}: '{query}'")
        print(f"Agent Predicted Command: '{command}'")
        print(f"CRM Response: '{response}'")
        print(f"Current CRM State: '{crm_system.get_state()}'\n")

    print("Demonstration complete.")
