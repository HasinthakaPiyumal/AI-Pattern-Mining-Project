import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 1. Mock CRM Environment Interface/Simulator
class CRMSimulator:
    def __init__(self, customers_data=None):
        self.customers = customers_data if customers_data else {
            "1": {"name": "Alice", "email": "alice@example.com", "orders": ["A101", "B202"]},
            "2": {"name": "Bob", "email": "bob@example.com", "orders": ["C303"]}
        }
        self.current_state = "Initial CRM dashboard. Awaiting command."

    def execute_command(self, command):
        response = ""
        if "search_customer" in command:
            try:
                customer_id = command.split("id=")[1].split(")")[0].strip()
                customer_info = self.customers.get(customer_id)
                if customer_info:
                    response = f"Customer found: ID {customer_id}, Name: {customer_info['name']}, Email: {customer_info['email']}."
                else:
                    response = f"Customer with ID {customer_id} not found."
            except IndexError:
                response = "Invalid search_customer command format. Use: search_customer(id=<id>)"
        elif "view_order_history" in command:
            try:
                customer_id = command.split("customer_id=")[1].split(")")[0].strip()
                customer_info = self.customers.get(customer_id)
                if customer_info and 'orders' in customer_info:
                    response = f"Order history for {customer_info['name']}: {', '.join(customer_info['orders'])}"
                elif customer_info:
                    response = f"No order history found for {customer_info['name']}."
                else:
                    response = f"Customer with ID {customer_id} not found."
            except IndexError:
                response = "Invalid view_order_history command format. Use: view_order_history(customer_id=<id>)"
        elif "create_ticket" in command:
            try:
                parts = command.split("(")[1].split(")")[0].split(", ")
                ticket_details = {p.split("=")[0].strip(): p.split("=")[1].strip().strip("'") for p in parts}
                customer_id = ticket_details.get("customer_id")
                issue = ticket_details.get("issue")
                if customer_id and issue:
                    response = f"Ticket created for customer ID {customer_id} with issue: '{issue}'."
                else:
                    response = "Missing customer_id or issue in create_ticket command."
            except IndexError:
                response = "Invalid create_ticket command format. Use: create_ticket(customer_id=<id>, issue='<issue>')"
        else:
            response = "Unknown command. Available commands: search_customer, view_order_history, create_ticket."
        self.current_state = response
        return response

# 2. Data Collection Module (Mock Demonstrations)
class CustomerSupportDataset(Dataset):
    def __init__(self, tokenizer, demonstrations, max_length=128):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for obs, cmd in demonstrations:
            # Input: CRM state, Output: Command
            # For behavior cloning, we want the model to predict the command given the state
            # We concatenate obs and cmd for training, and set labels for cmd part.

            # Encode the observation (input context)
            obs_tokens = self.tokenizer.encode(obs, add_special_tokens=False)
            # Encode the command (target output)
            cmd_tokens = self.tokenizer.encode(cmd, add_special_tokens=False)

            # Combine them for causal language modeling objective
            full_sequence = obs_tokens + cmd_tokens + [self.tokenizer.eos_token_id]

            # Create labels: -100 for input, actual token IDs for target command
            labels = [-100] * len(obs_tokens) + cmd_tokens + [self.tokenizer.eos_token_id]

            # Truncate if necessary
            if len(full_sequence) > max_length:
                full_sequence = full_sequence[:max_length]
                labels = labels[:max_length]

            # Pad if necessary
            padding_length = max_length - len(full_sequence)
            self.input_ids.append(full_sequence + [self.tokenizer.pad_token_id] * padding_length)
            self.attention_mask.append([1] * len(full_sequence) + [0] * padding_length)
            self.labels.append(labels + [-100] * padding_length)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# 3. LLM-powered Customer Support Agent
class BehaviorCloningAgent:
    def __init__(self, model_name="gpt2", tokenizer=None, model=None):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = model
        self.model.eval()

    def get_command(self, crm_state, max_new_tokens=50):
        prompt = crm_state
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Generate command
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated part
        # We need to find where the generated part starts after the prompt
        input_length = inputs["input_ids"].shape[1]
        generated_sequence = outputs[0, input_length:].tolist()
        
        command = self.tokenizer.decode(generated_sequence, skip_special_tokens=True).strip()
        
        # Simple post-processing to extract a clean command if generation includes extra text
        if "(" in command and ")" in command:
            command_start = command.find("(")
            command_end = command.find(")")
            # Find the command name before the first parenthesis
            command_name_start = max(command.rfind(' ', 0, command_start), command.rfind('\n', 0, command_start), -1) + 1
            clean_command = command[command_name_start:command_end+1]
            return clean_command.strip()
        return command.strip()


# Main execution logic
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME = "gpt2"
    MAX_SEQUENCE_LENGTH = 128
    BATCH_SIZE = 2
    NUM_EPOCHS = 3

    # --- 1. Initialize Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # --- 2. Mock Data Collection: Human Demonstrations ---
    # Each demonstration is (CRM_state_observation, human_command)
    demonstrations = [
        ("Current state: Initial CRM dashboard. Customer inquiries are pending.", "search_customer(id=1)"),
        ("Current state: Customer found: ID 1, Name: Alice, Email: alice@example.com. Need order history.", "view_order_history(customer_id=1)"),
        ("Current state: Order history for Alice: A101, B202. Alice reports a payment issue for B202.", "create_ticket(customer_id=1, issue='payment_problem_B202')"),
        ("Current state: Initial CRM dashboard. A new customer inquiry arrived, ID 2.", "search_customer(id=2)"),
        ("Current state: Customer found: ID 2, Name: Bob, Email: bob@example.com. Bob needs help with a forgotten password.", "create_ticket(customer_id=2, issue='forgotten_password')"),
    ]

    # --- 3. Prepare Dataset for Behavior Cloning Fine-tuning ---
    train_dataset = CustomerSupportDataset(tokenizer, demonstrations, MAX_SEQUENCE_LENGTH)

    # --- 4. Behavior Cloning Fine-tuning Module ---
    print("\n--- Starting Behavior Cloning Fine-tuning ---")
    training_args = TrainingArguments(
        output_dir="./bc_model",
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=10_000, # Only save at the end for this example
        save_total_limit=1, # Keep only the last checkpoint
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=2e-5, # Standard fine-tuning learning rate
        report_to="none" # Disable reporting to external services like Weights & Biases
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    print("--- Behavior Cloning Fine-tuning Complete ---")

    # --- 5. Save the fine-tuned model (optional) ---
    # model.save_pretrained("./fine_tuned_cs_agent_model")
    # tokenizer.save_pretrained("./fine_tuned_cs_agent_model")

    # --- 6. LLM-powered Customer Support Agent (Inference) ---
    print("\n--- Initializing Fine-tuned Agent for Interaction ---")
    fine_tuned_agent = BehaviorCloningAgent(tokenizer=tokenizer, model=model)
    crm = CRMSimulator()

    print("\n--- Agent-CRM Interaction Simulation ---")
    current_crm_state = crm.current_state
    print(f"CRM Initial State: {current_crm_state}")

    # Example interaction 1: Search for customer and view orders
    print("\nAgent Task: Find customer ID 1 and view their orders.")
    for _ in range(2): # Two steps to search and then view orders
        generated_command = fine_tuned_agent.get_command(current_crm_state)
        print(f"Agent generates command: {generated_command}")
        if not generated_command: # Handle empty generation
            print("Agent generated an empty command. Ending interaction.")
            break
        new_crm_response = crm.execute_command(generated_command)
        print(f"CRM Response: {new_crm_response}")
        current_crm_state = new_crm_response # Update state for next turn

        if "Customer found" in new_crm_response and "search_customer" in generated_command:
            # If found, try to view order history next
            current_crm_state += " Need to view order history for this customer."

    current_crm_state = "Initial CRM dashboard. Awaiting command."
    crm.current_state = current_crm_state # Reset CRM state for new scenario
    print(f"\nCRM Reset State: {current_crm_state}")

    # Example interaction 2: Create a ticket for a new customer
    print("\nAgent Task: Find customer ID 3 (non-existent) and create a general inquiry ticket.")
    for _ in range(2): # Two steps, maybe search then create ticket
        generated_command = fine_tuned_agent.get_command(current_crm_state)
        print(f"Agent generates command: {generated_command}")
        if not generated_command: # Handle empty generation
            print("Agent generated an empty command. Ending interaction.")
            break
        new_crm_response = crm.execute_command(generated_command)
        print(f"CRM Response: {new_crm_response}")
        current_crm_state = new_crm_response # Update state for next turn
        
        if "Customer with ID 3 not found." in new_crm_response:
            current_crm_state += " Customer not found, create a general inquiry ticket for a new customer."

    print("\n--- Simulation Complete ---")