import random
import json
import re
import os

# --- 1. Data Generation/Simulation Module (`data_simulator.py`) ---
class DataSimulator:
    def __init__(self):
        self.customer_names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        self.issues = [
            "product not working",
            "billing error",
            "shipping delay",
            "account access issue",
            "technical support needed",
        ]
        self.statuses = ["open", "in progress", "resolved", "closed"]

    def generate_customer_query(self):
        query_types = [
            f"My {random.choice(self.issues)}. My name is {random.choice(self.customer_names)}.",
            f"Can you help me with a {random.choice(self.issues)}? I am {random.choice(self.customer_names)}.",
            f"I need to create a ticket about a {random.choice(self.issues)}. My name is {random.choice(self.customer_names)}.",
            f"What is the status of my ticket? I'm {random.choice(self.customer_names)}.",
            f"Update my ticket for {random.choice(self.customer_names)}. It's about a {random.choice(self.issues)}."
        ]
        return random.choice(query_types)

    def generate_crm_state(self, existing_tickets, existing_customers):
        state = {"tickets": [], "customers": []}
        if existing_tickets:
            # Add a subset of existing tickets to the state for context
            state["tickets"] = random.sample(existing_tickets, k=min(len(existing_tickets), 2))
        if existing_customers:
            # Add a subset of existing customers to the state
            state["customers"] = random.sample(existing_customers, k=min(len(existing_customers), 2))
        return json.dumps(state)

    def generate_agent_actions(self, customer_query, crm_state, existing_tickets, existing_customers):
        actions = []
        query_lower = customer_query.lower()

        customer_name_match = re.search(r"(?:name is|i am)\s+([a-zA-Z]+)", query_lower)
        customer_name = customer_name_match.group(1).capitalize() if customer_name_match else random.choice(self.customer_names)

        issue_match = re.search(r"(?:product not working|billing error|shipping delay|account access issue|technical support needed)", query_lower)
        issue = issue_match.group(0) if issue_match else random.choice(self.issues)

        if "create ticket" in query_lower or "need to create a ticket" in query_lower:
            ticket_id = len(existing_tickets) + 1
            actions.append(f"create_ticket(issue='{issue}', customer='{customer_name}')")
            existing_tickets.append({"id": ticket_id, "issue": issue, "customer": customer_name, "status": "open"})
        elif "status of my ticket" in query_lower:
            if existing_tickets:
                customer_tickets = [t for t in existing_tickets if t["customer"].lower() == customer_name.lower()]
                if customer_tickets:
                    ticket = random.choice(customer_tickets)
                    actions.append(f"search_ticket_status(id='{ticket["id"]}')")
                else:
                    actions.append(f"search_customer(name='{customer_name}')")
            else:
                actions.append(f"search_customer(name='{customer_name}')")
        elif "update my ticket" in query_lower:
            if existing_tickets:
                customer_tickets = [t for t in existing_tickets if t["customer"].lower() == customer_name.lower()]
                if customer_tickets:
                    ticket = random.choice(customer_tickets)
                    new_status = random.choice([s for s in self.statuses if s != ticket["status"]])
                    actions.append(f"update_ticket_status(id='{ticket["id"]}', status='{new_status}')")
                else:
                    actions.append(f"search_customer(name='{customer_name}')")
            else:
                actions.append(f"search_customer(name='{customer_name}')")
        else:
            if not actions:
                actions.append(f"search_customer(name='{customer_name}')")

        return "; ".join(actions)

    def generate_synthetic_data(self, num_samples):
        demonstrations = []
        existing_tickets = []
        existing_customers = [] # Simplified, just names
        for _ in range(num_samples):
            customer_query = self.generate_customer_query()
            crm_state_before = self.generate_crm_state(existing_tickets, existing_customers)
            agent_actions = self.generate_agent_actions(customer_query, crm_state_before, existing_tickets, existing_customers)
            # Simulate CRM state after actions for completeness
            crm_state_after = self.generate_crm_state(existing_tickets, existing_customers) # Re-generate with potentially updated tickets

            demonstrations.append({
                "customer_query": customer_query,
                "crm_state_before": crm_state_before,
                "agent_actions": agent_actions,
                "crm_state_after": crm_state_after,
            })

            # Ensure customers mentioned in queries/tickets are in existing_customers
            customer_name_match = re.search(r"(?:name is|i am)\s+([a-zA-Z]+)", customer_query.lower())
            if customer_name_match:
                name = customer_name_match.group(1).capitalize()
                if name not in existing_customers:
                    existing_customers.append(name)

        return demonstrations

# --- 2. CRM Simulator Module (`crm_simulator.py`) ---
class CRMSystem:
    def __init__(self):
        self.customers = {}
        self.tickets = {}
        self.next_ticket_id = 1

    def _parse_command(self, command_string):
        match = re.match(r"([a-zA-Z_]+)\((\w+=\'[^\']+\'(?:,\s*\w+=\'[^\']+\')*)*\)", command_string)
        if not match:
            return None, None
        command_name = match.group(1)
        args_str = match.group(2)
        args = {}
        if args_str:
            for arg_pair in args_str.split(","):
                key, value = arg_pair.split("=", 1)
                args[key.strip()] = value.strip().strip("'")
        return command_name, args

    def execute_command(self, command_string):
        command_name, args = self._parse_command(command_string)
        response = "Unknown command or invalid format."

        if command_name == "create_ticket":
            issue = args.get("issue")
            customer = args.get("customer")
            if issue and customer:
                ticket_id = self.next_ticket_id
                self.next_ticket_id += 1
                self.tickets[ticket_id] = {"id": ticket_id, "issue": issue, "customer": customer, "status": "open"}
                if customer not in self.customers:
                    self.customers[customer] = []
                self.customers[customer].append(ticket_id)
                response = f"Ticket {ticket_id} created for {customer} with issue: {issue}. Status: open."
            else:
                response = "Error: 'create_ticket' requires 'issue' and 'customer'."
        elif command_name == "update_ticket_status":
            ticket_id = int(args.get("id")) if args.get("id") else None
            status = args.get("status")
            if ticket_id in self.tickets and status:
                self.tickets[ticket_id]["status"] = status
                response = f"Ticket {ticket_id} status updated to {status}."
            else:
                response = "Error: 'update_ticket_status' requires valid 'id' and 'status'."
        elif command_name == "search_customer":
            customer_name = args.get("name")
            if customer_name in self.customers:
                tickets_for_customer = [self.tickets[tid] for tid in self.customers[customer_name]]
                response = f"Customer {customer_name} found. Tickets: {json.dumps(tickets_for_customer)}"
            else:
                response = f"Customer {customer_name} not found."
        elif command_name == "search_ticket_status":
            ticket_id = int(args.get("id")) if args.get("id") else None
            if ticket_id in self.tickets:
                response = f"Ticket {ticket_id} status: {self.tickets[ticket_id]["status"]}. Issue: {self.tickets[ticket_id]["issue"]}"
            else:
                response = f"Ticket {ticket_id} not found."
        
        return response

    def get_current_state(self):
        return json.dumps({"customers": self.customers, "tickets": self.tickets})

# --- 3. Model Fine-tuning Module (`model_finetuning.py`) ---
# Note: This section is a MOCK-UP. Actual fine-tuning with transformers library
# requires installing the library and running on a suitable environment (e.g., with GPU).
# For this demonstration, we simulate the process by saving/loading a dummy model file.

class ModelFinetuner:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = "finetuned_crm_assistant_model.json"

    def load_pretrained_model(self, model_name="mock_model"):
        print(f"Mocking loading pretrained model: {model_name}")
        # In a real scenario, you'd load a model and tokenizer from transformers
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = {"encode": lambda text: [ord(c) for c in text], "decode": lambda tokens: "".join([chr(t) for t in tokens])}
        self.model = {"name": model_name, "config": "mock_config"}
        print("Mock tokenizer and model loaded.")

    def prepare_dataset(self, demonstrations):
        print("Preparing dataset (mock tokenization)...")
        prepared_data = []
        for demo in demonstrations:
            input_text = f"Query: {demo['customer_query']} CRM State: {demo['crm_state_before']}"
            target_text = demo['agent_actions']
            # Mock tokenization - in reality, use self.tokenizer
            input_ids = self.tokenizer["encode"](input_text)
            labels = self.tokenizer["encode"](target_text)
            prepared_data.append({"input_ids": input_ids, "labels": labels})
        print(f"Prepared {len(prepared_data)} samples.")
        # In a real scenario, you'd create torch.utils.data.Dataset or datasets.Dataset
        return prepared_data, prepared_data # Dummy train and eval

    def fine_tune_model(self, train_dataset, eval_dataset):
        print("Mocking fine-tuning process...")
        # In a real scenario, this would involve Trainer from transformers
        # trainer = Trainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        # )
        # trainer.train()
        # trainer.save_model(self.model_path)

        # Simulate saving a fine-tuned model (just save a marker file)
        with open(self.model_path, "w") as f:
            json.dump({"mock_model_finetuned": True, "description": "This is a mock fine-tuned model file."},
f, indent=4)
        print(f"Mock fine-tuned model saved to {self.model_path}")

# --- 4. CRM Interaction Assistant Module (`crm_assistant.py`) ---
class CRMAssistant:
    def __init__(self, model_path="finetuned_crm_assistant_model.json"):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self._load_finetuned_model()

    def _load_finetuned_model(self):
        if os.path.exists(self.model_path):
            print(f"Mocking loading fine-tuned model from {self.model_path}")
            with open(self.model_path, "r") as f:
                self.model = json.load(f)
            # Mock tokenizer as well
            self.tokenizer = {"encode": lambda text: [ord(c) for c in text], "decode": lambda tokens: "".join([chr(t) for t in tokens])}
            print("Mock fine-tuned model and tokenizer loaded.")
        else:
            print(f"Warning: Fine-tuned model not found at {self.model_path}. Assistant will use simple rule-based predictions.")
            self.model = {"mock_model_finetuned": False}
            self.tokenizer = {"encode": lambda text: [ord(c) for c in text], "decode": lambda tokens: "".join([chr(t) for t in tokens])}

    def predict_commands(self, customer_query, crm_state):
        if not self.model or not self.model.get("mock_model_finetuned"): # If no real model loaded
            # Fallback to simple rule-based prediction for demonstration
            query_lower = customer_query.lower()
            customer_name_match = re.search(r"(?:name is|i am|for)\s+([a-zA-Z]+)", query_lower)
            customer_name = customer_name_match.group(1).capitalize() if customer_name_match else "UnknownCustomer"

            if "create ticket" in query_lower or "need to create a ticket" in query_lower:
                issue_match = re.search(r"(?:product not working|billing error|shipping delay|account access issue|technical support needed)", query_lower)
                issue = issue_match.group(0) if issue_match else "general issue"
                return [f"create_ticket(issue='{issue}', customer='{customer_name}')"]
            elif "status of my ticket" in query_lower or "check ticket" in query_lower:
                return [f"search_customer(name='{customer_name}')"]
            elif "update my ticket" in query_lower:
                status_match = re.search(r"to (open|in progress|resolved|closed)", query_lower)
                status = status_match.group(1) if status_match else "in progress"
                # This is a guess, a real model would need ticket ID
                return [f"search_customer(name='{customer_name}'); update_ticket_status(id='[TICKET_ID]', status='{status}')"] 
            else:
                return [f"search_customer(name='{customer_name}')"]
        
        # Mock model inference
        input_text = f"Query: {customer_query} CRM State: {crm_state}"
        # In a real scenario, perform model inference
        # input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        # outputs = self.model.generate(input_ids, max_length=100)
        # predicted_commands_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_commands_str = f"create_ticket(issue='mock issue', customer='MockCustomer')" if "create" in customer_query.lower() else f"search_customer(name='MockCustomer')"
        return [cmd.strip() for cmd in predicted_commands_str.split(';') if cmd.strip()]

    def main_interaction_loop(self):
        crm_system = CRMSystem()
        print("\n--- CRM Interaction Assistant Started ---")
        print("Type 'quit' to exit.")

        while True:
            customer_query = input("\nCustomer Query: ")
            if customer_query.lower() == 'quit':
                break

            current_crm_state = crm_system.get_current_state()
            print(f"Current CRM State: {current_crm_state}")

            predicted_commands = self.predict_commands(customer_query, current_crm_state)
            print(f"Assistant's Predicted Commands: {'; '.join(predicted_commands)}")

            for cmd in predicted_commands:
                if '[TICKET_ID]' in cmd: # Placeholder for rule-based prediction
                    # Attempt to find a relevant ticket if a customer search was done first
                    customer_name_match = re.search(r"name='([a-zA-Z]+)'", cmd)
                    customer_name = customer_name_match.group(1) if customer_name_match else None
                    if customer_name and customer_name in crm_system.customers and crm_system.customers[customer_name]:
                        # Use the first ticket for this customer as a mock
                        first_ticket_id = crm_system.customers[customer_name][0]
                        cmd = cmd.replace("[TICKET_ID]", str(first_ticket_id))
                    else:
                        print("Could not find a specific ticket ID to update. Command skipped.")
                        continue # Skip this command if ticket ID cannot be resolved
                
                crm_response = crm_system.execute_command(cmd)
                print(f"CRM Response to '{cmd}': {crm_response}")
            
            print(f"Updated CRM State: {crm_system.get_current_state()}")


if __name__ == "__main__":
    # 1. Generate Synthetic Data
    data_sim = DataSimulator()
    synthetic_demonstrations = data_sim.generate_synthetic_data(num_samples=20)
    with open("synthetic_demonstrations.json", "w") as f:
        json.dump(synthetic_demonstrations, f, indent=4)
    print("\nGenerated synthetic_demonstrations.json")

    # 2. Model Fine-tuning (Mock-up)
    finetuner = ModelFinetuner()
    finetuner.load_pretrained_model()
    train_data, eval_data = finetuner.prepare_dataset(synthetic_demonstrations)
    finetuner.fine_tune_model(train_data, eval_data)

    # 3. CRM Interaction Assistant (using the mock fine-tuned model or rule-based fallback)
    assistant = CRMAssistant()
    assistant.main_interaction_loop()