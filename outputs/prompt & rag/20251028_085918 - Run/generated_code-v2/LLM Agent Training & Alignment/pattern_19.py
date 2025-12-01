import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random

class CRMSimulator:
    def __init__(self):
        self.state = {"customer_id": None, "ticket_status": "open", "issue_description": "None"}
        self.valid_commands = [
            "create_ticket",
            "update_ticket_status_closed",
            "assign_ticket_to_agent",
            "view_customer_history",
            "escalate_issue",
            "send_confirmation_email",
            "do_nothing",
        ]

    def get_state(self):
        return f"Current CRM State: Customer ID: {self.state['customer_id']}, Ticket Status: {self.state['ticket_status']}, Issue: {self.state['issue_description']}"

    def execute_command(self, command):
        if command not in self.valid_commands:
            return f"Invalid command: {command}. No state change."

        if command == "create_ticket":
            self.state["customer_id"] = random.randint(1000, 9999)
            self.state["ticket_status"] = "open"
            self.state["issue_description"] = "New issue reported"
            return f"Ticket {self.state['customer_id']} created and assigned as open."
        elif command == "update_ticket_status_closed":
            if self.state['customer_id'] is None: return "Cannot close ticket: no customer ID."
            self.state["ticket_status"] = "closed"
            return f"Ticket {self.state['customer_id']} status updated to closed."
        elif command == "assign_ticket_to_agent":
            if self.state['customer_id'] is None: return "Cannot assign ticket: no customer ID."
            return f"Ticket {self.state['customer_id']} assigned to agent."
        elif command == "view_customer_history":
            if self.state['customer_id'] is None: return "Cannot view history: no customer ID."
            return f"Displaying history for customer {self.state['customer_id']}."
        elif command == "escalate_issue":
            if self.state['customer_id'] is None: return "Cannot escalate issue: no customer ID."
            self.state["ticket_status"] = "escalated"
            return f"Issue for ticket {self.state['customer_id']} escalated."
        elif command == "send_confirmation_email":
            if self.state['customer_id'] is None: return "Cannot send email: no customer ID."
            return f"Confirmation email sent for ticket {self.state['customer_id']}."
        elif command == "do_nothing":
            return "Agent chose to do nothing."
        return "Command executed."

    def reset(self):
        self.state = {"customer_id": None, "ticket_status": "open", "issue_description": "None"}

class CRMDataset(Dataset):
    def __init__(self, demonstrations, tokenizer, command_to_idx, max_len=128):
        self.demonstrations = demonstrations
        self.tokenizer = tokenizer
        self.command_to_idx = command_to_idx
        self.max_len = max_len

        self.inputs = []
        self.labels = []
        self._prepare_data()

    def _prepare_data(self):
        for crm_state, customer_query, human_action in self.demonstrations:
            combined_input = f"CRM State: {crm_state} Customer Query: {customer_query}"
            tokenized_input = self.tokenizer(
                combined_input,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(self.command_to_idx[human_action]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"].squeeze(),
            "labels": self.labels[idx]
        }

class BehaviorCloningAgent(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_commands, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_commands)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        logits = self.fc(hidden[-1, :, :])
        return logits

def train_agent(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

def run_inference(agent_model, tokenizer, crm_simulator, idx_to_command, device, max_len=128):
    agent_model.eval()
    agent_model.to(device)
    print("\n--- Starting Agent Inference ---")
    crm_simulator.reset()

    for _ in range(5): # Simulate 5 turns of interaction
        current_crm_state_text = crm_simulator.get_state()
        print(f"\nCRM State: {current_crm_state_text}")

        customer_query = input("Enter customer query (e.g., 'My internet is not working.'): ")
        if not customer_query:
            print("Exiting inference.")
            break

        combined_input = f"CRM State: {current_crm_state_text} Customer Query: {customer_query}"
        tokenized_input = tokenizer(
            combined_input,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        input_ids = tokenized_input["input_ids"].to(device)

        with torch.no_grad():
            logits = agent_model(input_ids)
            predicted_idx = torch.argmax(logits, dim=-1).item()
            predicted_command = idx_to_command[predicted_idx]

        print(f"Agent predicts command: '{predicted_command}'")
        command_result = crm_simulator.execute_command(predicted_command)
        print(f"CRM System Response: {command_result}")

        if "closed" in crm_simulator.state["ticket_status"] or "escalated" in crm_simulator.state["ticket_status"]:
            print("Ticket resolved, closed, or escalated. Ending interaction.")
            break

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crm_simulator = CRMSimulator()
    command_to_idx = {cmd: i for i, cmd in enumerate(crm_simulator.valid_commands)}
    idx_to_command = {i: cmd for i, cmd in enumerate(crm_simulator.valid_commands)}
    num_commands = len(crm_simulator.valid_commands)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_seq_len = 128

    demonstrations = [
        ("Customer ID: None, Ticket Status: open, Issue: None", "My internet is not working.", "create_ticket"),
        ("Customer ID: 1234, Ticket Status: open, Issue: New issue reported", "I want to know the status of my ticket.", "view_customer_history"),
        ("Customer ID: 1234, Ticket Status: open, Issue: New issue reported", "The issue is resolved now.", "update_ticket_status_closed"),
        ("Customer ID: 5678, Ticket Status: open, Issue: New issue reported", "This is urgent, I need help immediately!", "escalate_issue"),
        ("Customer ID: 1234, Ticket Status: closed, Issue: New issue reported", "Did you send the confirmation?", "send_confirmation_email"),
        ("Customer ID: None, Ticket Status: open, Issue: None", "I just called about a new problem.", "create_ticket"),
        ("Customer ID: 9876, Ticket Status: open, Issue: New issue reported", "Can you assign this to agent John?", "assign_ticket_to_agent"),
        ("Customer ID: 1122, Ticket Status: open, Issue: New issue reported", "Everything is fine, no further action.", "do_nothing"),
    ]

    dataset = CRMDataset(demonstrations, tokenizer, command_to_idx, max_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    vocab_size = len(tokenizer)
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    model = BehaviorCloningAgent(vocab_size, embedding_dim, hidden_dim, num_commands, num_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 20

    print("--- Starting Training ---")
    train_agent(model, dataloader, optimizer, criterion, device, num_epochs)
    print("--- Training Complete ---")

    run_inference(model, tokenizer, crm_simulator, idx_to_command, device, max_len=max_seq_len)
