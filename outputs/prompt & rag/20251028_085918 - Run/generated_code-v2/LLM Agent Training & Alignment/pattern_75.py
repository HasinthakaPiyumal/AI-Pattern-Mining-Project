import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

class CLIDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

class Chatbot:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.model.to(self.device)

    def generate_command(self, user_query, max_new_tokens=50):
        prompt = user_query + " [CLI]:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=torch.ones(input_ids.shape, device=self.device)
        )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "[CLI]:" in generated_text:
            command_start_index = generated_text.find("[CLI]:") + len("[CLI]:")
            cli_command = generated_text[command_start_index:].strip()
            if self.tokenizer.eos_token in cli_command:
                cli_command = cli_command.split(self.tokenizer.eos_token)[0].strip()
            return cli_command.split("\n")[0].strip()

        return generated_text.strip()


def train_model(model, tokenizer, train_dataset, epochs=3, batch_size=2, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")
    return model

if __name__ == "__main__":
    MODEL_NAME = "gpt2"
    
    DEMONSTRATIONS = [
        {"user_query": "How do I list all active users?", "cli_command": "user list --active"},
        {"user_query": "Change the password for user 'admin'", "cli_command": "user password set --username admin --new-password new_secret_pw"},
        {"user_query": "Show me the network configuration", "cli_command": "network config show"},
        {"user_query": "Restart the 'webserver' service", "cli_command": "service restart --name webserver"},
        {"user_query": "What is the status of the 'database' service?", "cli_command": "service status --name database"},
        {"user_query": "Create a new user called 'developer'", "cli_command": "user add --username developer --role developer"},
        {"user_query": "Delete user 'testuser'", "cli_command": "user delete --username testuser"},
        {"user_query": "Display system logs from last hour", "cli_command": "log show --last-hour"},
        {"user_query": "Enable debugging for the 'auth' module", "cli_command": "module debug enable --name auth"},
        {"user_query": "Disable the firewall", "cli_command": "firewall disable"},
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    tokenized_inputs = {"input_ids": [], "attention_mask": []}
    for demo in DEMONSTRATIONS:
        full_text = f"{demo['user_query']} [CLI]: {demo['cli_command']}{tokenizer.eos_token}"
        encoding = tokenizer(full_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        tokenized_inputs["input_ids"].append(encoding["input_ids"].squeeze())
        tokenized_inputs["attention_mask"].append(encoding["attention_mask"].squeeze())

    train_dataset = CLIDataset(tokenized_inputs)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    print("Starting model training...")
    fine_tuned_model = train_model(model, tokenizer, train_dataset, epochs=5, batch_size=2)
    print("Model training finished.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chatbot = Chatbot(fine_tuned_model, tokenizer, device)

    print("\nChatbot ready! Type your queries or 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        generated_command = chatbot.generate_command(user_input)
        print(f"Chatbot (CLI Command): {generated_command}")