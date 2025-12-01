import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import re

# 1. Tool Simulation Module
class ECommerceTools:
    def get_order_status(self, order_id):
        if order_id == "ORD123":
            return "Tool Output: Order ORD123 status is 'Shipped'."
        elif order_id == "ORD456":
            return "Tool Output: Order ORD456 status is 'Pending Shipment'."
        else:
            return "Tool Output: Order ID not found."

    def track_shipping(self, order_id):
        if order_id == "ORD123":
            return "Tool Output: Tracking for ORD123: Carrier 'FedEx', Tracking ID 'FEDEX987654', Estimated Delivery '2023-11-25'."
        elif order_id == "ORD456":
            return "Tool Output: Tracking for ORD456: Not yet shipped."
        else:
            return "Tool Output: Order ID not found for tracking."

    def initiate_return(self, order_id, reason):
        if order_id == "ORD123":
            return f"Tool Output: Return initiated for ORD123 with reason: '{reason}'. A return label will be sent."
        else:
            return "Tool Output: Return initiation failed. Order ID not eligible or found."

    def execute_tool(self, tool_call_str):
        match = re.match(r"(\w+)\((.*)\)", tool_call_str)
        if not match:
            return "Tool Output: Invalid tool call format."

        tool_name = match.group(1)
        args_str = match.group(2)

        args = {}
        # Basic parsing for key=value pairs, handles simple cases
        for arg_pair in args_str.split(', '):
            if '=' in arg_pair:
                key, value = arg_pair.split('=', 1)
                args[key.strip()] = value.strip().strip("'").strip('"')

        if hasattr(self, tool_name) and callable(getattr(self, tool_name)):
            try:
                # Filter args to only those expected by the method
                import inspect
                sig = inspect.signature(getattr(self, tool_name))
                filtered_args = {k: v for k, v in args.items() if k in sig.parameters}
                return getattr(self, tool_name)(**filtered_args)
            except TypeError as e:
                return f"Tool Output: Error executing tool '{tool_name}': Invalid arguments. {e}"
        else:
            return f"Tool Output: Unknown tool: {tool_name}"

# 2. Data Curation Module (Simulated)
# Each trajectory step includes a 'type' (rationale, tool_call, tool_output) and 'content'
# For training, we convert this into a sequence of tokens.

trajectories = [
    {
        "problem": "What is the status of my order ORD123?",
        "steps": [
            {"type": "rationale", "content": "Okay, I need to check the order status. I will use the 'get_order_status' tool."},
            {"type": "tool_call", "content": "get_order_status(order_id='ORD123')"},
            {"type": "tool_output", "content": "Tool Output: Order ORD123 status is 'Shipped'."},
            {"type": "rationale", "content": "Your order ORD123 has been shipped."}
        ]
    },
    {
        "problem": "I want to return my order ORD123, the item is damaged.",
        "steps": [
            {"type": "rationale", "content": "Understood. I need to initiate a return. I will use the 'initiate_return' tool."},
            {"type": "tool_call", "content": "initiate_return(order_id='ORD123', reason='Item damaged')"},
            {"type": "tool_output", "content": "Tool Output: Return initiated for ORD123 with reason: 'Item damaged'. A return label will be sent."},
            {"type": "rationale", "content": "A return for your order ORD123 due to damage has been initiated. A return label will be sent to your email."}
        ]
    },
    {
        "problem": "Can you tell me where is my order ORD123?",
        "steps": [
            {"type": "rationale", "content": "Certainly, I will check the shipping details for your order. I will use the 'track_shipping' tool."},
            {"type": "tool_call", "content": "track_shipping(order_id='ORD123')"},
            {"type": "tool_output", "content": "Tool Output: Tracking for ORD123: Carrier 'FedEx', Tracking ID 'FEDEX987654', Estimated Delivery '2023-11-25'."},
            {"type": "rationale", "content": "Your order ORD123 is currently with FedEx, tracking ID FEDEX987654, and is estimated to arrive by November 25, 2023."}
        ]
    },
]

# Special tokens for the agent
PROBLEM_TOKEN = "<PROBLEM>"
RATIONALE_TOKEN = "<RATIONALE>"
TOOL_CALL_TOKEN = "<TOOL_CALL>"
TOOL_OUTPUT_TOKEN = "<TOOL_OUTPUT>"
RESPONSE_TOKEN = "<RESPONSE>"
END_TOKEN = "<END>"

class ToolUseDataset(Dataset):
    def __init__(self, trajectories, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        special_tokens_dict = {"additional_special_tokens": [PROBLEM_TOKEN, RATIONALE_TOKEN, TOOL_CALL_TOKEN, TOOL_OUTPUT_TOKEN, RESPONSE_TOKEN, END_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)

        for traj in trajectories:
            problem_text = f"{PROBLEM_TOKEN}{traj['problem']}"
            current_history = problem_text

            for i in range(len(traj['steps'])):
                step = traj['steps'][i]

                input_sequence = current_history
                target_sequence = ""

                if step['type'] == 'rationale':
                    target_sequence = f"{RATIONALE_TOKEN}{step['content']}"
                elif step['type'] == 'tool_call':
                    target_sequence = f"{TOOL_CALL_TOKEN}{step['content']}"
                elif step['type'] == 'tool_output':
                    target_sequence = f"{TOOL_OUTPUT_TOKEN}{step['content']}"
                
                if i == len(traj['steps']) - 1: # Last step in trajectory is a final response or end
                    target_sequence += f"{END_TOKEN}"

                # Prepare input_ids and labels
                tokenized_input = tokenizer(input_sequence, truncation=True, max_length=self.max_length, return_tensors='pt')
                tokenized_target = tokenizer(target_sequence, truncation=True, max_length=self.max_length, return_tensors='pt')

                input_ids = tokenized_input['input_ids'].squeeze(0)
                attention_mask = tokenized_input['attention_mask'].squeeze(0)
                labels = tokenized_target['input_ids'].squeeze(0)

                # Pad/truncate to max_length
                input_ids = torch.cat([input_ids, torch.full((self.max_length - len(input_ids),), tokenizer.pad_token_id)])[:self.max_length]
                attention_mask = torch.cat([attention_mask, torch.zeros(self.max_length - len(attention_mask))])[:self.max_length]
                labels = torch.cat([labels, torch.full((self.max_length - len(labels),), -100)])[:self.max_length]

                self.input_ids.append(input_ids)
                self.attention_masks.append(attention_mask)
                self.labels.append(labels)

                # Update current_history for the next step's input
                current_history += target_sequence

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

# 3. LLM Agent Model & 4. Imitation Learning Trainer
def train_agent(model, tokenizer, train_dataloader, epochs=3, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")
    return model

# 5. Interactive Inference Module
class CustomerSupportAgent:
    def __init__(self, model, tokenizer, tools, max_turns=5, max_new_tokens=100):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def converse(self, query):
        history = f"{PROBLEM_TOKEN}{query}"
        print(f"Customer: {query}")
        
        for turn in range(self.max_turns):
            input_ids = self.tokenizer.encode(history, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=True, # Allow for more varied responses
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id, # Use EOS for padding in generation
                    early_stopping=True # Stop if EOS is generated
                )
            
            new_tokens = output_ids[0, len(input_ids[0]):].tolist()
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            
            # Extract the actual generated segment based on special tokens
            segment = generated_text.split(END_TOKEN)[0].strip()

            if RATIONALE_TOKEN in segment:
                rationale = segment.replace(RATIONALE_TOKEN, '').strip()
                print(f"Agent (Rationale): {rationale}")
                history += f"{RATIONALE_TOKEN}{rationale}"
            elif TOOL_CALL_TOKEN in segment:
                tool_call_str = segment.replace(TOOL_CALL_TOKEN, '').strip()
                print(f"Agent (Tool Call): {tool_call_str}")
                history += f"{TOOL_CALL_TOKEN}{tool_call_str}"
                tool_output = self.tools.execute_tool(tool_call_str)
                print(f"Tool Output: {tool_output}")
                history += f"{TOOL_OUTPUT_TOKEN}{tool_output}"
            elif segment.startswith("Your order") or segment.startswith("A return") or segment.startswith("Your order ORD123 is currently with") or segment.startswith("Order ID not found") or segment.startswith("Return initiation failed") or segment.startswith("Tracking for ORD456: Not yet shipped.") or segment.startswith("Order ORD123 has been shipped.") or segment.startswith("A return for your order ORD123 due to damage has been initiated. A return label will be sent to your email.") or segment.startswith("Your order ORD123 is currently with FedEx, tracking ID FEDEX987654, and is estimated to arrive by November 25, 2023."):
                response = segment.strip()
                print(f"Agent (Response): {response}")
                history += f"{RESPONSE_TOKEN}{response}"
                print("Conversation Ended.")
                return response
            elif END_TOKEN in segment:
                print("Agent (Partial Response/End): No specific action/final response generated yet, ending conversation.")
                print("Conversation Ended.")
                return "I could not fully resolve your request at this moment. Please try rephrasing."
            else:
                print(f"Agent (Unknown Segment): {segment}")
                print("Conversation Ended due to unexpected output.")
                return "I encountered an unexpected response. Please try again."
            
            if END_TOKEN in generated_text: # Explicitly check for END_TOKEN for early stopping in inference
                print("Conversation Ended by END_TOKEN.")
                return history.split(RESPONSE_TOKEN)[-1].strip() if RESPONSE_TOKEN in history else "I have processed your request to the best of my ability."

        print("Conversation Ended: Max turns reached.")
        return "I could not fully resolve your request within the given turns. Please provide more details or try again later."

# Main execution block
if __name__ == "__main__":
    # Initialize tokenizer and model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add special tokens to tokenizer and resize model embeddings
    special_tokens_dict = {"additional_special_tokens": [PROBLEM_TOKEN, RATIONALE_TOKEN, TOOL_CALL_TOKEN, TOOL_OUTPUT_TOKEN, RESPONSE_TOKEN, END_TOKEN]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    # Set padding token if not already set (important for generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # Prepare dataset and dataloader
    train_dataset = ToolUseDataset(trajectories, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Train the agent
    print("\n--- Training Agent ---")
    trained_model = train_agent(model, tokenizer, train_dataloader, epochs=5)
    print("--- Training Complete ---")

    # Initialize tools and inference agent
    ecommerce_tools = ECommerceTools()
    agent = CustomerSupportAgent(trained_model, tokenizer, ecommerce_tools)

    # Conduct conversations
    print("\n--- Starting Conversations ---")
    agent.converse("What is the status of my order ORD123?")
    print("\n-----------------------------------")
    agent.converse("I want to return my order ORD123, it's too small.")
    print("\n-----------------------------------")
    agent.converse("Where is my order ORD123?")
    print("\n-----------------------------------")
    agent.converse("What about order ORD456?")
    print("\n-----------------------------------")
    agent.converse("I need help with something else.")

