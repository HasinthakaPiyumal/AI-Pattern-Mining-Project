import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, DataCollatorWithPadding
from accelerate import Accelerator


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, tokenizer):
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.tokenized_trajectories = []
        for traj in self.trajectories:
            # Combine all parts of the trajectory into a single string for tokenization
            # This is a simplification; in reality, you might use special tokens
            # to delineate problem, rationale, program, output.
            full_text = " "
            for step in traj:
                full_text += f"Problem: {step.get('problem', '')}\nRationale: {step.get('rationale', '')}\nProgram: {step.get('program', '')}\nOutput: {step.get('output', '')}\n"
            
            tokenized_input = self.tokenizer(full_text, truncation=True, max_length=512, return_tensors="pt")
            self.tokenized_trajectories.append({
                "input_ids": tokenized_input["input_ids"].squeeze(),
                "attention_mask": tokenized_input["attention_mask"].squeeze(),
                "labels": tokenized_input["input_ids"].squeeze() # For causal LM, labels are typically input_ids
            })

    def __len__(self):
        return len(self.tokenized_trajectories)

    def __getitem__(self, idx):
        return self.tokenized_trajectories[idx]


class ToolInterface:
    def __init__(self):
        self.tools = {
            "CRM": self._mock_crm,
            "OrderManagement": self._mock_order_management,
            "KnowledgeBase": self._mock_knowledge_base,
            "BillingSystem": self._mock_billing_system,
        }

    def _mock_crm(self, customer_id=None, action=None, data=None):
        if action == "get_info":
            return f"CRM: Customer {customer_id} details: Name: John Doe, Email: john.doe@example.com"
        elif action == "update_status":
            return f"CRM: Customer {customer_id} status updated to {data}"
        return "CRM: Invalid action."

    def _mock_order_management(self, order_id=None, action=None):
        if action == "get_status":
            return f"OrderManagement: Order {order_id} status: Shipped, Tracking: TRK123"
        elif action == "cancel_order":
            return f"OrderManagement: Order {order_id} cancelled."
        return "OrderManagement: Invalid action."

    def _mock_knowledge_base(self, query=None):
        if "shipping" in query.lower():
            return "KnowledgeBase: Shipping policy: Orders typically arrive within 3-5 business days."
        return "KnowledgeBase: No relevant article found."

    def _mock_billing_system(self, customer_id=None, action=None):
        if action == "get_bill":
            return f"BillingSystem: Customer {customer_id} outstanding balance: $50.00"
        return "BillingSystem: Invalid action."

    def execute_tool_call(self, tool_name, *args, **kwargs):
        if tool_name in self.tools:
            try:
                return self.tools[tool_name](*args, **kwargs)
            except Exception as e:
                return f"Tool execution error for {tool_name}: {e}"
        return f"Error: Tool '{tool_name}' not found."


class ImitationLearningModule:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.accelerator = Accelerator()

    def train_model(self, num_epochs=3, batch_size=4, learning_rate=5e-5):
        train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=batch_size, collate_fn=self.data_collator
        )
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader
        )

        self.model.train()
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        print("Training complete.")


class InteractionEngine:
    def __init__(self, model, tokenizer, tool_interface, max_steps=10):
        self.model = model
        self.tokenizer = tokenizer
        self.tool_interface = tool_interface
        self.max_steps = max_steps
        self.model.eval()

    def _generate_response(self, prompt, max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True, # Enable sampling for more diverse responses
            top_k=50, # Sample from top 50 probable words
            top_p=0.95, # Sample from words that sum up to 95% probability
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the response
        return response[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

    def resolve_query(self, initial_query):
        interaction_history = [f"Customer Query: {initial_query}"]
        print(f"\n--- Resolving Query: {initial_query} ---")

        for step in range(self.max_steps):
            current_prompt = "\n".join(interaction_history) + "\nAgent:"
            print(f"\n[Step {step+1}] Agent's turn. Current context:\n{current_prompt}")
            
            agent_response = self._generate_response(current_prompt, max_new_tokens=100)
            interaction_history.append(f"Agent: {agent_response}")
            print(f"Agent says: {agent_response}")

            # Simple heuristic to detect tool calls (e.g., starts with 'CALL_TOOL')
            if agent_response.startswith("CALL_TOOL(") and ")" in agent_response:
                try:
                    tool_call_str = agent_response[len("CALL_TOOL("):-1]
                    parts = tool_call_str.split(",", 1)
                    tool_name = parts[0].strip()
                    tool_args_str = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Attempt to parse arguments as a dictionary (simple example)
                    tool_kwargs = {}
                    if tool_args_str:
                        for arg_pair in tool_args_str.split(","):
                            if "=" in arg_pair:
                                key, val = arg_pair.split("=", 1)
                                tool_kwargs[key.strip()] = val.strip().strip("'")

                    print(f"Executing tool: {tool_name} with args: {tool_kwargs}")
                    tool_output = self.tool_interface.execute_tool_call(tool_name, **tool_kwargs)
                    interaction_history.append(f"Tool Output: {tool_output}")
                    print(f"Tool Output: {tool_output}")
                except Exception as e:
                    error_message = f"Error parsing/executing tool call: {e}"
                    interaction_history.append(f"Tool Output: {error_message}")
                    print(f"Tool Output: {error_message}")
            elif "RESOLUTION:" in agent_response.upper() or "ESCALATE:" in agent_response.upper():
                print("Query resolved or escalated.")
                return "\n".join(interaction_history)
            
        print("Max steps reached. Query not fully resolved.")
        return "\n".join(interaction_history)


if __name__ == "__main__":
    # 1. Configuration
    MODEL_NAME = "distilgpt2"  # A small model for demonstration purposes
    BATCH_SIZE = 2
    NUM_EPOCHS = 1
    LEARNING_RATE = 5e-5

    # 2. Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # 3. Create Mock Trajectories (Simplified)
    # Each trajectory is a list of steps. Each step can have problem, rationale, program, output.
    mock_trajectories = [
        [
            {"problem": "My order #12345 is late.", "rationale": "I need to check the order status.", "program": "CALL_TOOL(OrderManagement, action='get_status', order_id='12345')"},
            {"output": "OrderManagement: Order 12345 status: Shipped, Tracking: TRK123", "rationale": "The order has shipped. I should inform the customer and provide tracking.", "program": "RESOLUTION: Your order #12345 has been shipped. Tracking number: TRK123."}
        ],
        [
            {"problem": "I want to know my account balance.", "rationale": "I need to access the billing system.", "program": "CALL_TOOL(BillingSystem, action='get_bill', customer_id='CUST789')"},
            {"output": "BillingSystem: Customer CUST789 outstanding balance: $50.00", "rationale": "I will provide the customer with their balance.", "program": "RESOLUTION: Your outstanding balance is $50.00."}
        ],
        [
            {"problem": "What is your shipping policy?", "rationale": "I need to look up information in the knowledge base.", "program": "CALL_TOOL(KnowledgeBase, query='shipping policy')"},
            {"output": "KnowledgeBase: Shipping policy: Orders typically arrive within 3-5 business days.", "rationale": "I will inform the customer about the shipping policy.", "program": "RESOLUTION: Our shipping policy states that orders typically arrive within 3-5 business days."}
        ]
    ]

    # 4. Dataset and DataLoader
    train_dataset = TrajectoryDataset(mock_trajectories, tokenizer)

    # 5. Imitation Learning
    imitation_learner = ImitationLearningModule(model, tokenizer, train_dataset)
    print("Starting imitation learning...")
    imitation_learner.train_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    print("Imitation learning finished.")

    # 6. Initialize Tool Interface
    tool_interface = ToolInterface()

    # 7. Initialize and Run Interaction Engine
    interaction_engine = InteractionEngine(model, tokenizer, tool_interface)

    test_queries = [
        "My order 12345 is still not here.",
        "How much do I owe? (customer CUST789)",
        "I need information about returns."
    ]

    for query in test_queries:
        resolved_interaction = interaction_engine.resolve_query(query)
        print(f"\nFinal Interaction Log:\n{resolved_interaction}")
        print("="*80)
