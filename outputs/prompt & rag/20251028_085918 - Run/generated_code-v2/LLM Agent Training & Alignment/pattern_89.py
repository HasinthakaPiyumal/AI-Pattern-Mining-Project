
# --- File: config.py ---
MODEL_NAME = "bert-base-uncased"
DEMONSTRATION_DATA_PATH = "data/demonstrations.json"
FINE_TUNED_MODEL_PATH = "models/behavior_cloning_model"

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3

MAX_SEQUENCE_LENGTH = 128

ENVIRONMENT_API_URL = "http://mock-enterprise-software.com/api/v1/"

# --- File: data_collector_mock.py ---
import json
import os

def generate_mock_demonstrations(num_demonstrations=100):
    demonstrations = []
    scenarios = [
        ("Customer wants to reset their password.", "navigate_to_settings -> click_password_reset -> confirm_reset"),
        ("Customer is unable to log in, forgotten username.", "search_user_database -> retrieve_username -> send_username_email"),
        ("Customer asks to update their billing information.", "navigate_to_billing -> edit_payment_method -> save_changes"),
        ("Customer wants to check their recent order status.", "access_order_history -> search_order_id -> display_order_status"),
        ("Customer reports a bug in the application.", "open_bug_report_form -> fill_bug_details -> submit_bug_report"),
        ("Customer asks for a refund for a recent purchase.", "check_refund_policy -> initiate_refund_process -> confirm_refund_amount"),
        ("Customer needs help with product configuration.", "open_product_manual -> search_config_section -> provide_step_by_step_guide"),
    ]

    for i in range(num_demonstrations):
        observation, action_sequence = scenarios[i % len(scenarios)]
        demonstrations.append({"observation": observation, "action": action_sequence})

    output_dir = os.path.dirname("data/demonstrations.json")
    os.makedirs(output_dir, exist_ok=True)
    with open("data/demonstrations.json", "w") as f:
        json.dump(demonstrations, f, indent=4)
    print(f"Generated {num_demonstrations} mock demonstrations at data/demonstrations.json")

if __name__ == "__main__":
    generate_mock_demonstrations()

# --- File: train_behavior_cloning_model.py ---
import json
import os

import config

class CustomerSupportDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train_behavior_cloning_model():
    print("--- Starting Conceptual Model Training ---")
    if not os.path.exists(config.DEMONSTRATION_DATA_PATH):
        print(f"Mock demonstration data not found at {config.DEMONSTRATION_DATA_PATH}. Please run data_collector_mock.py first.")
        return

    with open(config.DEMONSTRATION_DATA_PATH, "r") as f:
        demonstrations = json.load(f)

    observations = [d["observation"] for d in demonstrations]
    actions = [d["action"] for d in demonstrations]

    print(f"Loaded {len(demonstrations)} demonstrations.")

    print(f"Conceptual: Initializing tokenizer and model from {config.MODEL_NAME}")

    print("Conceptual: Data tokenized and dataset prepared.")

    print("Conceptual: Training arguments defined.")

    print(f"Conceptual: Model would be saved to {config.FINE_TUNED_MODEL_PATH}")
    os.makedirs(config.FINE_TUNED_MODEL_PATH, exist_ok=True)
    with open(os.path.join(config.FINE_TUNED_MODEL_PATH, "mock_model.pt"), "w") as f:
        f.write("mock model weights")
    with open(os.path.join(config.FINE_TUNED_MODEL_PATH, "mock_tokenizer.json"), "w") as f:
        f.write("mock tokenizer config")
    print("--- Conceptual Model Training Completed ---")

if __name__ == "__main__":
    train_behavior_cloning_model()

# --- File: chatbot_inference_agent.py ---
import os

import config

class ChatbotAgent:
    def __init__(self, model_path=config.FINE_TUNED_MODEL_PATH):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Conceptual: Attempting to load model from {self.model_path}")
        if not os.path.exists(self.model_path) or \
           not os.path.exists(os.path.join(self.model_path, "mock_model.pt")):
            print(f"Conceptual: Fine-tuned model not found at {self.model_path} or incomplete. Please run train_behavior_cloning_model.py first.")
            return
        
        print(f"Conceptual: Successfully simulated loading tokenizer and model from {self.model_path}")
        self.tokenizer = "<mock_tokenizer_loaded>"
        self.model = "<mock_model_loaded>"

    def get_action(self, observation: str) -> str:
        if self.model is None or self.tokenizer is None:
            return "Error: Chatbot model not loaded. Cannot generate action."

        print(f"Conceptual: Processing observation: '{observation}'")
        
        observation_lower = observation.lower()
        if "password" in observation_lower:
            generated_action = "navigate_to_settings -> click_password_reset -> confirm_reset"
        elif "login" in observation_lower or "username" in observation_lower:
            generated_action = "search_user_database -> retrieve_username -> send_username_email"
        elif "billing" in observation_lower or "payment" in observation_lower or "credit card" in observation_lower:
            generated_action = "navigate_to_billing -> edit_payment_method -> save_changes"
        elif "order status" in observation_lower:
            generated_action = "access_order_history -> search_order_id -> display_order_status"
        elif "bug" in observation_lower or "error" in observation_lower:
            generated_action = "open_bug_report_form -> fill_bug_details -> submit_bug_report"
        elif "refund" in observation_lower:
            generated_action = "check_refund_policy -> initiate_refund_process -> confirm_refund_amount"
        elif "product configuration" in observation_lower or "setup help" in observation_lower:
            generated_action = "open_product_manual -> search_config_section -> provide_step_by_step_guide"
        else:
            generated_action = "unrecognized_query -> ask_for_clarification"

        print(f"Conceptual: Generated action: '{generated_action}'")
        return generated_action

    def _execute_action(self, action_sequence: str) -> str:
        print(f"Conceptual: Executing action sequence: '{action_sequence}'")
        steps = [step.strip() for step in action_sequence.split('->')]
        results = []
        for step in steps:
            if "navigate_to_" in step:
                results.append(f"Navigated to {step.replace('navigate_to_','').replace('_',' ')}")
            elif "click_" in step:
                results.append(f"Clicked {step.replace('click_','').replace('_',' ')}")
            elif "search_" in step:
                results.append(f"Searched for {step.replace('search_','').replace('_',' ')}")
            elif "confirm_" in step:
                results.append(f"Confirmed action: {step.replace('confirm_','').replace('_',' ')}")
            elif "send_username_email" == step:
                results.append("Sent username email.")
            elif "edit_payment_method" == step:
                results.append("Edited payment method.")
            elif "save_changes" == step:
                results.append("Saved changes.")
            elif "display_order_status" == step:
                results.append("Displayed order status.")
            elif "fill_bug_details" == step:
                results.append("Filled bug details.")
            elif "submit_bug_report" == step:
                results.append("Submitted bug report.")
            elif "initiate_refund_process" == step:
                results.append("Initiated refund process.")
            elif "provide_step_by_step_guide" == step:
                results.append("Provided step-by-step guide.")
            elif "unrecognized_query" == step:
                results.append("Query unrecognized.")
            elif "ask_for_clarification" == step:
                results.append("Asked for clarification.")
            else:
                results.append(f"Performed generic action: {step}")
        
        final_result = "; ".join(results)
        print(f"Conceptual: Action execution result: {final_result}")
        return f"Successfully performed: {action_sequence}. Result: {final_result}"

    def handle_customer_query(self, query: str) -> str:
        print(f"\nCustomer Query: {query}")
        action_sequence = self.get_action(query)
        if action_sequence.startswith("Error"):
            return action_sequence
        
        response = self._execute_action(action_sequence)
        return f"Chatbot Response: {response}"

if __name__ == "__main__":
    
    chatbot = ChatbotAgent()

    if chatbot.model is None:
        print("Cannot run inference without a loaded model. Please check the training script and model path.")
    else:
        print("\n--- Simulating Customer Interactions ---")
        print(chatbot.handle_customer_query("I forgot my password."))
        print(chatbot.handle_customer_query("My login is not working, I don't remember my username."))
        print(chatbot.handle_customer_query("I need to change my credit card details."))
        print(chatbot.handle_customer_query("What's the status of my order 12345?"))
        print(chatbot.handle_customer_query("I found a bug in the application."))
        print(chatbot.handle_customer_query("I want a refund for my recent purchase."))
        print(chatbot.handle_customer_query("How do I set up feature X?"))
        print(chatbot.handle_customer_query("Can you help me with something else?"))
