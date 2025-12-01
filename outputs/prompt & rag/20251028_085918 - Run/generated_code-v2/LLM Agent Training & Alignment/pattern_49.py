from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import random
import time
import gradio as gr


# --- 1. Data Models ---
class Action(BaseModel):
    type: str  # e.g., "search_product", "get_order_status", "initiate_return"
    args: Dict[str, Any]

class Observation(BaseModel):
    state: Dict[str, Any]  # e.g., {"product_found": true, "product_details": {...}}
    text: Optional[str] = None # Natural language observation

class Demonstration(BaseModel):
    customer_query: str
    actions: List[Action]
    observations: List[Observation]
    final_response: str

class Comparison(BaseModel):
    customer_query: str
    response_a: str
    response_b: str
    preferred_response: str # "A" or "B"


# --- 2. Data Collection Module ---
# Simplified Simulation for Demonstration Recorder
def simulate_demonstration_recording(num_demonstrations: int = 2) -> List[Demonstration]:
    demonstrations = []
    for i in range(num_demonstrations):
        query = f"Customer query {i+1}: I need help with my order #{1000 + i}."
        actions = [
            Action(type="get_order_status", args={"order_id": 1000 + i}),
            Action(type="search_product", args={"query": "shipping policy"}),
        ]
        observations = [
            Observation(state={"order_status": "shipped", "tracking_number": "TRK12345"}, text="Order 1001 is shipped."),
            Observation(state={"policy_found": True, "policy_details": "..."}, text="Shipping policy retrieved."),
        ]
        final_response = f"Your order #{1000 + i} has been shipped with tracking number TRK12345."
        demonstrations.append(Demonstration(customer_query=query, actions=actions, observations=observations, final_response=final_response))
    print(f"Simulated {num_demonstrations} demonstrations.")
    return demonstrations

# Gradio App for Comparison Collector (conceptual)
def save_comparison_feedback(query, response_a, response_b, preference):
    if not preference:
        return "Please select a preference!"
    comparison = Comparison(customer_query=query, response_a=response_a, response_b=response_b, preferred_response=preference)
    print(f"Saved comparison: {comparison}")
    return f"Feedback recorded for query: '{query}' - Preferred: {preference}"

def create_comparison_collector_app():
    with gr.Blocks() as app:
        gr.Markdown("# Customer Support Agent Comparison Collector")
        gr.Markdown("Evaluate pairs of responses to customer queries.")

        query_input = gr.Textbox(label="Customer Query", placeholder="e.g., Where is my order?", lines=2)
        response_a_input = gr.Textbox(label="Response A (Model Generated)", placeholder="e.g., Your order is on its way...", lines=3)
        response_b_input = gr.Textbox(label="Response B (Model Generated/Human)", placeholder="e.g., We have dispatched your item...", lines=3)
        preference_radio = gr.Radio(["A", "B"], label="Which response is better?")
        submit_btn = gr.Button("Submit Feedback")
        output_message = gr.Textbox(label="Status")

        submit_btn.click(
            save_comparison_feedback,
            inputs=[query_input, response_a_input, response_b_input, preference_radio],
            outputs=output_message
        )
    return app


# --- 3. Training Module (Conceptual) ---
class BehaviorCloningTrainer:
    def __init__(self, model_name: str = "t5-small"):
        print(f"Initializing BC Trainer with {model_name}...")
        # In a real scenario, load a tokenizer and model from transformers
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model_name = model_name

    def prepare_data(self, demonstrations: List[Demonstration]):
        print(f"Preparing {len(demonstrations)} demonstrations for BC training.")
        # This is where you'd tokenize inputs and targets
        # Example: input = query + actions + observations, target = final_response
        # Using datasets library: tokenized_datasets = Dataset.from_list(demonstrations).map(...)
        print("Data preparation complete.")
        return {"tokenized_data": "simulated_tokenized_data"}

    def train(self, prepared_data: Any, num_epochs: int = 3):
        print(f"Starting Behavior Cloning training for {num_epochs} epochs.")
        # Here, you would use Trainer from transformers or a custom training loop
        # trainer = Trainer(model=self.model, args=training_args, train_dataset=prepared_data['tokenized_data'])
        # trainer.train()
        time.sleep(1) # Simulate training time
        print("Behavior Cloning training finished.")
        return {"bc_model": "trained_bc_model_weights"}

class RewardModelTrainer:
    def __init__(self, base_model_name: str = "bert-base-uncased"):
        print(f"Initializing Reward Model Trainer with {base_model_name}...")
        # In a real scenario, load a tokenizer and model from transformers
        # self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
        self.base_model_name = base_model_name

    def prepare_data(self, comparisons: List[Comparison]):
        print(f"Preparing {len(comparisons)} comparisons for RM training.")
        # This is where you'd create pairs of (query, response) and labels (preferred/not preferred)
        # Using trl.RewardTrainer expects a specific dataset format
        print("Data preparation complete.")
        return {"tokenized_data_rm": "simulated_tokenized_rm_data"}

    def train(self, prepared_data: Any, num_epochs: int = 3):
        print(f"Starting Reward Model training for {num_epochs} epochs.")
        # Here, you would use trl.RewardTrainer
        # trainer = RewardTrainer(model=self.model, tokenizer=self.tokenizer, args=training_args, train_dataset=prepared_data['tokenized_data_rm'])
        # trainer.train()
        time.sleep(1) # Simulate training time
        print("Reward Model training finished.")
        return {"rm_model": "trained_rm_model_weights"}


# --- 4. Agent Orchestration Module (Conceptual) ---
class EcommerceTools:
    def get_order_status(self, order_id: int) -> Dict[str, Any]:
        print(f"Tool: Getting status for order {order_id}")
        if order_id % 2 == 0:
            return {"order_id": order_id, "status": "shipped", "tracking": f"TRK{order_id}"}
        else:
            return {"order_id": order_id, "status": "processing"}

    def search_product(self, query: str) -> Dict[str, Any]:
        print(f"Tool: Searching for product '{query}'")
        if "laptop" in query.lower():
            return {"products": [{"id": 1, "name": "Gaming Laptop", "price": 1200}]}
        return {"products": []}

    def initiate_return(self, order_id: int, item_id: int) -> Dict[str, Any]:
        print(f"Tool: Initiating return for order {order_id}, item {item_id}")
        return {"success": True, "return_id": random.randint(10000, 99999)}

class CustomerSupportAgent:
    def __init__(self, bc_model_weights: Any, rm_model_weights: Any):
        print("Initializing Customer Support Agent...")
        # In a real scenario, load actual models
        self.bc_model = bc_model_weights # Placeholder for the trained BC model
        self.rm_model = rm_model_weights # Placeholder for the trained Reward Model
        self.tools = EcommerceTools()
        print("Agent ready.")

    def _call_tool(self, action: Action) -> Observation:
        tool_func = getattr(self.tools, action.type, None)
        if tool_func:
            result = tool_func(**action.args)
            return Observation(state=result, text=f"Tool '{action.type}' executed. Result: {result}")
        return Observation(state={"error": f"Unknown tool: {action.type}"}, text="Failed to execute tool.")

    def process_query(self, query: str) -> str:
        print(f"Agent processing query: '{query}'")

        # Step 1: LLM (BC model) decides on actions based on query
        # In a real scenario, this would involve prompting the LLM
        # and parsing its output to determine actions.
        # For simulation, we'll hardcode some actions.
        simulated_actions: List[Action] = []
        simulated_observations: List[Observation] = []
        final_response_parts = []

        if "order status" in query.lower():
            order_id = 1002 # Assume parsing order_id from query
            action = Action(type="get_order_status", args={"order_id": order_id})
            simulated_actions.append(action)
            observation = self._call_tool(action)
            simulated_observations.append(observation)
            final_response_parts.append(f"Your order {order_id} status is: {observation.state.get('status')}. Tracking: {observation.state.get('tracking')}")
        elif "return" in query.lower():
            order_id = 1001 # Assume parsing
            item_id = 123 # Assume parsing
            action = Action(type="initiate_return", args={"order_id": order_id, "item_id": item_id})
            simulated_actions.append(action)
            observation = self._call_tool(action)
            simulated_observations.append(observation)
            final_response_parts.append(f"Return initiated for order {order_id}, item {item_id}. Return ID: {observation.state.get('return_id')}")
        elif "product" in query.lower():
            product_query = "gaming laptop" # Assume parsing
            action = Action(type="search_product", args={"query": product_query})
            simulated_actions.append(action)
            observation = self._call_tool(action)
            simulated_observations.append(observation)
            if observation.state.get('products'):
                final_response_parts.append(f"Found products: {[p['name'] for p in observation.state['products']]}")
            else:
                final_response_parts.append("Could not find any products matching your query.")
        else:
            final_response_parts.append("I am sorry, I can only help with order status, returns, or product search at the moment.")

        # Step 2: LLM generates a response based on observations and internal state
        # The quality of this response would be influenced by the RM during training (RLHF)
        generated_response = " ".join(final_response_parts)
        print(f"Agent generated response: {generated_response}")

        # Step 3: (Conceptual) Reward Model provides a score for the generated response
        # In a full RLHF loop, this score would guide the LLM's next actions/generations.
        # For this demo, we'll just print a conceptual score.
        conceptual_reward = random.uniform(0.5, 0.9) # Higher is better
        print(f"Conceptual Reward Model score for response: {conceptual_reward:.2f}")

        return generated_response


# --- Main Execution Flow for Demonstration ---
if __name__ == "__main__":
    print("--- Starting Dual Data Collection and Agent Training Simulation ---")

    # Simulate Data Collection
    print("\n--- Simulating Demonstration Data Collection ---")
    demonstrations = simulate_demonstration_recording()

    print("\n--- Starting Comparison Collector Gradio App (Run in browser) ---")
    print("    (Open the displayed URL in your browser to interact with the app)")
    comparison_app = create_comparison_collector_app()
    # For a real run, this would be comparison_app.launch(share=True) or similar
    # For this consolidated script, we'll just show it's ready.
    # comparison_app.launch()
    print("    Comparison Collector App ready (conceptual launch).")
    # In a real scenario, you would collect actual comparison data and load it here
    simulated_comparisons = [
        Comparison(customer_query="Where is my order?", response_a="It's on the way.", response_b="Your order #1234 is currently in transit and expected by Friday.", preferred_response="B"),
        Comparison(customer_query="How do I return a product?", response_a="Go to returns page.", response_b="You can initiate a return by visiting your order history and clicking 'Return Item'.", preferred_response="B"),
    ]
    print(f"Simulated {len(simulated_comparisons)} comparison data points.")

    # Simulate Training
    print("\n--- Simulating Behavior Cloning Training ---")
    bc_trainer = BehaviorCloningTrainer()
    prepared_bc_data = bc_trainer.prepare_data(demonstrations)
    trained_bc_model = bc_trainer.train(prepared_bc_data)

    print("\n--- Simulating Reward Model Training ---")
    rm_trainer = RewardModelTrainer()
    prepared_rm_data = rm_trainer.prepare_data(simulated_comparisons)
    trained_rm_model = rm_trainer.train(prepared_rm_data)

    # Simulate Agent Interaction
    print("\n--- Simulating Customer Support Agent Interaction ---")
    agent = CustomerSupportAgent(bc_model_weights=trained_bc_model, rm_model_weights=trained_rm_model)

    queries = [
        "What is the status of my order?",
        "I need to return an item.",
        "Can you find me a product?"
    ]

    for q in queries:
        print(f"\nCustomer: {q}")
        agent_response = agent.process_query(q)
        print(f"Agent: {agent_response}")

    print("\n--- Simulation Complete ---")
