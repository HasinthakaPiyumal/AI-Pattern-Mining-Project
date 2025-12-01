from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import random
import time

# --- 1. Data Collection & Annotation Platform (Simplified Models) ---

class Demonstration(BaseModel):
    agent_id: str
    timestamp: float
    customer_inquiry: str
    actions: list[dict]  # e.g., [{'type': 'click', 'target': 'order_button'}, {'type': 'text_input', 'field': 'search_bar', 'value': 'product_name'}]
    observations: list[dict] # e.g., [{'ui_state': 'product_page', 'tool_output': 'order_details'}]
    final_response: str

class Comparison(BaseModel):
    annotator_id: str
    timestamp: float
    customer_inquiry: str
    response_a: str
    response_b: str
    preferred_response: str # 'A', 'B', or 'Neither'

class Database:
    def __init__(self):
        self.demonstrations = []
        self.comparisons = []

    def save_demonstration(self, demo: Demonstration):
        self.demonstrations.append(demo)
        return {"status": "Demonstration saved", "id": len(self.demonstrations) - 1}

    def save_comparison(self, comp: Comparison):
        self.comparisons.append(comp)
        return {"status": "Comparison saved", "id": len(self.comparisons) - 1}

    def get_demonstrations(self):
        return self.demonstrations

    def get_comparisons(self):
        return self.comparisons

db = Database()

# --- 2. Model Training Pipeline (Simplified Mock Classes) ---

class BehaviorCloningModel:
    def __init__(self):
        self.is_trained = False
        self.model_state = None

    def train(self, demonstrations: list[Demonstration]):
        if not demonstrations:
            print("No demonstrations to train BC model.")
            return
        print(f"Training Behavior Cloning model with {len(demonstrations)} demonstrations...")
        time.sleep(2)  # Simulate training time
        self.model_state = {"weights": "simulated_bc_weights"}
        self.is_trained = True
        print("Behavior Cloning model trained.")

    def predict_action(self, state: dict) -> str:
        if not self.is_trained:
            return "I need to be trained first to predict actions."
        # Simplified: just return a canned response or mock action
        if "order" in state.get("inquiry", "").lower():
            return "Use 'OrderLookupTool' to find order details."
        return "Search 'ProductCatalogTool' for product information."

    def generate_response(self, inquiry: str, context: str) -> str:
        if not self.is_trained:
            return "(BC Model needs training) I am unable to generate a response at this moment."
        # Simulate LLM response based on BC
        return f"Based on your inquiry about '{inquiry}', I would typically look for '{context}' details. Can you provide more specifics?"


class RewardModel:
    def __init__(self):
        self.is_trained = False
        self.model_state = None

    def train(self, comparisons: list[Comparison]):
        if not comparisons:
            print("No comparisons to train Reward Model.")
            return
        print(f"Training Reward Model with {len(comparisons)} comparisons...")
        time.sleep(1.5)  # Simulate training time
        self.model_state = {"weights": "simulated_rm_weights"}
        self.is_trained = True
        print("Reward Model trained.")

    def predict_reward(self, response: str, inquiry: str) -> float:
        if not self.is_trained:
            return 0.0 # Default to no reward if not trained
        # Simplified: higher reward for responses that seem helpful
        if "specifics" in response.lower() or "details" in response.lower():
            return random.uniform(0.7, 0.9)
        if "unable" in response.lower():
            return random.uniform(0.1, 0.3)
        return random.uniform(0.4, 0.6)


class RLHFTrainer:
    def __init__(self, policy_model: BehaviorCloningModel, reward_model: RewardModel):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.is_fine_tuned = False

    def fine_tune(self, generated_responses_for_rm: list[tuple[str, str, str]]): # (inquiry, response_a, response_b)
        if not self.policy_model.is_trained or not self.reward_model.is_trained:
            print("Policy model or Reward Model not trained. Cannot perform RLHF.")
            return
        if not generated_responses_for_rm:
            print("No responses provided for RLHF fine-tuning.")
            return
        
        print(f"Performing RLHF fine-tuning...")
        # In a real scenario, this would involve iterative generation, reward calculation, and policy update.
        # Here, we just simulate the outcome.
        time.sleep(3)
        self.is_fine_tuned = True
        print("RLHF fine-tuning complete. Policy model is now aligned.")

# Instantiate models
bc_model = BehaviorCloningModel()
rm_model = RewardModel()
rlhf_trainer = RLHFTrainer(bc_model, rm_model)


# --- 3. Agent Orchestration & Inference ---

class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def use(self, *args, **kwargs):
        raise NotImplementedError

class OrderLookupTool(Tool):
    def __init__(self):
        super().__init__("OrderLookupTool", "Looks up customer order details using an order ID or customer email.")

    def use(self, order_id: str = None, customer_email: str = None) -> dict:
        print(f"Using OrderLookupTool with order_id={order_id}, customer_email={customer_email}")
        if order_id == "12345":
            return {"status": "success", "order_id": "12345", "item": "Laptop", "price": "$1200", "status": "Shipped"}
        elif customer_email == "test@example.com":
            return {"status": "success", "orders": [{"id": "12345", "item": "Laptop"}, {"id": "67890", "item": "Mouse"}]}
        return {"status": "error", "message": "Order not found."}

class ProductCatalogTool(Tool):
    def __init__(self):
        super().__init__("ProductCatalogTool", "Searches the product catalog for product information.")

    def use(self, product_name: str) -> dict:
        print(f"Using ProductCatalogTool with product_name={product_name}")
        if "laptop" in product_name.lower():
            return {"status": "success", "product": "High-Performance Laptop", "description": "Powerful laptop for gaming and work.", "price": "$1500"}
        return {"status": "error", "message": "Product not found."}

class CustomerSupportAgent:
    def __init__(self, policy_model: BehaviorCloningModel, reward_model: RewardModel, tools: list[Tool]):
        self.policy_model = policy_model
        self.reward_model = reward_model # Potentially used for self-evaluation or logging
        self.tools = {tool.name: tool for tool in tools}
        self.conversation_history = []

    def process_inquiry(self, inquiry: str) -> str:
        self.conversation_history.append({"role": "user", "content": inquiry})
        
        # Agent's reasoning process (simplified)
        context = {"inquiry": inquiry, "history": self.conversation_history}
        predicted_action = self.policy_model.predict_action(context)
        
        response = "I'm processing your request..."
        tool_output = None

        if "OrderLookupTool" in predicted_action:
            tool = self.tools.get("OrderLookupTool")
            if tool:
                # Mock extracting params from inquiry
                order_id = "12345" if "order 12345" in inquiry else None
                customer_email = "test@example.com" if "email test@example.com" in inquiry else None
                tool_output = tool.use(order_id=order_id, customer_email=customer_email)
                response = f"I've looked up the order: {tool_output.get('message', str(tool_output))}"
        elif "ProductCatalogTool" in predicted_action:
            tool = self.tools.get("ProductCatalogTool")
            if tool:
                product_name = next((word for word in ['laptop', 'mouse'] if word in inquiry.lower()), 'generic product')
                tool_output = tool.use(product_name=product_name)
                response = f"Here's what I found in the catalog: {tool_output.get('message', str(tool_output))}"
        else:
            # If no specific tool action, use LLM for general response
            response = self.policy_model.generate_response(inquiry, "general customer support")

        # Add agent's response to history
        self.conversation_history.append({"role": "agent", "content": response, "tool_output": tool_output})
        return response

# Instantiate tools and agent
order_tool = OrderLookupTool()
product_tool = ProductCatalogTool()
agent_tools = [order_tool, product_tool]
customer_agent = CustomerSupportAgent(bc_model, rm_model, agent_tools)

# --- FastAPI Application ---

app = FastAPI(
    title="E-commerce Customer Support Agent API",
    description="API for an intelligent customer support agent using dual data collection."
)

class InquiryRequest(BaseModel):
    customer_inquiry: str

class AgentResponse(BaseModel):
    response: str
    reward_score: float = None

@app.post("/demonstrations", summary="Submit a human demonstration")
async def submit_demonstration(demo: Demonstration):
    return db.save_demonstration(demo)

@app.post("/comparisons", summary="Submit a human comparison/preference")
async def submit_comparison(comp: Comparison):
    return db.save_comparison(comp)

@app.post("/train/bc", summary="Trigger Behavior Cloning model training")
async def train_bc_model():
    demonstrations = db.get_demonstrations()
    bc_model.train(demonstrations)
    return {"message": "BC model training initiated.", "is_trained": bc_model.is_trained}

@app.post("/train/rm", summary="Trigger Reward Model training")
async def train_rm_model():
    comparisons = db.get_comparisons()
    rm_model.train(comparisons)
    return {"message": "Reward Model training initiated.", "is_trained": rm_model.is_trained}

@app.post("/train/rlhf", summary="Trigger RLHF fine-tuning for the agent")
async def train_rlhf():
    # In a real scenario, we'd generate responses using the current policy
    # and then gather human feedback (comparisons) which would then be used here.
    # For this simplified example, we'll just simulate some generated responses for RM evaluation.
    mock_generated_responses = [
        ("Where is my order?", "I can help locate your order. Please provide your order ID or email.", "I don't know where your order is."),
        ("Tell me about the new laptop.", "We have a high-performance laptop. What features are you interested in?", "New laptop is good."),
    ]
    rlhf_trainer.fine_tune(mock_generated_responses)
    return {"message": "RLHF fine-tuning initiated.", "is_fine_tuned": rlhf_trainer.is_fine_tuned}


@app.post("/agent/inquire", response_model=AgentResponse, summary="Send an inquiry to the customer support agent")
async def agent_inquire(request: InquiryRequest):
    if not bc_model.is_trained:
        raise HTTPException(status_code=400, detail="Agent's policy model is not trained yet. Please train BC model first.")

    response_text = customer_agent.process_inquiry(request.customer_inquiry)
    reward_score = rm_model.predict_reward(response_text, request.customer_inquiry) if rm_model.is_trained else None

    return AgentResponse(response=response_text, reward_score=reward_score)


@app.get("/status", summary="Get training status of models")
async def get_status():
    return {
        "bc_model_trained": bc_model.is_trained,
        "rm_model_trained": rm_model.is_trained,
        "rlhf_fine_tuned": rlhf_trainer.is_fine_tuned,
        "demonstrations_count": len(db.get_demonstrations()),
        "comparisons_count": len(db.get_comparisons()),
    }


if __name__ == "__main__":
    # To run this FastAPI app:
    # 1. Save the code as `main.py`
    # 2. Install dependencies: `pip install fastapi uvicorn pydantic`
    # 3. Run from your terminal: `uvicorn main:app --reload`
    # 4. Access the API at http://127.0.0.1:8000/docs for interactive documentation.
    print("\n--- To run the FastAPI server, save this code as main.py and execute: ---")
    print("pip install fastapi uvicorn pydantic")
    print("uvicorn main:app --reload\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
