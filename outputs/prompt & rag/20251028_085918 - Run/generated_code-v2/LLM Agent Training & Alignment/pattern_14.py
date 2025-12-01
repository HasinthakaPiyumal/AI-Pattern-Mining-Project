import random
import time

# --- 1. Data Collection & Storage Layer ---

class DemonstrationDataCollector:
    def __init__(self):
        self.demonstrations = []

    def record_interaction(self, observation, action):
        # Simulate recording a human agent's interaction
        interaction = {"observation": observation, "action": action, "timestamp": time.time()}
        self.demonstrations.append(interaction)
        print(f"[Data Collector] Recorded demonstration: {observation} -> {action}")
        return interaction

    def get_demonstrations(self):
        return self.demonstrations

class ComparisonDataCollector:
    def __init__(self):
        self.comparisons = []

    def record_comparison(self, query, response_a, response_b, preference_label):
        # Simulate recording human preference for LLM responses
        comparison = {
            "query": query,
            "response_a": response_a,
            "response_b": response_b,
            "preference": preference_label,
            "timestamp": time.time()
        }
        self.comparisons.append(comparison)
        print(f"[Data Collector] Recorded comparison for query '{query}': {preference_label}")
        return comparison

    def get_comparisons(self):
        return self.comparisons

# --- 2. Data Processing & Feature Engineering Layer ---

class DemonstrationDataPreprocessor:
    def preprocess(self, raw_demonstrations):
        # Simulate cleaning and formatting demonstration data
        processed_data = []
        for demo in raw_demonstrations:
            # Dummy processing: just pass through for now
            processed_data.append({
                "input_sequence": demo["observation"],
                "target_action": demo["action"]
            })
        print(f"[Data Preprocessor] Processed {len(raw_demonstrations)} demonstrations.")
        return processed_data

class ComparisonDataPreprocessor:
    def preprocess(self, raw_comparisons):
        # Simulate formatting comparison data for Reward Model training
        processed_data = []
        for comp in raw_comparisons:
            # Dummy processing: create a simple tuple structure
            processed_data.append((
                comp["query"],
                comp["response_a"],
                comp["response_b"],
                comp["preference"]
            ))
        print(f"[Data Preprocessor] Processed {len(raw_comparisons)} comparisons.")
        return processed_data

# --- 3. Model Training Layer ---

class DummyLLM:
    """A very basic dummy LLM for simulation."""
    def __init__(self, model_name="dummy-llm"):
        self.model_name = model_name
        self.weights = {"dummy_weight": 0.5}

    def generate(self, prompt, **kwargs):
        # Simulate text generation
        if "order status" in prompt.lower():
            return "I am checking the order status for you."
        elif "technical issue" in prompt.lower():
            return "Please describe your technical issue in more detail."
        else:
            return f"This is a dummy response to: {prompt}"

    def __call__(self, inputs):
        # Simulate a forward pass for training/inference
        return self.generate(inputs)


class BehaviorCloningModel:
    def __init__(self, llm_backbone=None):
        self.model = llm_backbone if llm_backbone else DummyLLM("bc-llm")

    def train(self, processed_demonstrations):
        print(f"[BC Model] Starting Behavior Cloning training with {len(processed_demonstrations)} samples...")
        # In a real scenario, this would fine-tune self.model
        time.sleep(1)
        print("[BC Model] Behavior Cloning training complete.")

    def predict_action(self, observation):
        # Simulate predicting the next action based on observation
        dummy_actions = [
            "search_knowledge_base",
            "retrieve_order_details",
            "escalate_to_human",
            "suggest_FAQ"
        ]
        action = random.choice(dummy_actions)
        print(f"[BC Model] Predicted action for observation '{observation}': {action}")
        return action

class RewardModel:
    def __init__(self, llm_backbone=None):
        self.model = llm_backbone if llm_backbone else DummyLLM("rm-llm")

    def train(self, processed_comparisons):
        print(f"[Reward Model] Starting Reward Model training with {len(processed_comparisons)} samples...")
        # In a real scenario, this would fine-tune self.model to output reward scores
        time.sleep(1)
        print("[Reward Model] Reward Model training complete.")

    def predict_reward(self, query, response):
        # Simulate predicting a scalar reward for a given query and response
        reward = random.uniform(0.1, 1.0)
        print(f"[Reward Model] Predicted reward for response to '{query}': {reward:.2f}")
        return reward

# --- 4. Agentic LLM Fine-tuning (RLHF) ---

class AgenticLLMFinetuner:
    def __init__(self, bc_model, reward_model):
        self.bc_model = bc_model
        self.reward_model = reward_model
        self.agentic_llm = bc_model.model # Start with BC model as base

    def finetune_with_rlhf(self, simulated_environment_interactions=5):
        print(f"[RLHF Finetuner] Starting RLHF fine-tuning...")
        for i in range(simulated_environment_interactions):
            print(f"[RLHF Finetuner] - Iteration {i+1}: Simulating agent interaction...")
            # Simulate agent interacting with an environment
            simulated_observation = f"customer_query_step_{i}"
            agent_action = self.bc_model.predict_action(simulated_observation)
            agent_response = self.agentic_llm.generate(f"Perform {agent_action} based on {simulated_observation}")

            # Get reward from RM
            reward = self.reward_model.predict_reward(simulated_observation, agent_response)

            # In a real scenario, PPO/DPO would update self.agentic_llm weights
            print(f"[RLHF Finetuner]   Agent generated response with reward {reward:.2f}")
            time.sleep(0.5)

        print("[RLHF Finetuner] RLHF fine-tuning complete. Agentic LLM is ready.")
        return self.agentic_llm

# --- 5. Deployment & Inference Layer ---

class InternalSystemAPI:
    def get_order_details(self, order_id):
        print(f"[Internal System] Querying order details for {order_id}...")
        time.sleep(0.2)
        if order_id == "ORDER123":
            return {"order_id": order_id, "status": "shipped", "items": ["Laptop"], "delivery_date": "2023-12-25"}
        else:
            return {"order_id": order_id, "status": "not_found"}

    def search_knowledge_base(self, query):
        print(f"[Internal System] Searching knowledge base for '{query}'...")
        time.sleep(0.3)
        if "refund policy" in query.lower():
            return "Our refund policy allows returns within 30 days of purchase."
        elif "password reset" in query.lower():
            return "To reset your password, visit our website and click 'Forgot Password'."
        else:
            return "No relevant articles found in knowledge base."

class AgentOrchestrator:
    def __init__(self, agentic_llm, internal_system_api):
        self.agentic_llm = agentic_llm
        self.internal_system_api = internal_system_api
        self.tools = {
            "get_order_details": self.internal_system_api.get_order_details,
            "search_knowledge_base": self.internal_system_api.search_knowledge_base,
            # Add more tools as needed
        }

    def handle_customer_query(self, customer_query):
        print(f"\n--- [Agent Orchestrator] Handling query: '{customer_query}' ---")
        context = []
        final_response = "I'm sorry, I couldn't fully resolve your query. Please provide more details or wait for a human agent."

        # Simulate initial thought process/tool selection by LLM
        if "order status" in customer_query.lower():
            order_id = "ORDER123" # In a real system, LLM would extract this
            tool_response = self.tools["get_order_details"](order_id)
            context.append(f"Order details: {tool_response}")
            if tool_response["status"] == "shipped":
                final_response = f"Your order {order_id} has been shipped and is expected by {tool_response['delivery_date']}."
            else:
                final_response = f"I couldn't find details for order {order_id}."
        elif "refund" in customer_query.lower() or "return" in customer_query.lower():
            tool_response = self.tools["search_knowledge_base"]("refund policy")
            context.append(f"Knowledge base search: {tool_response}")
            final_response = f"Regarding your query about refunds: {tool_response}"
        elif "password" in customer_query.lower():
            tool_response = self.tools["search_knowledge_base"]("password reset")
            context.append(f"Knowledge base search: {tool_response}")
            final_response = f"Here's how to reset your password: {tool_response}"
        else:
            # If no specific tool is triggered, let the agentic LLM try to generate a response directly
            llm_direct_response = self.agentic_llm.generate(f"Customer query: {customer_query}. Provide a helpful response.")
            final_response = llm_direct_response

        print(f"[Agent Orchestrator] Final Response: {final_response}")
        return final_response

class MonitoringService:
    def log_interaction(self, query, agent_response, success=True):
        print(f"[Monitoring] Logged interaction: Query='{query}', Response='{agent_response[:50]}...', Success={success}")

    def record_metric(self, metric_name, value):
        print(f"[Monitoring] Recorded metric: {metric_name} = {value}")

# --- Main Application Workflow --- 

def main():
    print("Starting Dual Data Collection Agent Training Workflow")

    # 1. Data Collection
    demo_collector = DemonstrationDataCollector()
    comp_collector = ComparisonDataCollector()

    # Simulate human demonstrations
    demo_collector.record_interaction("customer_asking_order_status", "check_order_system('ORDER123')")
    demo_collector.record_interaction("customer_has_technical_issue", "search_knowledge_base('common_issues')")
    demo_collector.record_interaction("customer_wants_refund", "consult_refund_policy")

    # Simulate LLM generating responses for comparison
    dummy_llm_for_comparison = DummyLLM("initial-model")
    query1 = "My order hasn't arrived."
    resp_a1 = dummy_llm_for_comparison.generate(query1 + " Version A")
    resp_b1 = dummy_llm_for_comparison.generate(query1 + " Version B")
    comp_collector.record_comparison(query1, resp_a1, resp_b1, "A_is_better")

    query2 = "How do I reset my password?"
    resp_a2 = dummy_llm_for_comparison.generate(query2 + " Version A")
    resp_b2 = dummy_llm_for_comparison.generate(query2 + " Version B")
    comp_collector.record_comparison(query2, resp_a2, resp_b2, "B_is_better")

    # 2. Data Processing
    demo_preprocessor = DemonstrationDataPreprocessor()
    processed_demonstrations = demo_preprocessor.preprocess(demo_collector.get_demonstrations())

    comp_preprocessor = ComparisonDataPreprocessor()
    processed_comparisons = comp_preprocessor.preprocess(comp_collector.get_comparisons())

    # 3. Model Training
    llm_backbone = DummyLLM("shared-llm-backbone") # Simulate a shared LLM base

    bc_model = BehaviorCloningModel(llm_backbone=llm_backbone)
    bc_model.train(processed_demonstrations)

    reward_model = RewardModel(llm_backbone=llm_backbone)
    reward_model.train(processed_comparisons)

    # 4. Agentic LLM Fine-tuning (RLHF)
    rlhf_finetuner = AgenticLLMFinetuner(bc_model, reward_model)
    agentic_llm = rlhf_finetuner.finetune_with_rlhf()

    # 5. Deployment & Inference
    internal_systems = InternalSystemAPI()
    agent_orchestrator = AgentOrchestrator(agentic_llm, internal_systems)
    monitoring_service = MonitoringService()

    # Simulate customer queries
    customer_queries = [
        "What is the status of my order ORDER123?",
        "I need help with my refund.",
        "My internet is not working. Can you help?",
        "How do I change my account password?"
    ]

    for query in customer_queries:
        response = agent_orchestrator.handle_customer_query(query)
        monitoring_service.log_interaction(query, response, success=True)
        monitoring_service.record_metric("response_length", len(response))

    print("\nDual Data Collection Agent Training Workflow Completed.")

if __name__ == "__main__":
    main()