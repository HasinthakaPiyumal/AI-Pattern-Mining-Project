import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI

# --- 1. Natural Language Understanding (NLU) Module ---
# Using dummy implementations for transformers and spacy for demonstration purposes.
# In a real scenario, you'd load pre-trained models.
class NLUModule:
    def __init__(self):
        self.mock_intents = {
            "order_status": ["where is my order", "track my package", "order status"],
            "product_info": ["tell me about product X", "features of item Y", "product details"],
            "return_request": ["how to return", "initiate a return for order"],
            "greeting": ["hello", "hi", "good morning"]
        }
        self.mock_entities = {
            "product_X": "product_id_123",
            "item_Y": "product_id_456",
            "order_123": "order_id_123",
            "package": None
        }

    def understand_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        intent = "unknown"
        entities = {}

        for k, v in self.mock_intents.items():
            for phrase in v:
                if phrase in query_lower:
                    intent = k
                    break
            if intent != "unknown":
                break

        for entity_name, entity_id in self.mock_entities.items():
            if entity_name.lower() in query_lower:
                entities[entity_name] = entity_id
        
        # Simple mock NER using spacy-like functionality
        if "order" in query_lower and any(char.isdigit() for char in query_lower):
            import re
            match = re.search(r"order[\s-]*(\w+)", query_lower)
            if match: 
                entities["order_id"] = match.group(1)
            else: # Fallback for just digits
                match = re.search(r"\d+", query_lower)
                if match: entities["order_id"] = match.group(0)

        return {"intent": intent, "entities": entities}

# --- 2. Knowledge Retrieval Module (RAG) ---
# Mocking Chroma and sentence-transformers
class KnowledgeRetrievalModule:
    def __init__(self):
        # Dummy knowledge base
        self.knowledge_base = {
            "product_id_123": "Product X is a high-performance gadget with 128GB storage. Price: $499.",
            "product_id_456": "Item Y is a sustainable fashion accessory made from recycled materials. Price: $79.",
            "faq_returns": "Returns are accepted within 30 days of purchase. Please visit our returns portal.",
            "faq_shipping": "Standard shipping takes 3-5 business days. Express options are available."
        }

    def retrieve_info(self, query_intent: str, entities: Dict[str, Any]) -> str:
        retrieved_data = []

        if "product_id_123" in entities.values():
            retrieved_data.append(self.knowledge_base["product_id_123"])
        if "product_id_456" in entities.values():
            retrieved_data.append(self.knowledge_base["product_id_456"])
        if query_intent == "return_request":
            retrieved_data.append(self.knowledge_base["faq_returns"])
        if query_intent == "order_status" and "order_id" in entities:
            retrieved_data.append(f"Checking status for order: {entities['order_id']}. This usually takes a moment...") # Simulate external call
        elif query_intent == "order_status":
            retrieved_data.append(self.knowledge_base["faq_shipping"])

        return " ".join(retrieved_data) if retrieved_data else "No specific information found."

# --- 3. Large Language Model (LLM) Core ---
# Using a basic text generation model from transformers for demonstration
class LLMCore:
    def __init__(self):
        # In a real application, you'd load a larger, fine-tuned LLM
        # from transformers import pipeline
        # self.generator = pipeline("text-generation", model="distilgpt2")
        pass

    def generate_response(self, prompt: str) -> str:
        # Mock LLM response generation
        if "Product X" in prompt and "features" in prompt:
            return "Product X is known for its high performance and 128GB storage, ideal for demanding users."
        elif "return" in prompt:
            return "To initiate a return, please visit our returns portal within 30 days of purchase."
        elif "order status" in prompt and "order_id" in prompt:
            order_id = prompt.split("order_id:")[1].split(" ")[0].strip()
            return f"Your order {order_id} is currently being processed and is expected to ship within 1-2 business days."
        else:
            return f"I understand you're asking about: {prompt}. How can I assist you further?"

# --- 4. Reward Model (RM) and Reinforcement Learning from Human Feedback (RLHF) Module ---
class RewardModel:
    def __init__(self):
        # Dummy reward model weights
        self.weights = {"relevance": 0.6, "helpfulness": 0.3, "safety": 0.1}

    def predict_score(self, response: str, query: str) -> float:
        # Simple mock scoring based on keywords
        score = 0.0
        if "assist" in response or "help" in response: score += 0.2
        if "product X" in query and "performance" in response: score += 0.3
        if "return" in query and "portal" in response: score += 0.3
        return min(1.0, score + 0.1) # Base score

class RLHFModule:
    def __init__(self, llm_core: LLMCore, reward_model: RewardModel):
        self.llm_core = llm_core
        self.reward_model = reward_model
        # In a real scenario, trl.PPO would be used for training
        self.optimized_responses = {}

    def optimize_llm(self, query: str, current_response: str, human_feedback_score: float):
        # Simulate a tiny optimization step
        if human_feedback_score > 0.7:
            self.optimized_responses[query] = current_response
            # In reality, this would involve updating LLM weights based on reward signal
        print(f"RLHF: Query '{query}' received feedback score {human_feedback_score}. Optimized: {query in self.optimized_responses}")

    def select_best_response(self, candidate_responses: List[str], query: str) -> str:
        # Rejection Sampling or selection based on RM
        best_response = candidate_responses[0]
        best_score = -1.0
        for response in candidate_responses:
            score = self.reward_model.predict_score(response, query)
            if score > best_score:
                best_score = score
                best_response = response
        return best_response

# --- 5. Behavior Cloning (BC) Module ---
class BehaviorCloningModule:
    def __init__(self, llm_core: LLMCore):
        self.llm_core = llm_core
        self.demonstrations = [] # List of {'input': ..., 'output': ...}

    def add_demonstration(self, input_text: str, output_text: str):
        self.demonstrations.append({"input": input_text, "output": output_text})

    def fine_tune_llm(self):
        # Simulate fine-tuning an LLM on demonstrations
        print(f"BC: Fine-tuning LLM with {len(self.demonstrations)} demonstrations.")
        # In reality, this would involve actual training using transformers Trainer
        if self.demonstrations:
            print(f"BC: LLM learned from: {self.demonstrations[-1]['input']} -> {self.demonstrations[-1]['output']}")

# --- 6. Data Collection and Management Module (Dual Data Collection) ---
class DataCollectionModule:
    def __init__(self):
        self.human_demonstrations = pd.DataFrame(columns=["query", "expert_response"])
        self.human_comparisons = pd.DataFrame(columns=["query", "response_A", "response_B", "preference"])

    def collect_demonstration(self, query: str, expert_response: str):
        new_demo = pd.DataFrame([{"query": query, "expert_response": expert_response}])
        self.human_demonstrations = pd.concat([self.human_demonstrations, new_demo], ignore_index=True)
        print(f"Data Collection: Collected new demonstration for query: {query}")

    def collect_comparison(self, query: str, response_A: str, response_B: str, preference: str):
        new_comp = pd.DataFrame([{"query": query, "response_A": response_A, "response_B": response_B, "preference": preference}])
        self.human_comparisons = pd.concat([self.human_comparisons, new_comp], ignore_index=True)
        print(f"Data Collection: Collected new comparison for query: {query}, preferred: {preference}")

    def get_demonstrations(self):
        return self.human_demonstrations.to_dict(orient='records')

    def get_comparisons(self):
        return self.human_comparisons.to_dict(orient='records')

# --- 7. Multi-stage RL with Reference Reuse Module ---
class MultiStageRLModule:
    def __init__(self, llm_core: LLMCore, reward_model: RewardModel):
        self.llm_core = llm_core
        self.reward_model = reward_model
        self.successful_trajectories = {}

    def process_multistage_query(self, initial_query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        if conversation_history is None: conversation_history = []

        # Simulate a multi-stage process with reference reuse
        current_prompt = initial_query
        if conversation_history:
            # Try to find a similar successful trajectory
            for history_key, saved_response in self.successful_trajectories.items():
                if initial_query in history_key:
                    print(f"Multi-stage RL: Reusing reference for '{initial_query}'")
                    return saved_response # Reuse a successful past response
            current_prompt = " ".join([f"{m['role']}: {m['content']}" for m in conversation_history]) + f" User: {initial_query}"

        # Generate a response using LLM Core
        response = self.llm_core.generate_response(current_prompt)
        return response

    def store_successful_trajectory(self, query: str, full_conversation: str, final_response: str, score: float):
        if score > 0.8: # Only store highly rated trajectories
            self.successful_trajectories[query + "_" + full_conversation] = final_response
            print(f"Multi-stage RL: Stored successful trajectory for query '{query}'")

# --- 8. Deployment and API Layer (FastAPI) ---
app = FastAPI()

# Initialize modules
nlu_module = NLUModule()
rag_module = KnowledgeRetrievalModule()
llm_core = LLMCore()
reward_model = RewardModel()
rlhf_module = RLHFModule(llm_core, reward_model)
bc_module = BehaviorCloningModule(llm_core)
data_collection_module = DataCollectionModule()
multistage_rl_module = MultiStageRLModule(llm_core, reward_model)

@app.post("/query")
async def handle_query(user_query: Dict[str, str]):
    query = user_query.get("query", "")
    conversation_history = user_query.get("history", []) # For multi-stage queries

    # 1. NLU
    nlu_output = nlu_module.understand_query(query)
    intent = nlu_output["intent"]
    entities = nlu_output["entities"]
    print(f"NLU Output: Intent='{intent}', Entities={entities}")

    # 2. RAG
    retrieved_info = rag_module.retrieve_info(intent, entities)
    print(f"RAG Output: Retrieved Info='{retrieved_info}'")

    # Prepare prompt for LLM
    llm_prompt = f"Customer query: {query}. \nRelevant information: {retrieved_info}. \nInstruction: Provide a helpful and concise response."

    # 7. Multi-stage RL (if applicable, otherwise directly use LLM core)
    if conversation_history:
        response = multistage_rl_module.process_multistage_query(query, conversation_history)
    else:
        # 3. LLM Core
        response = llm_core.generate_response(llm_prompt)

    # Simulate multiple candidate responses for Rejection Sampling
    candidate_responses = [response, 
                           llm_core.generate_response(f"Alternative: {llm_prompt}"),
                           llm_core.generate_response(f"Another option: {llm_prompt}")]

    # 4. RLHF (Selection)
    final_response = rlhf_module.select_best_response(candidate_responses, query)
    
    # For demonstration, simulate a fixed human feedback score for RLHF optimization
    # In a real system, this would come from actual human feedback.
    if "return" in query.lower():
        rlhf_module.optimize_llm(query, final_response, 0.9) # Simulate good feedback
    
    # 7. Store successful trajectories (simplified)
    multistage_rl_module.store_successful_trajectory(query, llm_prompt, final_response, reward_model.predict_score(final_response, query))

    return {"response": final_response, "intent": intent, "entities": entities}

@app.post("/admin/collect_demonstration")
async def collect_demonstration_endpoint(data: Dict[str, str]):
    query = data.get("query")
    expert_response = data.get("expert_response")
    data_collection_module.collect_demonstration(query, expert_response)
    bc_module.add_demonstration(query, expert_response) # Add to BC for fine-tuning
    bc_module.fine_tune_llm() # Trigger a mock fine-tune
    return {"status": "Demonstration collected and LLM mock fine-tuned."}

@app.post("/admin/collect_comparison")
async def collect_comparison_endpoint(data: Dict[str, str]):
    query = data.get("query")
    response_A = data.get("response_A")
    response_B = data.get("response_B")
    preference = data.get("preference")
    data_collection_module.collect_comparison(query, response_A, response_B, preference)
    return {"status": "Comparison collected."}

# To run this FastAPI app, save it as `ecommerce_ai_agent.py` and run:
# uvicorn ecommerce_ai_agent:app --reload

# Example usage with curl:
# Query endpoint:
# curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query": "Where is my order 123?"}'
# curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query": "Tell me about Product X"}'
# curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query": "How do I return something?"}'

# Admin demonstration collection endpoint:
# curl -X POST "http://127.00.1:8000/admin/collect_demonstration" -H "Content-Type: application/json" -d '{"query": "My package is late.", "expert_response": "I apologize for the delay. Could you please provide your order number?"}'

# Admin comparison collection endpoint:
# curl -X POST "http://127.00.1:8000/admin/collect_comparison" -H "Content-Type: application/json" -d '{"query": "Product X features?", "response_A": "It has 128GB storage.", "response_B": "It's a great gadget!", "preference": "A"}'
