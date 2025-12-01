import pandas as pd
import random
import json
import time

# --- Mock Libraries for Demonstration Purposes ---
# In a real application, these would be actual library imports
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
# from datasets import Dataset
# from accelerate import Accelerator
# from trl import RewardTrainer, SFTTrainer, PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
# from langchain.agents import AgentExecutor, create_react_agent, tool
# from langchain_core.prompts import ChatPromptTemplate
# from fastapi import FastAPI, HTTPException
# import uvicorn
# import gradio as gr
# import torch

class MockTokenizer:
    def encode(self, text): return [random.randint(0, 1000) for _ in text.split()]
    def decode(self, tokens): return " ".join([f"word_{t}" for t in tokens])
    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[1,2,3] for _ in texts], "attention_mask": [[1,1,1] for _ in texts]}

class MockModel:
    def __init__(self, name="mock_model"): self.name = name
    def generate(self, input_ids, max_length=50): return [[random.randint(0, 1000) for _ in range(random.randint(5, max_length))]]
    def __call__(self, *args, **kwargs): return self # Simulate model output

class MockDataset:
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
    def map(self, func, batched=False):
        return MockDataset([func(item) for item in self.data])
    @classmethod
    def from_pandas(cls, df): return cls(df.to_dict(orient="records"))

class MockTool:
    def __call__(self, *args, **kwargs): return f"MockTool result for {args[0] if args else kwargs}"
    def name(self): return "mock_tool"
    def description(self): return "A mock tool for demonstration."

class MockAgentExecutor:
    def __init__(self, agent_llm, tools):
        self.agent_llm = agent_llm
        self.tools = tools
    def invoke(self, inputs): return {"output": f"Agent processed query: {inputs['input']} using {self.tools[0].name()}"}

# --- Phase 1: Behavioral Cloning (How to interact) ---

def collect_demonstrations(num_demonstrations=10):
    demonstrations = []
    print(f"Collecting {num_demonstrations} simulated demonstrations...")
    for i in range(num_demonstrations):
        session_id = f"sess_{i}"
        customer_query = f"I need to find a {random.choice(['laptop', 'smartphone', 'headphone'])} under ${random.randint(500, 1500)}."
        actions = [
            {"action_type": "navigate", "action_details": {"url": "/products"}, "observation_html_snippet": "<title>Products</title>"},
            {"action_type": "click", "action_details": {"xpath": "//button[contains(text(), 'Electronics')]"}, "observation_html_snippet": "<title>Electronics</title>"},
            {"action_type": "type", "action_details": {"xpath": "//input[@id='search']", "text_input": customer_query.split()[-1]}, "observation_html_snippet": "<input value='smartphone'>"},
            {"action_type": "click", "action_details": {"xpath": "//button[contains(text(), 'Search')]"}, "observation_html_snippet": "<div class='results'>...</div>"},
            {"action_type": "answer", "action_details": {"text": f"Found several {customer_query.split()[-1]} options."}} # Final agent answer
        ]
        for step_number, action in enumerate(actions):
            demonstrations.append({
                "session_id": session_id,
                "step_number": step_number,
                "action_type": action["action_type"],
                "action_details": json.dumps(action["action_details"]),
                "observation_url": f"http://ecommerce.com/mock/{action['action_type']}",
                "observation_html_snippet": action["observation_html_snippet"],
                "customer_query": customer_query if step_number == 0 else ""
            })
    df_demonstrations = pd.DataFrame(demonstrations)
    df_demonstrations.to_json("demonstrations.jsonl", orient="records", lines=True)
    print("Demonstrations collected and saved to demonstrations.jsonl")
    return MockDataset.from_pandas(df_demonstrations)

def train_behavioral_cloning_model(demonstration_dataset):
    print("Simulating Behavioral Cloning model training...")
    # In a real scenario, this would involve loading a pre-trained LLM,
    # tokenizing the demonstration_dataset, and fine-tuning it.
    # Example with transformers (conceptual):
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    # def preprocess_function(examples):
    #     inputs = [f"Query: {q} History: {hist}" for q, hist in zip(examples["customer_query"], examples["history_context"])]
    #     targets = [json.dumps({"action_type": at, "action_details": ad}) for at, ad in zip(examples["action_type"], examples["action_details"])]
    #     model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    #     labels = tokenizer(targets, max_length=512, truncation=True)
    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs

    # tokenized_dataset = demonstration_dataset.map(preprocess_function, batched=True)
    # training_args = TrainingArguments(output_dir="./bc_model_output", num_train_epochs=3)
    # trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    # trainer.train()

    bc_model = MockModel("bc_llm")
    bc_tokenizer = MockTokenizer()
    print("Behavioral Cloning model training simulated. Returning mock model.")
    return bc_model, bc_tokenizer

# --- Phase 2: Reward Modeling (What constitutes a good output) ---

def generate_agent_responses(bc_model, bc_tokenizer, customer_queries, num_responses_per_query=2):
    print(f"Generating {num_responses_per_query} responses per query using BC model...")
    generated_responses = []
    for query_id, query_text in enumerate(customer_queries):
        for i in range(num_responses_per_query):
            # Simulate BC model generating a response based on the query
            inputs = bc_tokenizer(query_text, return_tensors="pt")
            output_tokens = bc_model.generate(inputs["input_ids"], max_length=50)
            response_text = bc_tokenizer.decode(output_tokens[0])
            generated_responses.append({
                "query_id": f"q_{query_id}",
                "response_id": f"resp_{query_id}_{i}",
                "customer_query": query_text,
                "agent_response": f"Simulated response {i+1} for '{query_text}': {response_text}."
            })
    df_responses = pd.DataFrame(generated_responses)
    df_responses.to_json("generated_responses.jsonl", orient="records", lines=True)
    print("Generated responses saved to generated_responses.jsonl")
    return df_responses

def collect_comparisons(generated_responses_df, num_comparisons=20):
    comparisons = []
    print(f"Collecting {num_comparisons} simulated comparisons from human annotators...")
    unique_queries = generated_responses_df["customer_query"].unique()
    for i in range(num_comparisons):
        query_text = random.choice(unique_queries)
        responses_for_query = generated_responses_df[generated_responses_df["customer_query"] == query_text]
        if len(responses_for_query) < 2:
            continue # Need at least two responses to compare

        response_pair = random.sample(responses_for_query.to_dict(orient="records"), 2)
        response_A = response_pair[0]
        response_B = response_pair[1]

        # Simulate human preference (randomly for this demo)
        preferred_response_id = response_A["response_id"] if random.random() > 0.5 else response_B["response_id"]
        comparison_reason = random.choice(["More concise", "More helpful", "Better accuracy", "More polite"])

        comparisons.append({
            "query_id": response_A["query_id"],
            "response_A_id": response_A["response_id"],
            "response_B_id": response_B["response_id"],
            "preferred_response_id": preferred_response_id,
            "comparison_reason": comparison_reason,
            "customer_query": query_text,
            "response_A_text": response_A["agent_response"],
            "response_B_text": response_B["agent_response"]
        })
    df_comparisons = pd.DataFrame(comparisons)
    df_comparisons.to_json("comparisons.jsonl", orient="records", lines=True)
    print("Comparisons collected and saved to comparisons.jsonl")
    return MockDataset.from_pandas(df_comparisons)

def train_reward_model(comparison_dataset, bc_tokenizer):
    print("Simulating Reward Model training...")
    # In a real scenario, this would involve training a model to predict
    # a scalar reward score based on the query and response, learning from preferences.
    # Example with trl (conceptual):
    # rm_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
    # def preprocess_rm_function(examples):
    #     # Combine query and response, tokenize
    #     pairs = [[q, r] for q, r in zip(examples["customer_query"], examples["response_A_text"])]
    #     # ... create input for pairwise ranking
    #     return rm_inputs
    # tokenized_rm_dataset = comparison_dataset.map(preprocess_rm_function, batched=True)
    # rm_trainer = RewardTrainer(model=rm_model, args=TrainingArguments(...), train_dataset=tokenized_rm_dataset, tokenizer=bc_tokenizer)
    # rm_trainer.train()

    reward_model = MockModel("reward_model")
    print("Reward Model training simulated. Returning mock model.")
    return reward_model

# --- Phase 3: Agent Deployment & Refinement ---

class EcommerceTool:
    def __init__(self, api_url="http://mock-ecommerce-api.com"): self.api_url = api_url

    def search_product(self, query: str): # @tool
        print(f"[Tool] Searching for product: {query}")
        time.sleep(0.5) # Simulate API call delay
        mock_results = {
            "laptop": ["Laptop X", "Laptop Y"],
            "smartphone": ["Smartphone A", "Smartphone B"],
            "headphone": ["Headphone 1", "Headphone 2"]
        }
        found = mock_results.get(query.lower().split()[-1], ["No product found"])
        return f"Found: {', '.join(found)}. Relevant to query '{query}'."

    def get_product_details(self, product_name: str): # @tool
        print(f"[Tool] Getting details for: {product_name}")
        time.sleep(0.5)
        if "Laptop X" in product_name: return "Laptop X: 16GB RAM, 512GB SSD, Intel i7. Price: $1200."
        if "Smartphone A" in product_name: return "Smartphone A: 6.1 inch display, 128GB storage. Price: $700."
        return f"Details for {product_name} not available."

    def get_order_status(self, order_id: str): # @tool
        print(f"[Tool] Checking order status for: {order_id}")
        time.sleep(0.5)
        if order_id == "ORD123": return "Order ORD123: Shipped, expected delivery 3-5 business days."
        return f"Order {order_id} not found."

class CustomerSupportAgent:
    def __init__(self, llm, tokenizer, tools):
        self.llm = llm # This would be the BC-trained and potentially RLHF-refined LLM
        self.tokenizer = tokenizer
        self.tools = tools

        # Simulate Langchain agent setup
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", "You are a helpful e-commerce customer support agent. Use the provided tools to answer customer questions."),
        #     ("human", "{input}")
        # ])
        # agent = create_react_agent(llm, tools, prompt)
        # self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        self.agent_executor = MockAgentExecutor(llm, tools) # Using mock agent executor

    def ask(self, query: str): # Simulate the agent's interaction
        print(f"\nAgent processing query: '{query}'")
        response = self.agent_executor.invoke({"input": query})
        return response["output"]

def run_rlhf_refinement(bc_model, bc_tokenizer, reward_model, comparison_dataset):
    print("Simulating RLHF refinement (Conceptual Placeholder)...")
    # This phase would use the BC model as the policy and the Reward Model
    # to provide feedback for an algorithm like PPO.
    # Example with trl (conceptual):
    # ppo_config = PPOConfig(batch_size=4, mini_batch_size=1, learning_rate=1e-5)
    # ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(bc_model.name, trust_remote_code=True)
    # ppo_trainer = PPOTrainer(config=ppo_config, model=ppo_model, ref_model=None,
    #                            tokenizer=bc_tokenizer, dataset=comparison_dataset,
    #                            reward_fn=lambda samples: reward_model(samples))
    # ppo_trainer.train()
    print("RLHF refinement simulated.")
    # For this demo, we just return the original BC model as the 'refined' model
    return bc_model

# --- FastAPI Application --- 

app = FastAPI()

ag_llm = None
ag_tokenizer = None
ag_agent = None

@app.on_event("startup")
async def startup_event():
    global ag_llm, ag_tokenizer, ag_agent
    print("Running AI training pipeline on startup...")

    # Phase 1: Behavioral Cloning
    demonstration_dataset = collect_demonstrations(num_demonstrations=5) # Reduced for quick startup
    bc_llm, bc_tokenizer = train_behavioral_cloning_model(demonstration_dataset)

    # Phase 2: Reward Modeling
    customer_queries_for_rm = [
        "What are the features of the latest iPhone?",
        "Can you recommend a gaming laptop?",
        "Where is my order ORD123?"
    ]
    generated_responses_df = generate_agent_responses(bc_llm, bc_tokenizer, customer_queries_for_rm, num_responses_per_query=2)
    comparison_dataset = collect_comparisons(generated_responses_df, num_comparisons=5) # Reduced for quick startup
    reward_model = train_reward_model(comparison_dataset, bc_tokenizer)

    # Phase 3: Agent Deployment & Refinement (Optional RLHF)
    refined_llm = run_rlhf_refinement(bc_llm, bc_tokenizer, reward_model, comparison_dataset)

    ecommerce_tools = [
        EcommerceTool().search_product,
        EcommerceTool().get_product_details,
        EcommerceTool().get_order_status
    ]
    ag_llm = refined_llm
    ag_tokenizer = bc_tokenizer # Using BC tokenizer for the agent
    ag_agent = CustomerSupportAgent(ag_llm, ag_tokenizer, ecommerce_tools)
    print("AI Agent pipeline completed. Agent ready for deployment.")

@app.post("/ask")
async def ask_agent(query: dict):
    if ag_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Please wait for startup.")
    customer_query = query.get("query")
    if not customer_query:
        raise HTTPException(status_code=400, detail="'query' field is required.")
    try:
        response = ag_agent.ask(customer_query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Gradio Interface ---

def gradio_interface(customer_query):
    if ag_agent is None:
        return "Agent is still initializing. Please wait a moment..."
    try:
        response = ag_agent.ask(customer_query)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # To run FastAPI: uvicorn customer_support_agent:app --reload
    # To run Gradio: python customer_support_agent.py

    print("Starting the customer support agent application. This will run the training pipeline.")
    print("Wait for 'AI Agent pipeline completed' message before interacting.")

    # Simulating FastAPI startup for the Gradio part
    # In a real setup, Gradio would connect to the running FastAPI endpoint.
    # For this single file demo, we directly call the startup_event and then run Gradio.
    import asyncio
    asyncio.run(startup_event())

    if ag_agent:
        gr_interface = gr.Interface(
            fn=gradio_interface,
            inputs=gr.Textbox(lines=2, placeholder="Ask the agent a question..."),
            outputs="text",
            title="E-commerce Customer Support AI Agent",
            description="An AI agent capable of navigating an e-commerce site and answering customer queries."
        )
        gr_interface.launch(inbrowser=True)
    else:
        print("Agent failed to initialize. Gradio interface will not be launched.")


# This combined script simulates the entire architecture. 
# To run the FastAPI server (in a separate terminal after installing uvicorn and fastapi):
# uvicorn customer_support_agent:app --reload
# Then you can send POST requests to http://127.0.0.1:8000/ask
# To run the Gradio interface (this will also trigger the training pipeline):
# python customer_support_agent.py
# The Gradio interface will open in your browser.
