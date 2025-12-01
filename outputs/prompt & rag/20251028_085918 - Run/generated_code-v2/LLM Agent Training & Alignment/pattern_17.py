import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from trl import PPOTrainer, PPOConfig
from datasets import Dataset as HFDataset

# 1. Base LLM Setup
class LLMGenerator:
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

    def generate_response(self, prompt, max_new_tokens=50, num_return_sequences=1):
        outputs = self.generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, do_sample=True, top_k=50, top_p=0.95)
        return [output["generated_text"][len(prompt):].strip() for output in outputs]

# 2. Simulated Human Feedback Collection
def simulate_human_feedback(queries):
    feedback_data = []
    # Simulate preference where a 'better' response might be slightly longer or more detailed
    for query in queries:
        # Dummy responses for demonstration
        response_a = f"This is a short answer to your question: {query}."
        response_b = f"Thank you for contacting us. Regarding your inquiry about {query}, we can provide a more comprehensive explanation. Please refer to our FAQ for more details."
        response_c = f"We received your question about {query}. Our team is currently reviewing it and will get back to you shortly."

        # Simulate human preference: B > A, C < A, B > C
        # For RLHF, we need pairs and a label or a score
        feedback_data.append({"query": query, "response_0": response_a, "response_1": response_b, "label": 1}) # response_1 (B) preferred over response_0 (A)
        feedback_data.append({"query": query, "response_0": response_a, "response_1": response_c, "label": 0}) # response_0 (A) preferred over response_1 (C)
        feedback_data.append({"query": query, "response_0": response_c, "response_1": response_b, "label": 1}) # response_1 (B) preferred over response_0 (C)
    return feedback_data

# 3. Reward Model (RM) Definition and Training
class RewardModel(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 1)

    def forward(self, embeddings):
        x = self.linear1(embeddings)
        x = self.relu(x)
        return self.linear2(x)

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, embedder):
        self.data = data
        self.tokenizer = tokenizer
        self.embedder = embedder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        response_0 = item["response_0"]
        response_1 = item["response_1"]
        label = item["label"]

        # For simplicity, we embed the full text (query + response)
        text_0 = query + " " + response_0
        text_1 = query + " " + response_1

        # Embeddings will be computed on the fly for this example
        # In a real scenario, you might pre-compute or batch embed
        return {"text_0": text_0, "text_1": text_1, "label": label}

def train_reward_model(feedback_data, reward_model, embedder, tokenizer, epochs=3, batch_size=4, learning_rate=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)
    optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)

    preference_dataset = PreferenceDataset(feedback_data, tokenizer, embedder)
    dataloader = DataLoader(preference_dataset, batch_size=batch_size, shuffle=True)

    # Hinge loss for preference ranking
    # We want reward(preferred) > reward(rejected)
    loss_fn = nn.MarginRankingLoss(margin=0.1) 

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            # Embed texts using sentence_transformers
            embeddings_0 = embedder.encode(batch["text_0"], convert_to_tensor=True, device=device)
            embeddings_1 = embedder.encode(batch["text_1"], convert_to_tensor=True, device=device)

            # Get reward scores
            score_0 = reward_model(embeddings_0)
            score_1 = reward_model(embeddings_1)

            # Label: 1 if response_1 preferred, 0 if response_0 preferred
            # For MarginRankingLoss, target is 1 if score_0 is expected to be lower than score_1
            # i.e., score_1 - score_0 > margin. So if label is 1 (response_1 preferred), target is 1.
            # If label is 0 (response_0 preferred), target is -1 (score_0 is expected to be higher than score_1).
            target = torch.tensor([1 if l == 1 else -1 for l in batch["label"]], dtype=torch.float, device=device)

            loss = loss_fn(score_1, score_0, target.unsqueeze(1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Reward Model Loss: {total_loss / len(dataloader):.4f}")

# 4. RLHF Optimization (Simplified using TRL's PPOTrainer)
class CustomRewardFunction:
    def __init__(self, reward_model, embedder):
        self.reward_model = reward_model
        self.embedder = embedder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_model.to(self.device)

    def __call__(self, samples):
        rewards = []
        for query, response in samples:
            full_text = query + " " + response
            embedding = self.embedder.encode(full_text, convert_to_tensor=True, device=self.device)
            reward = self.reward_model(embedding.unsqueeze(0)).item()
            rewards.append(torch.tensor(reward, device=self.device))
        return rewards

def run_rlhf(llm_generator, reward_model, embedder, ppo_epochs=2, generation_batches=2, samples_per_batch=4):
    config = PPOConfig(
        model_name=llm_generator.model.config._name_or_path,
        learning_rate=1e-5,
        ppo_epochs=ppo_epochs,
        batch_size=samples_per_batch,
        mini_batch_size=samples_per_batch,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    ppo_trainer = PPOTrainer(
        config,
        llm_generator.model,
        ref_model,
        llm_generator.tokenizer,
    )

    queries = ["How can I reset my password?", "What are your return policies?"]

    reward_fn = CustomRewardFunction(reward_model, embedder)

    for batch in range(generation_batches):
        print(f"RLHF Batch {batch+1}/{generation_batches}")
        # Generate responses for a batch of queries
        model_inputs = [llm_generator.tokenizer(q, return_tensors="pt").to(ppo_trainer.accelerator.device) for q in queries]
        
        generated_responses = []
        for input_ids in model_inputs:
            generation = ppo_trainer.generate(
                input_ids=input_ids["input_ids"],
                max_new_tokens=50,
                do_sample=True, top_k=50, top_p=0.95,
                return_prompts_only=False,
            )
            # Decode and store original query + generated text for reward computation
            decoded_query = llm_generator.tokenizer.decode(input_ids["input_ids"][0], skip_special_tokens=True)
            decoded_response = llm_generator.tokenizer.decode(generation[0], skip_special_tokens=True)[len(decoded_query):].strip()
            generated_responses.append((decoded_query, decoded_response))

        # Prepare for PPO trainer
        texts = [g[0] + " " + g[1] for g in generated_responses]
        rewards = reward_fn(generated_responses)

        # Dummy query_tensors and response_tensors for PPO trainer
        # In a real scenario, these would come from the generation process
        query_tensors = [llm_generator.tokenizer(q, return_tensors="pt").input_ids[0] for q, _ in generated_responses]
        response_tensors = []
        for i, (q, r) in enumerate(generated_responses):
            full_text_tokens = llm_generator.tokenizer(q + " " + r, return_tensors="pt").input_ids[0]
            response_tensors.append(full_text_tokens[len(query_tensors[i]):])
        
        # Create a dummy PPO dataset
        ppo_data = HFDataset.from_dict({
            "query": query_tensors,
            "response": response_tensors,
            "rewards": rewards
        })

        # This step performs the actual PPO update
        ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        print("\n--- LLM responses after RLHF step ---")
        for q in queries:
            print(f"Query: {q}")
            optimized_response = llm_generator.generate_response(q, num_return_sequences=1)[0]
            print(f"Optimized Response: {optimized_response}")
        print("---\n")

# 5. Rejection Sampling
def apply_rejection_sampling(llm_generator, reward_model, embedder, prompt, num_samples=5):
    candidate_responses = llm_generator.generate_response(prompt, num_return_sequences=num_samples)
    scored_responses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)

    for res in candidate_responses:
        full_text = prompt + " " + res
        embedding = embedder.encode(full_text, convert_to_tensor=True, device=device)
        with torch.no_grad():
            score = reward_model(embedding.unsqueeze(0)).item()
        scored_responses.append((res, score))
    
    best_response = max(scored_responses, key=lambda item: item[1])
    return best_response[0], best_response[1]

# 6. Main Orchestration
if __name__ == "__main__":
    # Initialize LLM and Tokenizer
    print("Initializing LLM...")
    llm_gen = LLMGenerator("distilgpt2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = llm_gen.tokenizer

    # Simulate Human Feedback
    print("\nSimulating human feedback...")
    customer_queries = [
        "My internet is not working.",
        "How do I update my billing information?",
        "I need to cancel my subscription."
    ]
    feedback_data = simulate_human_feedback(customer_queries)
    print(f"Collected {len(feedback_data)} feedback samples.")

    # Train Reward Model
    print("\nTraining Reward Model...")
    reward_model = RewardModel(embedding_dim=embedder.get_sentence_embedding_dimension())
    train_reward_model(feedback_data, reward_model, embedder, tokenizer)
    print("Reward Model trained.")

    # Demonstrate initial LLM response
    print("\n--- Initial LLM Response ---")
    test_query = "I have a problem with my order #12345."
    initial_response = llm_gen.generate_response(test_query)[0]
    print(f"Query: {test_query}")
    print(f"Response: {initial_response}")
    initial_score, _ = apply_rejection_sampling(llm_gen, reward_model, embedder, test_query, num_samples=1)
    print(f"(Simulated initial RM score: {reward_model(embedder.encode(test_query + ' ' + initial_score, convert_to_tensor=True).unsqueeze(0)).item():.4f})")

    # Run RLHF Optimization (Simplified)
    print("\nRunning RLHF optimization (simplified demonstration)...")
    run_rlhf(llm_gen, reward_model, embedder, ppo_epochs=1, generation_batches=1)
    print("RLHF optimization complete (demonstration).")

    # Demonstrate LLM response after RLHF
    print("\n--- LLM Response After RLHF (Demonstration) ---")
    test_query_after_rlhf = "My account is locked."
    optimized_response_after_rlhf = llm_gen.generate_response(test_query_after_rlhf)[0]
    print(f"Query: {test_query_after_rlhf}")
    print(f"Response: {optimized_response_after_rlhf}")
    optimized_score, _ = apply_rejection_sampling(llm_gen, reward_model, embedder, test_query_after_rlhf, num_samples=1)
    print(f"(Simulated optimized RM score: {reward_model(embedder.encode(test_query_after_rlhf + ' ' + optimized_score, convert_to_tensor=True).unsqueeze(0)).item():.4f})")

    # Demonstrate Rejection Sampling
    print("\n--- Demonstrating Rejection Sampling ---")
    rs_query = "How do I check my order status?"
    best_response_rs, score_rs = apply_rejection_sampling(llm_gen, reward_model, embedder, rs_query, num_samples=5)
    print(f"Query: {rs_query}")
    print(f"Best response via rejection sampling: {best_response_rs}")
    print(f"Reward Model Score for best response: {score_rs:.4f}")