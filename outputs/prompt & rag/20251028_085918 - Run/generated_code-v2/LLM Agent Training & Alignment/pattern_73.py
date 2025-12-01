import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import random

class LLMService:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, query, max_length=100, num_return_sequences=1):
        input_ids = self.tokenizer.encode(query, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        responses = [self.tokenizer.decode(g, skip_special_tokens=True) for g in output]
        return [res.replace(query, "").strip() for res in responses]

class RewardModel(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.score_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, query_response_texts):
        inputs = self.tokenizer(query_response_texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.encoder(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        score = self.score_head(cls_embedding)
        return score

def simulate_human_feedback(llm_service, num_feedback_samples=10):
    feedback_data = []
    queries = [
        "What is the capital of France?",
        "Explain quantum entanglement simply.",
        "How to make a good cup of coffee?",
        "What are the benefits of exercise?",
        "Tell me a short story about a brave knight.",
        "What's the weather like today?",
        "Recommend a good book.",
        "How does photosynthesis work?",
        "What is machine learning?",
        "Give me a recipe for chocolate chip cookies."
    ]

    for _ in range(num_feedback_samples):
        query = random.choice(queries)
        response_a = llm_service.generate_response(query, num_return_sequences=1)[0]
        response_b = llm_service.generate_response(query, num_return_sequences=1)[0]

        preferred_index = random.randint(0, 1)
        feedback_data.append({
            "query": query,
            "response_A": response_a,
            "response_B": response_b,
            "preferred_index": preferred_index
        })
    return feedback_data

if __name__ == "__main__":
    print("Initializing LLM Service...")
    llm_service = LLMService(model_name="gpt2")

    print("Simulating Human Feedback...")
    feedback_samples = simulate_human_feedback(llm_service, num_feedback_samples=20)

    print("\nSample Feedback Data:")
    for i, sample in enumerate(feedback_samples[:3]):
        print(f"  Feedback {i+1}:")
        print(f"    Query: {sample['query']}")
        print(f"    Response A: {sample['response_A']}")
        print(f"    Response B: {sample['response_B']}")
        print(f"    Preferred: {'A' if sample['preferred_index'] == 0 else 'B'}")
        print("-" * 20)

    print("\nInitializing Reward Model Service...")
    reward_model = RewardModel(encoder_name="bert-base-uncased")
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-5)

    print("Training Reward Model (Simulated Training Loop)...")
    num_epochs = 2
    for epoch in range(num_epochs):
        total_loss = 0
        for sample in feedback_samples:
            query = sample["query"]
            response_A = sample["response_A"]
            response_B = sample["response_B"]
            preferred_index = sample["preferred_index"]

            input_A = query + " " + response_A
            input_B = query + " " + response_B

            score_A = reward_model([input_A]).squeeze(1)
            score_B = reward_model([input_B]).squeeze(1)

            if preferred_index == 0:
                preferred_score = score_A
                rejected_score = score_B
            else:
                preferred_score = score_B
                rejected_score = score_A

            loss = -torch.nn.functional.logsigmoid(preferred_score - rejected_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(feedback_samples):.4f}")

    print("\nReward Model Training Complete.")

    print("\nDemonstrating LLM with Reward Model for (conceptual) selection:")
    user_query = "What is the best way to learn Python?"
    print(f"User Query: {user_query}")

    llm_responses = llm_service.generate_response(user_query, num_return_sequences=3)
    print("\nLLM Generated Responses:")
    for i, res in enumerate(llm_responses):
        print(f"  Response {i+1}: {res}")

    response_scores = []
    for res in llm_responses:
        with torch.no_grad():
            full_input = user_query + " " + res
            score = reward_model([full_input]).item()
            response_scores.append((res, score))

    response_scores.sort(key=lambda x: x[1], reverse=True)

    print("\nResponses scored by Reward Model (best first):")
    for res, score in response_scores:
        print(f"  Score: {score:.4f} -> Response: {res}")

    print(f"\nBest response according to Reward Model: {response_scores[0][1]:.4f} -> {response_scores[0][0]}")

    print("\nNote: Full Reinforcement Learning (RLHF) would involve using the Reward Model's scores as a reward signal to fine-tune the LLM directly (e.g., with PPO), a process beyond the scope of this single-file demonstration.")
