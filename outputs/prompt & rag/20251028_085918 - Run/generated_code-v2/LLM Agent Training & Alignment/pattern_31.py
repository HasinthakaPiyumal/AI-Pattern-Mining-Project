import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random
from tqdm import tqdm

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    llm.config.pad_token_id = llm.config.eos_token_id

def get_response_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if input_ids.numel() == 0:
        return None

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        if last_hidden_states.shape[0] == 0:
            return None
        
        seq_lengths = attention_mask.sum(dim=1) - 1
        if (seq_lengths < 0).any():
            return None

        last_token_idx = min(seq_lengths[0].item(), last_hidden_states.shape[1] - 1)
        embedding = last_hidden_states[0, last_token_idx, :]
    return embedding

class RewardModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, embeddings):
        return self.layers(embeddings)

def simulate_human_feedback(query, responses):
    scores = []
    for i, res in enumerate(responses):
        score = random.uniform(0.1, 1.0) * (1 + len(res)/100)
        scores.append(score)
    
    total_score = sum(scores)
    if total_score > 0:
        scores = [s / total_score for s in scores]
    else:
        scores = [1.0 / len(responses)] * len(responses)

    return scores

def train_reward_model(rm, llm, tokenizer, num_training_samples=100, num_epochs=10, learning_rate=1e-3):
    optimizer = torch.optim.Adam(rm.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("Generating simulated human feedback data...")
    training_data = []
    sample_queries = [
        "What is your return policy?",
        "How do I reset my password?",
        "Can I track my order?",
        "What are your operating hours?",
        "Do you offer international shipping?"
    ]

    for _ in tqdm(range(num_training_samples)):
        query = random.choice(sample_queries)
        input_ids = tokenizer.encode(query, return_tensors="pt")
        candidate_responses_ids = llm.generate(
            input_ids,
            max_length=50 + len(input_ids[0]),
            num_return_sequences=3,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        candidate_responses = [tokenizer.decode(res, skip_special_tokens=True) for res in candidate_responses_ids]

        simulated_scores = simulate_human_feedback(query, candidate_responses)
        
        for i, res in enumerate(candidate_responses):
            if res.strip():
                training_data.append((res, simulated_scores[i]))

    print(f"Generated {len(training_data)} training samples for Reward Model.")

    print("Training Reward Model...")
    for epoch in range(num_epochs):
        rm.train()
        total_loss = 0
        random.shuffle(training_data)
        for response_text, true_score in tqdm(training_data, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                embedding = get_response_embedding(response_text, llm, tokenizer)
                if embedding is None:
                    continue

                predicted_score = rm(embedding.unsqueeze(0))
                loss = criterion(predicted_score.squeeze(), torch.tensor(true_score, dtype=torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                continue

        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    print("Reward Model training complete.")

class Chatbot:
    def __init__(self, llm, tokenizer, reward_model, num_candidate_responses=5):
        self.llm = llm
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.num_candidate_responses = num_candidate_responses
        self.reward_model.eval()

    def get_best_response(self, query):
        input_ids = self.tokenizer.encode(query, return_tensors="pt")

        candidate_responses_ids = self.llm.generate(
            input_ids,
            max_length=100 + len(input_ids[0]),
            num_return_sequences=self.num_candidate_responses,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

        candidate_responses = [self.tokenizer.decode(res, skip_special_tokens=True) for res in candidate_responses_ids]

        scores = []
        valid_responses = []
        for res_text in candidate_responses:
            if not res_text.strip():
                scores.append(-float('inf'))
                valid_responses.append(None)
                continue
            try:
                embedding = get_response_embedding(res_text, self.llm, self.tokenizer)
                if embedding is None:
                    scores.append(-float('inf'))
                    valid_responses.append(None)
                    continue

                with torch.no_grad():
                    score = self.reward_model(embedding.unsqueeze(0)).item()
                scores.append(score)
                valid_responses.append(res_text)
            except Exception as e:
                scores.append(-float('inf'))
                valid_responses.append(None)

        if not scores or all(s == -float('inf') for s in scores):
            return "I'm sorry, I couldn't generate a helpful response at this time."

        best_response_idx = np.argmax(scores)
        best_response = valid_responses[best_response_idx]

        return best_response if best_response else "I'm sorry, I couldn't generate a helpful response at this time."

if __name__ == "__main__":
    embedding_dim = llm.config.n_embd
    reward_model = RewardModel(embedding_dim)

    train_reward_model(reward_model, llm, tokenizer, num_training_samples=50, num_epochs=5)

    chatbot = Chatbot(llm, tokenizer, reward_model)

    print("\nChatbot initialized. Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break
        
        response = chatbot.get_best_response(user_query)
        print(f"Chatbot: {response}")