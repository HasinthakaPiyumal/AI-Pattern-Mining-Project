import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

class BaseLLM:
    def __init__(self, model_name="distilgpt2", max_length=50):
        self.generator = pipeline("text-generation", model=model_name)
        self.max_length = max_length

    def generate_candidates(self, prompt: str, num_samples: int = 5) -> list[str]:
        responses = []
        for _ in range(num_samples):
            output = self.generator(
                prompt, 
                max_length=self.max_length, 
                num_return_sequences=1, 
                truncation=True,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            responses.append(output[0]['generated_text'].replace(prompt, "").strip())
        return responses

class RewardModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score_responses(self, query: str, responses: list[str]) -> list[float]:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        response_embeddings = self.model.encode(responses, convert_to_tensor=True)

        cosine_scores = util.cos_sim(query_embedding, response_embeddings)[0]
        return cosine_scores.tolist()

def generate_best_response(query: str, num_candidates: int = 5) -> str:
    llm = BaseLLM()
    reward_model = RewardModel()

    candidate_responses = llm.generate_candidates(query, num_candidates)
    scores = reward_model.score_responses(query, candidate_responses)

    if not scores:
        return "No valid responses generated."

    best_response_index = scores.index(max(scores))
    return candidate_responses[best_response_index]

if __name__ == "__main__":
    interface = gr.Interface(
        fn=generate_best_response,
        inputs=gr.Textbox(lines=2, placeholder="Enter customer query here..."),
        outputs="text",
        title="AI Customer Support Response Generator (Best-of-N Sampling)",
        description="Enter a customer query to get a high-quality response. The system generates multiple candidates and selects the best one using a reward model."
    )

    interface.launch()