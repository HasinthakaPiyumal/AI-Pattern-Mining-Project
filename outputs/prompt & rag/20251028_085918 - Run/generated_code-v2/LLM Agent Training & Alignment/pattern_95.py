import gradio as gr
import numpy as np
import random

class SimulatedLLM:
    def generate_responses(self, query, num_samples=3):
        base_responses = [
            f"Thank you for contacting us regarding '{query}'. We are looking into this for you.",
            f"We've received your inquiry about '{query}' and our team will get back to you shortly.",
            f"Regarding '{query}', please allow us some time to investigate and provide a comprehensive answer.",
            f"Your question about '{query}' is important to us. A support agent will be in touch soon.",
            f"We appreciate you reaching out about '{query}'. We're working to resolve this."
        ]
        return random.sample(base_responses, min(num_samples, len(base_responses)))

class SimulatedRewardModel:
    def score_response(self, query, response):
        # Simulate scoring based on keywords and length
        score = 0
        if "thank you" in response.lower() or "appreciate" in response.lower():
            score += 0.5
        if "looking into" in response.lower() or "investigate" in response.lower() or "resolve" in response.lower():
            score += 0.8
        if len(response) > 80:
            score += 0.3 # Prefer slightly more detailed responses
        if query.lower() in response.lower():
            score += 1.0 # Highly relevant
        return score + random.uniform(-0.2, 0.2) # Add some randomness to simulate real model variation

llm = SimulatedLLM()
reward_model = SimulatedRewardModel()

def generate_best_response(customer_query, num_samples=5):
    candidate_responses = llm.generate_responses(customer_query, num_samples)
    scores = []
    for response in candidate_responses:
        score = reward_model.score_response(customer_query, response)
        scores.append(score)
    
    best_response_index = np.argmax(scores)
    return candidate_responses[best_response_index]

iface = gr.Interface(
    fn=generate_best_response,
    inputs=[
        gr.Textbox(lines=2, label="Customer Query", placeholder="e.g., My internet is not working."),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of Samples (N)")
    ],
    outputs=gr.Textbox(lines=3, label="Best Generated Response"),
    title="Smart Customer Support Response Generator (Best-of-N Rejection Sampling)",
    description="Enter a customer query to get an optimized support response using rejection sampling (Best-of-N)."
)

if __name__ == "__main__":
    iface.launch()