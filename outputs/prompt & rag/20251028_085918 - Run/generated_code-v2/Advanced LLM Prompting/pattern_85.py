import os
import openai
import pandas as pd
import random
import time

# Set your OpenAI API key from environment variable or replace with your key
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

class AutomatedPromptOptimizer:
    def __init__(self, exemplars: pd.DataFrame, llm_model: str = "gpt-3.5-turbo"):
        self.exemplars = exemplars
        self.llm_model = llm_model

    def _call_llm(self, prompt_text: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for generating and optimizing customer support prompts."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return ""

    def generate_initial_prompts(self, num_prompts: int = 3) -> list[str]:
        exemplar_summary = "\n".join(self.exemplars.apply(lambda x: f"Customer: {x['customer_query']}\nAgent: {x['agent_response']}", axis=1).tolist())
        
        base_prompt = f"""Based on the following historical customer interactions, generate {num_prompts} distinct ZeroShot instruction prompts for a customer support chatbot. Each prompt should aim to guide the chatbot to efficiently resolve customer queries related to e-commerce issues (e.g., order status, returns, product information).

Historical Interactions:
{exemplar_summary}

Generate each prompt on a new line, prefixed with 'Prompt #:'
"""
        raw_prompts = self._call_llm(base_prompt, temperature=0.9, max_tokens=500)
        return [p.split(':', 1)[1].strip() for p in raw_prompts.split('\n') if p.startswith('Prompt #:')]

    def paraphrase_prompt(self, prompt: str, num_variations: int = 2) -> list[str]:
        paraphrase_instruction = f"""Paraphrase the following instruction prompt in {num_variations} different ways. The paraphrased prompts should convey the same core instruction but use different wording, structure, or emphasis. Each paraphrase should be on a new line, prefixed with 'Variation #:'.

Original Prompt: {prompt}
"""
        raw_variations = self._call_llm(paraphrase_instruction, temperature=0.8, max_tokens=300)
        return [v.split(':', 1)[1].strip() for v in raw_variations.split('\n') if v.startswith('Variation #:')] if raw_variations else []

    def _simulate_chatbot_response(self, prompt: str, customer_query: str) -> str:
        chatbot_interaction = f"""You are a customer support chatbot. Follow the instruction prompt carefully to answer the customer's query.

Instruction Prompt: {prompt}

Customer Query: {customer_query}

Chatbot Response:"""
        return self._call_llm(chatbot_interaction, temperature=0.5, max_tokens=100)

    def score_prompt(self, prompt: str, test_queries: list[str]) -> float:
        scores = []
        for query in test_queries:
            response = self._simulate_chatbot_response(prompt, query)
            
            # Simple scoring heuristic: check for keywords indicating resolution or helpfulness
            # In a real system, this would involve NLP metrics, human evaluation, or A/B testing.
            resolution_keywords = ["order status", "tracking", "return process", "refund", "shipping", "product info", "assistance", "help", "solution"]
            helpful_score = sum(1 for keyword in resolution_keywords if keyword in response.lower()) * 10
            
            # Penalize for overly generic or unhelpful responses
            if len(response) < 20 or "I cannot help with that" in response.lower():
                helpful_score -= 20
            
            scores.append(max(0, helpful_score))
        
        return sum(scores) / len(scores) if scores else 0.0

    def optimize_prompts(self,
                         num_iterations: int = 5,
                         initial_prompt_count: int = 3,
                         variations_per_best_prompt: int = 2,
                         top_k_prompts: int = 2,
                         test_queries: list[str] = None) -> str:
        
        if test_queries is None:
            test_queries = [
                "Where is my order #12345?",
                "How do I return a faulty item?",
                "Can I get more information about the XYZ product?",
                "My payment failed, what should I do?"
            ]

        current_prompts = self.generate_initial_prompts(initial_prompt_count)
        best_prompt = ""
        highest_score = -1.0

        for i in range(num_iterations):
            print(f"\n--- Iteration {i+1}/{num_iterations} ---")
            scored_prompts = []
            for prompt in current_prompts:
                score = self.score_prompt(prompt, test_queries)
                scored_prompts.append((prompt, score))
                print(f"  Prompt: '{prompt}' | Score: {score:.2f}")
            
            scored_prompts.sort(key=lambda x: x[1], reverse=True)
            
            if not scored_prompts:
                print("No prompts to evaluate. Exiting.")
                break

            iteration_best_prompt, iteration_highest_score = scored_prompts[0]

            if iteration_highest_score > highest_score:
                highest_score = iteration_highest_score
                best_prompt = iteration_best_prompt
                print(f"  New best prompt found: '{best_prompt}' with score {highest_score:.2f}")

            if i < num_iterations - 1: # Don't generate variations on the last iteration
                top_prompts = [p for p, s in scored_prompts[:top_k_prompts]]
                next_iteration_prompts = []
                for p_idx, prompt in enumerate(top_prompts):
                    print(f"  Generating {variations_per_best_prompt} variations for top prompt #{p_idx+1}: '{prompt}'")
                    variations = self.paraphrase_prompt(prompt, variations_per_best_prompt)
                    next_iteration_prompts.extend(variations)
                
                # Add some new random prompts to prevent local optima
                new_random_prompts = self.generate_initial_prompts(max(0, initial_prompt_count - len(next_iteration_prompts)))
                current_prompts = list(set(next_iteration_prompts + new_random_prompts)) # Remove duplicates
                if not current_prompts:
                    print("Failed to generate new prompts for next iteration. Re-generating initial prompts.")
                    current_prompts = self.generate_initial_prompts(initial_prompt_count)
                
                # Small delay to avoid hitting API rate limits if many calls are made quickly
                time.sleep(1)

        print(f"\n--- Optimization Complete ---")
        print(f"Best optimized prompt: '{best_prompt}'")
        print(f"Achieved score: {highest_score:.2f}")
        return best_prompt

if __name__ == "__main__":
    # Example Historical Customer Interaction Data (Exemplars)
    exemplar_data = [
        {"customer_query": "My order #54321 hasn't shipped yet.", "agent_response": "I'll check the status of your order 54321 for you. Please hold."},
        {"customer_query": "How can I return a shirt that doesn't fit?", "agent_response": "You can initiate a return through our website's 'My Orders' section. Look for order details and select 'Return Item'."},
        {"customer_query": "Do you have the new XYZ headphones in stock?", "agent_response": "Let me check the current inventory for the XYZ headphones. What color are you interested in?"},
        {"customer_query": "My package was damaged upon arrival. What should I do?", "agent_response": "I apologize for the damaged package. Please provide your order number and we can arrange a replacement or refund."}
    ]
    exemplars_df = pd.DataFrame(exemplar_data)

    # Initialize and run the optimizer
    optimizer = AutomatedPromptOptimizer(exemplars=exemplars_df)
    optimized_prompt = optimizer.optimize_prompts(num_iterations=3, initial_prompt_count=3, variations_per_best_prompt=2, top_k_prompts=1)

    print(f"\nFinal Optimized Prompt for Chatbot: {optimized_prompt}")

    # Example of using the optimized prompt with a mock chatbot interaction
    print("\n--- Testing with Final Optimized Prompt ---")
    test_query_1 = "Where is my order?"
    test_query_2 = "How do I request a refund?"

    print(f"Query: '{test_query_1}'")
    print(f"Chatbot Response: {optimizer._simulate_chatbot_response(optimized_prompt, test_query_1)}")
    
    print(f"\nQuery: '{test_query_2}'")
    print(f"Chatbot Response: {optimizer._simulate_chatbot_response(optimized_prompt, test_query_2)}")
