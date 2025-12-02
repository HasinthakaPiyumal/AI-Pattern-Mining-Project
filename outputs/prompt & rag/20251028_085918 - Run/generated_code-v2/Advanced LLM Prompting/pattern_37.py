import os
import json
import logging
import dspy

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

EXAMPLE_DATA_CONTENT = [
    {"query": "How do I reset my password?", "ideal_response": "You can reset your password by visiting our website's login page and clicking on the 'Forgot Password' link. Follow the instructions sent to your registered email address."},
    {"query": "What are your operating hours?", "ideal_response": "Our customer support operates from Monday to Friday, 9 AM to 5 PM local time."},
    {"query": "How can I check my order status?", "ideal_response": "To check your order status, please log in to your account on our website and navigate to the 'Order History' section."},
    {"query": "Do you offer international shipping?", "ideal_response": "Yes, we offer international shipping to select countries. Please see our shipping policy for more details and a list of eligible countries."}
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data Module
class DataLoader:
    def load_exemplars(self, data_content):
        return data_content

# Prompt Engineering Core
class CustomerSupportSignature(dspy.Signature):
    query = dspy.InputField()
    response = dspy.OutputField()

class InitialPromptGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("exemplars, task_description -> initial_prompt")

    def forward(self, exemplars, task_description):
        return self.generate(exemplars=exemplars, task_description=task_description)

class PromptParaphraser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.paraphrase = dspy.ChainOfThought("original_prompt, num_variations -> varied_prompts")

    def forward(self, original_prompt, num_variations):
        return self.paraphrase(original_prompt=original_prompt, num_variations=num_variations)

class ResponseScorer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.score_response = dspy.ChainOfThought("query, generated_response, ideal_response -> score, explanation")

    def forward(self, query, generated_response, ideal_response):
        return self.score_response(
            query=query,
            generated_response=generated_response,
            ideal_response=ideal_response
        )

class PromptScorer(dspy.Metric):
    def __init__(self, llm):
        self.llm = llm
        self.scorer_module = ResponseScorer()

    def __call__(self, prediction, gold, query):
        with dspy.settings(lm=self.llm):
            score_output = self.scorer_module(query=query, generated_response=prediction, ideal_response=gold)
            try:
                score_value = float(score_output.score)
            except ValueError:
                logging.warning(f"Could not parse score: {score_output.score}. Defaulting to 0.")
                score_value = 0.0
            return score_value

class APE_Optimizer:
    def __init__(self, exemplars, llm, metric, iterations=5, prompt_variations_per_step=3):
        self.exemplars = exemplars
        self.llm = llm
        self.metric = metric
        self.iterations = iterations
        self.prompt_variations_per_step = prompt_variations_per_step
        dspy.settings.configure(lm=self.llm)
        self.initial_prompt_generator = InitialPromptGenerator()
        self.prompt_paraphraser = PromptParaphraser()
        self.best_overall_prompt = None
        self.best_overall_score = -float('inf')

    def generate_initial_prompts(self, num_prompts):
        logging.info(f"Generating {num_prompts} initial prompts.")
        initial_prompts = []
        task_description = "Generate a ZeroShot instruction prompt for a customer support chatbot that answers user queries based on provided information."
        for _ in range(num_prompts):
            sample_exemplars = json.dumps(self.exemplars[:2])
            generated_output = self.initial_prompt_generator(exemplars=sample_exemplars, task_description=task_description)
            initial_prompts.append(generated_output.initial_prompt)
        return initial_prompts

    def evaluate_prompt(self, prompt_text, exemplars_to_test):
        logging.info(f"Evaluating prompt: '{prompt_text[:50]}...' with {len(exemplars_to_test)} exemplars.")
        total_score = 0
        customer_support_predictor = dspy.Predict(CustomerSupportSignature, instruction=prompt_text)

        for exemplar in exemplars_to_test:
            query = exemplar["query"]
            ideal_response = exemplar["ideal_response"]
            try:
                prediction = customer_support_predictor(query=query).response
                score = self.metric(prediction, ideal_response, query)
                total_score += score
            except Exception as e:
                logging.error(f"Error evaluating prompt for query '{query}': {e}")
                total_score += -100
        return total_score / len(exemplars_to_test) if exemplars_to_test else 0

    def create_prompt_variations(self, best_prompt_text, num_variations):
        logging.info(f"Creating {num_variations} variations for prompt: '{best_prompt_text[:50]}...' ")
        generated_output = self.prompt_paraphraser(original_prompt=best_prompt_text, num_variations=num_variations)
        variations_str = generated_output.varied_prompts
        variations = [v.strip() for v in variations_str.split(',') if v.strip()]
        return variations

    def run_optimization(self):
        current_prompts = self.generate_initial_prompts(self.prompt_variations_per_step)

        for i in range(self.iterations):
            logging.info(f"--- Iteration {i+1}/{self.iterations} ---")
            prompt_scores = []
            for prompt in current_prompts:
                score = self.evaluate_prompt(prompt, self.exemplars)
                prompt_scores.append((prompt, score))

            prompt_scores.sort(key=lambda x: x[1], reverse=True)
            best_prompts_this_iteration = prompt_scores[:1]

            if not best_prompts_this_iteration:
                logging.warning("No prompts evaluated successfully in this iteration.")
                continue

            current_best_prompt_text, current_best_score = best_prompts_this_iteration[0]
            logging.info(f"Iteration {i+1} Best Prompt (Score: {current_best_score:.2f}): '{current_best_prompt_text[:100]}...' ")

            if current_best_score > self.best_overall_score:
                self.best_overall_score = current_best_score
                self.best_overall_prompt = current_best_prompt_text
                logging.info(f"New overall best prompt found!")

            if i < self.iterations - 1:
                new_variations = self.create_prompt_variations(current_best_prompt_text, self.prompt_variations_per_step)
                current_prompts = [current_best_prompt_text] + new_variations
                current_prompts = list(set(current_prompts))

        logging.info(f"Optimization finished. Best overall prompt found (Score: {self.best_overall_score:.2f}):")
        return self.best_overall_prompt, self.best_overall_score

# Chatbot Integration
class CustomerSupportChatbot:
    def __init__(self, optimized_prompt, llm):
        self.optimized_prompt = optimized_prompt
        self.llm = llm
        dspy.settings.configure(lm=self.llm)
        self.customer_support_predictor = dspy.Predict(CustomerSupportSignature, instruction=self.optimized_prompt)

    def get_response(self, query):
        try:
            response = self.customer_support_predictor(query=query).response
            return response
        except Exception as e:
            logging.error(f"Error generating response for query '{query}': {e}")
            return "I am sorry, I am experiencing technical difficulties. Please try again later."

def main():
    logging.info("Starting Automated Customer Support Response Generator with APE.")

    llm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=OPENAI_API_KEY)
    dspy.settings.configure(lm=llm)

    data_loader = DataLoader()
    exemplars = data_loader.load_exemplars(EXAMPLE_DATA_CONTENT)
    logging.info(f"Loaded {len(exemplars)} exemplars.")

    prompt_scorer = PromptScorer(llm=llm)
    ape_optimizer = APE_Optimizer(exemplars=exemplars, llm=llm, metric=prompt_scorer, iterations=3, prompt_variations_per_step=2)
    optimized_prompt, best_score = ape_optimizer.run_optimization()

    logging.info(f"\n--- Final Optimized Prompt ---\n{optimized_prompt}\n----------------------------\n")
    logging.info(f"Best Score Achieved: {best_score:.2f}")

    logging.info("\n--- Starting Customer Support Chatbot ---")
    chatbot = CustomerSupportChatbot(optimized_prompt, llm)

    print("Hello! How can I assist you today? (Type 'exit' to quit)")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        
        bot_response = chatbot.get_response(user_query)
        print(f"Bot: {bot_response}")

if __name__ == "__main__":
    main()