import collections
import random

class MockLLM:
    def __init__(self, variability=0.1):
        self.variability = variability

    def generate(self, prompt):
        # Simulate different reasoning paths and answers for self-consistency
        if "Let's think step by step." in prompt:
            if "How do I reset my password?" in prompt:
                reasoning_options = [
                    "Reasoning: The user wants to reset their password. I should guide them to the 'Forgot Password' link, usually found on the login page, and instruct them to follow the prompts. The final answer is: Visit the login page and click 'Forgot Password'.",
                    "Reasoning: Password reset typically involves going to the login screen, clicking 'reset password', and entering an email. The final answer is: Go to the login page and use the 'Reset Password' option.",
                    "Reasoning: To reset a password, navigate to the sign-in page, find the 'forgot password' link, and follow the instructions to enter your email for a reset link. The final answer is: Find the 'Forgot Password' link on the login page."
                ]
                answer_options = [
                    "Visit the login page and click 'Forgot Password'.",
                    "Go to the login page and use the 'Reset Password' option.",
                    "Find the 'Forgot Password' link on the login page."
                ]
            elif "What is your return policy?" in prompt:
                reasoning_options = [
                    "Reasoning: The user is asking about the return policy. Most e-commerce platforms have a dedicated returns page. I should direct them there and mention common conditions like a 30-day window and original packaging. The final answer is: Our return policy allows returns within 30 days, provided the item is in its original condition. Please see our returns page for full details.",
                    "Reasoning: Return policies usually involve a time limit and condition of the item. I will inform the user about the 30-day window and the need for original packaging. The final answer is: You can return items within 30 days if they are in original condition. Check our returns page.",
                    "Reasoning: For return policy, key points are the timeframe and item status. We accept returns within 30 days in original packaging. The final answer is: Our return policy is 30 days for items in original packaging. Refer to our returns policy page."
                ]
                answer_options = [
                    "Our return policy allows returns within 30 days, provided the item is in its original condition. Please see our returns page for full details.",
                    "You can return items within 30 days if they are in original condition. Check our returns page.",
                    "Our return policy is 30 days for items in original packaging. Refer to our returns policy page."
                ]
            elif "How do I track my order?" in prompt:
                reasoning_options = [
                    "Reasoning: To track an order, customers typically log into their account, go to their order history, and find the tracking link for the specific order. Alternatively, they might use a tracking number on the carrier's website. The final answer is: Log in to your account, go to 'Order History', and click the tracking link next to your order.",
                    "Reasoning: Order tracking is done through the 'My Orders' section after logging in. There's usually a button or link for tracking. The final answer is: Go to 'Order History' in your account to find the tracking information.",
                    "Reasoning: Customers can track orders by logging into their profile and accessing their order list. Each order will have a tracking option. The final answer is: Check your 'Order History' after logging into your account for tracking."
                ]
                answer_options = [
                    "Log in to your account, go to 'Order History', and click the tracking link next to your order.",
                    "Go to 'Order History' in your account to find the tracking information.",
                    "Check your 'Order History' after logging into your account for tracking."
                ]
            else:
                # Generic CoT response for unseen queries
                reasoning_options = [
                    f"Reasoning: Based on the query '{prompt}', I need to provide a step-by-step solution. This involves understanding the user's intent, breaking down the problem, and formulating a concise answer. The final answer is: I need more information to assist you accurately.",
                    f"Reasoning: The query '{prompt}' requires detailed thought. I will analyze the keywords, infer the most likely user need, and then generate a direct response. The final answer is: Please rephrase your question for better understanding.",
                    f"Reasoning: To answer '{prompt}', I will consider common solutions for similar problems and present a clear path. The final answer is: I can help with that, but I need specific details."
                ]
                answer_options = [
                    "I need more information to assist you accurately.",
                    "Please rephrase your question for better understanding.",
                    "I can help with that, but I need specific details."
                ]

            # Introduce some variability
            if random.random() < self.variability:
                idx = random.randint(0, len(reasoning_options) - 1)
                return f"{reasoning_options[idx]}\nFinal Answer: {answer_options[idx]}"
            else:
                # Return the most 'common' or 'intended' answer more often
                return f"{reasoning_options[0]}\nFinal Answer: {answer_options[0]}"
        else:
            # Simulate direct answer without CoT for non-CoT prompts
            if "How do I reset my password?" in prompt:
                return "Visit the login page and click 'Forgot Password'."
            elif "What is your return policy?" in prompt:
                return "Our return policy allows returns within 30 days. See our returns page for details."
            elif "How do I track my order?" in prompt:
                return "Log in to your account and go to 'Order History' to track your order."
            else:
                return f"Response to '{prompt}': I am a chatbot. How can I help?"

def _generate_cot_response(llm, prompt, num_responses):
    responses = []
    for _ in range(num_responses):
        raw_output = llm.generate(prompt)
        # Basic parsing: assume 'Final Answer: ' delineates the answer
        if "Final Answer: " in raw_output:
            parts = raw_output.split("Final Answer: ", 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip()
            responses.append({"reasoning": reasoning, "answer": answer, "raw_output": raw_output})
        else:
            responses.append({"reasoning": raw_output, "answer": raw_output, "raw_output": raw_output})
    return responses

def _determine_most_consistent_answer(responses):
    answers = [r["answer"] for r in responses]
    if not answers:
        return None, 0.0
    
    answer_counts = collections.Counter(answers)
    most_common_answer, count = answer_counts.most_common(1)[0]
    agreement_score = count / len(answers)
    
    # Find a reasoning path that leads to the most common answer
    consistent_reasoning = None
    for r in responses:
        if r["answer"] == most_common_answer and "Reasoning:" in r["reasoning"]:
            consistent_reasoning = r["reasoning"]
            break

    return most_common_answer, consistent_reasoning, agreement_score

class ExemplarCreator:
    @staticmethod
    def generate_exemplars(llm, seed_problems, agreement_threshold=0.8, num_cot_runs=5):
        exemplars = []
        print(f"\n--- Generating Exemplars (Agreement Threshold: {agreement_threshold}, CoT Runs: {num_cot_runs}) ---")
        for i, problem in enumerate(seed_problems):
            print(f"Processing seed problem {i+1}/{len(seed_problems)}: {problem['query'][:50]}...")
            cot_prompt = problem["query"] + "\nLet's think step by step."
            responses = _generate_cot_response(llm, cot_prompt, num_cot_runs)
            most_consistent_answer, consistent_reasoning, agreement_score = _determine_most_consistent_answer(responses)
            
            print(f"  -> Agreement: {agreement_score:.2f}, Most Consistent Answer: {most_consistent_answer[:50]}...")

            if agreement_score >= agreement_threshold and consistent_reasoning:
                exemplars.append({
                    "query": problem["query"],
                    "expected_answer": most_consistent_answer,
                    "reasoning": consistent_reasoning
                })
                print(f"  -> Exemplar added for '{problem['query'][:30]}...' (Agreement: {agreement_score:.2f})")
            else:
                print(f"  -> Skipping exemplar for '{problem['query'][:30]}...' (Agreement too low or no consistent reasoning)")
        print("--- Exemplar Generation Complete ---")
        return exemplars

def _build_few_shot_prompt(exemplars, new_query):
    prompt_parts = []
    for ex in exemplars:
        # Format exemplar as Q&A with CoT
        prompt_parts.append(f"Q: {ex['query']}\n{ex['reasoning']}\nFinal Answer: {ex['expected_answer']}\n")
    
    prompt_parts.append(f"Q: {new_query}\nLet's think step by step.")
    return "\n".join(prompt_parts)

class COSPipeline:
    def __init__(self, llm_interface, exemplars=None):
        self.llm = llm_interface
        self.exemplars = exemplars if exemplars is not None else []

    def train_exemplars(self, seed_problems, agreement_threshold=0.8, num_cot_runs=5):
        print("\nStarting exemplar training...")
        self.exemplars = ExemplarCreator.generate_exemplars(self.llm, seed_problems, agreement_threshold, num_cot_runs)
        print(f"Training complete. {len(self.exemplars)} exemplars generated.")

    def answer_query(self, query, num_final_cot_runs=5):
        print(f"\n--- Answering Query with COSP: '{query[:50]}...' ---")
        if not self.exemplars:
            print("Warning: No exemplars trained. Falling back to ZeroShot CoT.")
            cot_prompt = query + "\nLet's think step by step."
        else:
            print(f"Using {len(self.exemplars)} exemplars for FewShot CoT.")
            cot_prompt = _build_few_shot_prompt(self.exemplars, query)

        responses = _generate_cot_response(self.llm, cot_prompt, num_final_cot_runs)
        most_consistent_answer, _, agreement_score = _determine_most_consistent_answer(responses)

        print(f"  -> Final Agreement: {agreement_score:.2f}")
        print(f"  -> Most Consistent Answer: {most_consistent_answer}")
        print("--- Query Answering Complete ---")
        return most_consistent_answer

# --- Demonstration --- 
if __name__ == "__main__":
    # 1. Initialize Mock LLM
    mock_llm = MockLLM(variability=0.3) # Increased variability to show self-consistency in action

    # 2. Define Seed Problems for Exemplar Training
    seed_problems_data = [
        {"query": "How do I reset my password?"},
        {"query": "What is your return policy for electronics?"},
        {"query": "I received a damaged item, what should I do?"},
        {"query": "How can I track my recent order?"}
    ]

    # 3. Initialize COSPipeline
    cosp_chatbot = COSPipeline(mock_llm)

    # 4. Train Exemplars (Offline Phase)
    cosp_chatbot.train_exemplars(seed_problems_data, agreement_threshold=0.7, num_cot_runs=7)

    print(f"\nExemplars trained: {len(cosp_chatbot.exemplars)}")
    for i, ex in enumerate(cosp_chatbot.exemplars):
        print(f"Exemplar {i+1}: Q: {ex['query'][:40]}... A: {ex['expected_answer'][:40]}...")
    
    # 5. Answer New Customer Queries (Online Phase)
    print("\n--- Answering New Queries ---")
    query1 = "My laptop charger stopped working. What is the warranty claim process?"
    answer1 = cosp_chatbot.answer_query(query1, num_final_cot_runs=7)
    print(f"Chatbot Response to '{query1}': {answer1}\n")

    query2 = "I want to return a shirt, but I lost the receipt. Can I still do it?"
    answer2 = cosp_chatbot.answer_query(query2, num_final_cot_runs=7)
    print(f"Chatbot Response to '{query2}': {answer2}\n")
    
    query3 = "Where is my package? The tracking number is ABC123XYZ."
    answer3 = cosp_chatbot.answer_query(query3, num_final_cot_runs=7)
    print(f"Chatbot Response to '{query3}': {answer3}\n")

    # Demonstrate without exemplars (e.g., if training failed or not yet run)
    print("\n--- Answering without trained exemplars (for comparison) ---")
    cosp_chatbot_no_exemplars = COSPipeline(mock_llm)
    query4 = "Can I pay with Bitcoin?"
    answer4 = cosp_chatbot_no_exemplars.answer_query(query4, num_final_cot_runs=7)
    print(f"Chatbot Response to '{query4}': {answer4}\n")


