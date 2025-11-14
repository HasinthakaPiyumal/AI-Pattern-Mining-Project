class LLM:
    """
    Simulated Large Language Model for demonstration.
    In a real application, this would interface with a pre-trained LLM
    like those from Hugging Face Transformers, OpenAI, or Google.
    """
    def generate(self, prompt, **kwargs):
        # Simulate LLM response based on keywords
        prompt_lower = prompt.lower()
        if "product information" in prompt_lower:
            return "This product is a multi-functional widget with advanced features. It costs $99.99."
        elif "shipping status" in prompt_lower:
            return "Your order #12345 is currently in transit and expected to arrive by next Friday."
        elif "return policy" in prompt_lower:
            return "Our return policy allows returns within 30 days of purchase with a valid receipt."
        elif "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! How can I assist you today?"
        elif "unethical" in prompt_lower or "harmful" in prompt_lower:
            return "I cannot provide information that is harmful or unethical."
        elif "thank you" in prompt_lower:
            return "You're welcome! Is there anything else I can help you with?"
        else:
            return "I'm sorry, I don't have enough information to answer that. Could you please rephrase your question?"

class PromptEngineer:
    """Manages dynamic prompt generation."""

    def __init__(self):
        self.base_template = "You are a helpful customer support assistant. Respond to the user's query:\nQuery: {query}\nResponse:"
        self.few_shot_examples = [
            {"query": "What is the warranty for product X?", "response": "Product X comes with a 1-year limited warranty."},
            {"query": "How do I reset my password?", "response": "To reset your password, visit the 'Forgot Password' link on the login page and follow the instructions."}
        ]

    def zero_shot_prompt(self, query):
        return self.base_template.format(query=query)

    def few_shot_prompt(self, query):
        examples_str = "\n".join([f"Query: {ex['query']}\nResponse: {ex['response']}" for ex in self.few_shot_examples])
        return f"Here are some examples of how to answer customer queries:\n{examples_str}\n\n{self.base_template.format(query=query)}"

    def role_based_prompt(self, query, role="customer support expert"):
        return f"You are acting as a {role}. Your task is to provide concise and accurate answers. {self.base_template.format(query=query)}"

    def select_prompt(self, query, strategy="zero-shot"):
        if strategy == "few-shot":
            return self.few_shot_prompt(query)
        elif strategy == "role-based":
            # Simple heuristic for role-based: if query mentions "expert" or "detailed"
            if "expert" in query.lower() or "detailed" in query.lower():
                return self.role_based_prompt(query, role="senior customer support specialist")
            return self.role_based_prompt(query)
        else: # default to zero-shot
            return self.zero_shot_prompt(query)

class EvaluationFramework:
    """Handles real-time evaluation of LLM responses."""

    def __init__(self, llm):
        self.llm = llm

    def llm_autorating(self, query, response):
        # Simulate LLM rating based on keywords
        if "sorry" in response.lower() or "rephrase" in response.lower():
            return {"score": 2, "feedback": "Response was not directly helpful, asked for rephrasing."}
        elif "unethical" in response.lower() or "harmful" in response.lower():
            return {"score": 1, "feedback": "Response refused due to ethical concerns."}
        elif query.lower() in response.lower() and len(response) > 30: # Simple heuristic for relevance and length
            return {"score": 4, "feedback": "Response seems relevant and sufficiently detailed."}
        return {"score": 3, "feedback": "Response is acceptable but could be improved."}

    def round_trip_consistency_check(self, query, response, original_data_store=None):
        # In a real system, 'original_data_store' would be a database or knowledge base.
        # Here, we simulate checking for consistency with some expected info.
        if original_data_store is None:
            original_data_store = {
                "product information": "multi-functional widget, $99.99",
                "warranty": "1-year limited warranty",
                "return policy": "30 days of purchase"
            }

        is_consistent = True
        feedback = []

        if "product information" in query.lower():
            if "widget" not in response.lower() or "$99.99" not in response.lower():
                is_consistent = False
                feedback.append("Product details (widget, price) not fully consistent.")
        elif "warranty" in query.lower():
            if "1-year" not in response.lower() and "warranty" in response.lower(): # simple check
                is_consistent = False
                feedback.append("Warranty details not explicitly confirmed as 1-year.")
        elif "return policy" in query.lower():
            if "30 days" not in response.lower() and "return policy" in response.lower(): # simple check
                is_consistent = False
                feedback.append("Return policy duration not explicitly confirmed as 30 days.")

        return {"consistent": is_consistent, "feedback": feedback if feedback else ["Response appears consistent with internal data."]}

    def adversarial_evaluation(self, query, response):
        # Simulate adversarial check for bias or unsafe content
        # This is a very simplistic simulation. Real adversarial eval involves more sophisticated models.
        potential_bias_keywords = ["illegal", "harmful", "discriminatory", "unsafe", "unethical"]
        for keyword in potential_bias_keywords:
            if keyword in response.lower():
                return {"flagged": True, "reason": f"Potentially unsafe or biased content detected: '{keyword}'"}
        # Check if the response directly contradicts a common ethical principle (simulated)
        if "lie" in response.lower() or "deceive" in response.lower():
             return {"flagged": True, "reason": "Response suggests unethical action."}
        return {"flagged": False, "reason": "No obvious unsafe or biased content detected."}

class ConstitutionalAI:
    """Applies ethical and safety principles to refine responses."""

    def apply_principles(self, response, principles=None):
        # In a real system, this would involve an LLM refining its own output
        # or a chain of prompts to guide behavior.
        # Here, we apply hardcoded rules for demonstration.

        if principles is None:
            principles = [
                "Be helpful and assist the user.",
                "Be harmless and avoid generating unsafe content.",
                "Be honest and truthful, do not fabricate information.",
                "Adhere to company policies and privacy regulations."
            ]

        # Rule 1: Prevent harmful content
        if "kill" in response.lower() or "harm" in response.lower():
            return "I cannot provide information that promotes harm or violates ethical guidelines."

        # Rule 2: Prevent dishonest claims (very basic check)
        if "definitely the best" in response.lower() and "competitor" in response.lower():
            return "I can provide information about our products, but cannot make subjective claims about competitor superiority."

        # Rule 3: Ensure privacy (placeholder for sensitive data)
        if "social security number" in response.lower() or "credit card number" in response.lower():
            return "I cannot ask for or process sensitive personal information like social security numbers or credit card details."

        return response # If no principles are violated, return original response

class CustomerSupportChatbot:
    """
    Intelligent customer support chatbot system integrating
    Generative AI Behavior Control and Quality Assurance.
    """

    def __init__(self):
        self.llm = LLM()
        self.prompt_engineer = PromptEngineer()
        self.eval_framework = EvaluationFramework(self.llm)
        self.constitutional_ai = ConstitutionalAI()
        self.conversation_history = []

    def process_query(self, user_query):
        print(f"\nUser: {user_query}")

        # 1. Dynamic Prompt Engineering
        # Simple strategy selection based on query length/complexity
        if len(user_query.split()) > 7 or ("explain" in user_query.lower() or "detail" in user_query.lower()):
            prompt_strategy = "few-shot" # Try to get more detailed response
        elif "what is" in user_query.lower() or "how to" in user_query.lower():
            prompt_strategy = "zero-shot"
        else:
            prompt_strategy = "role-based" # Default for general inquiries

        prompt = self.prompt_engineer.select_prompt(user_query, strategy=prompt_strategy)
        print(f"DEBUG: Selected Prompt Strategy: {prompt_strategy}")

        # 2. LLM Response Generation
        raw_response = self.llm.generate(prompt)
        print(f"DEBUG: Raw LLM Response: {raw_response}")

        # 3. Real-time Evaluation
        # LLM-based Autorating
        rating = self.eval_framework.llm_autorating(user_query, raw_response)
        print(f"DEBUG: Auto-rating: Score={rating['score']}, Feedback='{rating['feedback']}'")

        # Round-trip Consistency Check
        consistency = self.eval_framework.round_trip_consistency_check(user_query, raw_response)
        print(f"DEBUG: Consistency Check: Consistent={consistency['consistent']}, Feedback='{', '.join(consistency['feedback'])}'")

        # Adversarial Evaluation
        adversarial_check = self.eval_framework.adversarial_evaluation(user_query, raw_response)
        print(f"DEBUG: Adversarial Check: Flagged={adversarial_check['flagged']}, Reason='{adversarial_check['reason']}'")

        # 4. Constitutional AI for Refinement
        final_response = raw_response
        if adversarial_check['flagged']:
            print("DEBUG: Applying Constitutional AI due to adversarial flag.")
            final_response = self.constitutional_ai.apply_principles(f"Based on user query '{user_query}', the LLM generated: '{raw_response}'. However, it was flagged for: {adversarial_check['reason']}. Please refine to be safe and ethical.")
            if final_response == f"Based on user query '{user_query}', the LLM generated: '{raw_response}'. However, it was flagged for: {adversarial_check['reason']}. Please refine to be safe and ethical.": # If no actual rule caught it, use a generic fallback.
                final_response = "I cannot provide information that is unsafe or unethical. Please ask a different question."
        else:
            final_response = self.constitutional_ai.apply_principles(raw_response)


        self.conversation_history.append({"user": user_query, "bot": final_response})

        return final_response

    def run_chat(self):
        print("Welcome to the Generative AI Customer Support Chatbot!")
        print("Type 'exit' to end the conversation.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                print("Thank you for chatting! Goodbye.")
                break
            bot_response = self.process_query(user_input)
            print(f"Chatbot: {bot_response}")

# Main execution block
if __name__ == "__main__":
    chatbot = CustomerSupportChatbot()
    chatbot.run_chat()