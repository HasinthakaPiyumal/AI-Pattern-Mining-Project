
class CustomerSupportAgent:
    """
    An Intelligent Customer Support Agent for E-commerce leveraging Generative AI.
    This agent employs advanced prompt engineering and aims for controlled, quality-assured responses.
    """

    def __init__(self, model_name="mock_LLM"): # In a real scenario, this would initialize an actual LLM client
        self.model_name = model_name
        self.chat_history = []

    def _mock_llm_predict(self, prompt):
        """
        A mock function to simulate an LLM's response based on the prompt.
        In a real application, this would be an API call to an LLM (e.g., OpenAI, Gemini).
        """
        print(f"\n--- Mock LLM Receiving Prompt ---\n{prompt}\n-----------------------------------")

        if "price of a product" in prompt.lower() or "cost of item" in prompt.lower():
            return "The price of the item you are asking about is generally around $49.99, but please check the product page for the most current pricing and availability."
        elif "shipping time" in prompt.lower() or "delivery estimate" in prompt.lower():
            return "Standard shipping usually takes 3-5 business days. Expedited options are available at checkout for faster delivery."
        elif "return policy" in prompt.lower() or "how to return" in prompt.lower():
            return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please visit our returns page for detailed instructions."
        elif "technical issue" in prompt.lower() or "troubleshoot" in prompt.lower():
            return "I understand you're facing a technical issue. Could you please provide more details? Our technical support team is also available to assist you directly via live chat or phone."
        elif "biased" in prompt.lower() or "offensive" in prompt.lower():
            return "I cannot generate responses that are biased or offensive. My purpose is to provide helpful and respectful information."
        elif "ethical" in prompt.lower():
             return "My responses are designed to be ethical and aligned with company policies."
        else:
            return f"Thank you for contacting us. I am here to help. Regarding '{prompt.splitlines()[-1].replace('User Query:', '').strip()}', I can assist with product information, orders, and common queries. What specifically can I help you with?"

    def _apply_ethical_guidelines(self, text):
        """
        Simulates a Constitutional AI-like filtering layer.
        In a real system, this would involve more sophisticated checks using another LLM or rule-based system.
        """
        lower_text = text.lower()
        if "offensive" in lower_text or "unethical" in lower_text or "biased" in lower_text:
            return "I apologize, but I cannot provide information that is offensive, unethical, or biased. My purpose is to provide helpful and respectful assistance."
        if len(text) > 500: # Example: prevent overly long, rambling responses
            return text[:450] + "... (Response truncated for brevity and focus. Please refine your query if you need more details.)"
        return text

    def _generate_prompt(self, user_query, chat_history=None, product_info=None, prompt_type="zero-shot"):
        """
        Generates a tailored prompt based on the chosen engineering technique.
        """
        if chat_history is None:
            chat_history = []

        # Base persona and instruction for the agent
        base_instruction = (
            "You are a friendly, helpful, and professional customer support agent for an e-commerce company. "
            "Your goal is to provide accurate, concise, and polite answers to customer queries. "
            "Always maintain a positive and respectful tone. Avoid speculation and refer to official sources when necessary."
        )

        context_info = ""
        if product_info:
            context_info = f"\nProduct Information:\n{product_info['name']}: {product_info['description']} (Price: {product_info['price']}, Stock: {product_info['stock']})\n"

        prompt_parts = []

        if prompt_type == "role-based":
            prompt_parts.append(base_instruction)

        elif prompt_type == "few-shot":
            # Example dialogues for few-shot prompting
            few_shot_examples = [
                "User: What is your return policy?",
                "Agent: Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please visit our returns page for detailed instructions.",
                "User: Can I change my shipping address after placing an order?",
                "Agent: Unfortunately, we cannot change the shipping address once an order has been placed. Please contact customer service immediately for assistance."
            ]
            prompt_parts.extend(few_shot_examples)
            prompt_parts.append(base_instruction) # Still good to reinforce role

        elif prompt_type == "template-driven":
            template = (
                "\nCustomer Query: {query}\n"
                "Relevant Product Context: {context}\n"
                "Agent's Task: Provide a clear, polite, and factual answer based on the query and context. Focus on direct answers and helpful next steps.\n"
                "Agent Response:"
            )
            prompt = template.format(query=user_query, context=context_info.strip() if context_info else "No specific product context provided.")
            return base_instruction + "\n" + prompt

        elif prompt_type == "zero-shot":
            # Zero-shot relies heavily on the base instruction and LLM's general knowledge
            prompt_parts.append(base_instruction)

        else:
            # Default to zero-shot if an unknown type is provided
            prompt_parts.append(base_instruction)

        # Add chat history for context
        for entry in chat_history:
            prompt_parts.append(f"User: {entry['user']}")
            prompt_parts.append(f"Agent: {entry['agent']}")

        prompt_parts.append(context_info)
        prompt_parts.append(f"User Query: {user_query}")

        return "\n".join(p.strip() for p in prompt_parts if p.strip())

    def get_response(self, user_query, prompt_type="zero-shot", product_info=None):
        """
        Generates a customer support response using the LLM with applied prompt engineering.
        """
        # 1. Prompt Engineering
        prompt = self._generate_prompt(user_query, self.chat_history, product_info, prompt_type)

        # 2. LLM Inference (Mocked)
        raw_response = self._mock_llm_predict(prompt)

        # 3. Behavior Control & Quality Assurance (Post-processing/Evaluation hooks)
        #    - This is where more advanced evaluation frameworks would integrate.
        #    - LLM-based autorating, round-trip consistency checks, adversarial evaluation
        #      would typically happen asynchronously or in a testing pipeline.
        #      For real-time control, we apply ethical guidelines.

        final_response = self._apply_ethical_guidelines(raw_response)

        # Update chat history
        self.chat_history.append({"user": user_query, "agent": final_response})

        return final_response


# --- Example Usage ---
if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("\n--- Zero-Shot Prompting Example ---")
    response_zero = agent.get_response("What is the shipping time for an order?", prompt_type="zero-shot")
    print(f"Agent (Zero-Shot): {response_zero}")

    print("\n--- Role-Based Prompting Example ---")
    response_role = agent.get_response("Can you tell me about your refund process?", prompt_type="role-based")
    print(f"Agent (Role-Based): {response_role}")

    print("\n--- Few-Shot Prompting Example ---")
    response_few_shot = agent.get_response("I have a technical issue with my new smart blender. What should I do?", prompt_type="few-shot")
    print(f"Agent (Few-Shot): {response_few_shot}")

    print("\n--- Template-Driven Prompting Example (with Product Info) ---")
    product_data = {
        "name": "E-commerce Widget Pro",
        "description": "A multi-functional widget for home automation with advanced AI features.",
        "price": "$199.99",
        "stock": "In Stock"
    }
    response_template = agent.get_response("Tell me more about the 'E-commerce Widget Pro' and its price.", prompt_type="template-driven", product_info=product_data)
    print(f"Agent (Template-Driven): {response_template}")

    print("\n--- Ethical Guideline Test (Offensive Input - will be moderated) ---")
    response_ethical_bad = agent.get_response("Give me a biased opinion about your competitors.", prompt_type="zero-shot")
    print(f"Agent (Ethical Check 1): {response_ethical_bad}")

    print("\n--- Ethical Guideline Test (Long Response - will be truncated) ---")
    long_query = "I need a very detailed, several-paragraph explanation of the entire history of your company, its mission, vision, values, every product ever released, and a comprehensive analysis of the current market trends affecting each product line. Please provide at least 600 words." 
    response_ethical_long = agent.get_response(long_query, prompt_type="zero-shot")
    print(f"Agent (Ethical Check 2 - Truncation): {response_ethical_long}")

    print("\n--- Consecutive Queries with History ---")
    agent_with_history = CustomerSupportAgent()
    resp1 = agent_with_history.get_response("What is the price of the new XYZ smartphone?", prompt_type="zero-shot")
    print(f"Agent (1st Query): {resp1}")
    resp2 = agent_with_history.get_response("And what about its battery life?", prompt_type="zero-shot")
    print(f"Agent (2nd Query): {resp2}")
    resp3 = agent_with_history.get_response("How do I return it if I don't like it?", prompt_type="few-shot")
    print(f"Agent (3rd Query): {resp3}")
