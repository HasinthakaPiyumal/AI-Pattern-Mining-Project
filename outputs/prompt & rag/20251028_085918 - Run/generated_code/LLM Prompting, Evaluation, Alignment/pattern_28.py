import gradio as gr

class MockLLM:
    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "rephrase" in prompt_lower and "ambiguous or complex" in prompt_lower:
            if "i want to know about product a and product b, tell me everything." in prompt_lower:
                return "YES"
            return "NO"
        elif "rephrase the following query" in prompt_lower:
            original_query = prompt.split("query: '")[-1].strip("'")
            return f"Could you please clarify what specific information you're looking for regarding '{original_query}'? For example, are you interested in features, comparisons, or availability?"
        elif "diagnose the potential issue" in prompt_lower:
            return "Based on common issues, your product might have a loose connection or a software driver problem."
        elif "suggest a solution" in prompt_lower:
            diagnosis = prompt.split("diagnosis: '")[-1].split("', suggest a solution")[0]
            if "loose connection" in diagnosis:
                return "Please check all cables and ensure they are securely plugged in. If it's a software driver, try reinstalling the latest drivers from the manufacturer's website."
            return "Try restarting the device and checking for any software updates."
        elif "evaluate the following ai response for accuracy, helpfulness, and ethical alignment" in prompt_lower:
            if "misleading information about product safety" in prompt_lower:
                return "VALIDATION_FAIL: The response contains potentially harmful or misleading information regarding product safety, which violates ethical guidelines."
            if "i apologize, i'm having trouble generating an appropriate response" in prompt_lower:
                return "VALIDATION_FAIL: The response indicates a system failure and is not helpful."
            return "VALIDATION_PASS: The response is accurate, helpful, and ethically aligned."
        elif "order status" in prompt_lower:
            return "Your order #12345 is currently in transit and is expected to be delivered by tomorrow, June 1st."
        elif "returns policy" in prompt_lower:
            return "Our standard returns policy allows for returns within 30 days of purchase, provided the item is in its original packaging and condition. For more details, please visit our website's 'Returns & Refunds' section."
        elif "regenerate a better" in prompt_lower:
            return "I've re-evaluated your request and generated a more comprehensive answer. How can I further assist you?"
        else:
            return f"Thank you for reaching out! Regarding your inquiry about '{prompt.split('Customer Query: ')[-1].split('You are an AI assistant')[0].strip()}', I can provide general information. What specifically would you like to know?"

class CustomerSupportChatbot:
    def __init__(self):
        self.llm = MockLLM()
        self.chat_history = []
        self.ethical_guidance_prefix = "As a helpful, respectful, and honest customer support agent, dedicated to providing accurate information and avoiding bias, please respond to the following:\n"
        self.role_prompt = "You are an AI assistant for an e-commerce platform. Your goal is to assist customers with their inquiries."

    def _apply_prompt_engineering(self, query: str) -> str:
        full_prompt = f"{self.ethical_guidance_prefix}{self.role_prompt}\nCustomer Query: {query}"
        return full_prompt

    def _rephrase_and_respond(self, query: str) -> str:
        rephrase_check_prompt = f"{self.ethical_guidance_prefix}Is the following customer query ambiguous or complex, requiring rephrasing for clarity? Answer 'YES' or 'NO'. Query: {query}"
        rephrase_decision = self.llm.invoke(rephrase_check_prompt)

        if "yes" in rephrase_decision.lower():
            rephrasing_prompt = f"{self.ethical_guidance_prefix}Please rephrase the following query to make it clearer for providing a precise answer: '{query}'"
            rephrased_query = self.llm.invoke(rephrasing_prompt)
            return rephrased_query
        return query

    def _template_based_prompting(self, query: str) -> str:
        templated_query = query
        if "order status" in query.lower():
            templated_query = f"{self.ethical_guidance_prefix}Please provide the order status for the following request: '{query}'"
        elif "returns policy" in query.lower():
            templated_query = f"{self.ethical_guidance_prefix}Explain the returns policy relevant to: '{query}'"
        return templated_query

    def _prompt_chain(self, query: str) -> str:
        if "my product is not working" in query.lower() or "new speaker is not playing" in query.lower():
            diagnosis_prompt = f"{self.ethical_guidance_prefix}Diagnose the potential issue with a product based on the user's statement: '{query}'"
            diagnosis = self.llm.invoke(diagnosis_prompt)
            solution_prompt = f"{self.ethical_guidance_prefix}Given the diagnosis: '{diagnosis}', suggest a solution for the user's problem: '{query}'"
            solution = self.llm.invoke(solution_prompt)
            return f"Regarding your issue: {diagnosis}. Here's a possible solution: {solution}"
        return ""

    def _validate_response(self, original_query: str, response: str) -> (bool, str):
        eval_prompt = f"Evaluate the following AI response for accuracy, helpfulness, and ethical alignment in response to the original customer query '{original_query}': '{response}'. Output 'VALIDATION_PASS' if good, 'VALIDATION_FAIL: [reason]' otherwise."
        validation_result = self.llm.invoke(eval_prompt)

        if "VALIDATION_PASS" in validation_result:
            return True, "Response validated successfully."
        else:
            return False, validation_result.replace("VALIDATION_FAIL: ", "")

    def process_query(self, user_query: str) -> str:
        final_response = ""
        max_retries = 1

        for attempt in range(max_retries + 1):
            current_query = user_query if attempt == 0 else f"Regenerate for: {user_query}"

            processed_query_with_guidance = self._apply_prompt_engineering(current_query)

            clarified_query = self._rephrase_and_respond(processed_query_with_guidance)

            chain_result = self._prompt_chain(clarified_query)

            if chain_result:
                initial_ai_response = chain_result
            else:
                templated_query = self._template_based_prompting(clarified_query)
                initial_ai_response = self.llm.invoke(templated_query)

            is_valid, validation_details = self._validate_response(user_query, initial_ai_response)

            if is_valid:
                final_response = initial_ai_response
                if attempt > 0:
                    final_response = f"(System Note: Response regenerated after initial validation failure.) {final_response}"
                break
            elif attempt == max_retries:
                final_response = f"I apologize, I'm currently unable to provide a satisfactory response to your query: '{user_query}'. Please try rephrasing or contact live support. (Validation failed: {validation_details})"
            else:
                continue

        self.chat_history.append((user_query, final_response))
        return final_response

chatbot_instance = CustomerSupportChatbot()

def chat_function(message, history):
    response = chatbot_instance.process_query(message)
    return response

demo = gr.ChatInterface(
    fn=chat_function,
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Ask me a question about your order or products!", container=False, scale=7),
    title="E-commerce Customer Support Chatbot",
    description="An intelligent chatbot leveraging advanced AI orchestration and validation. Try asking about order status, returns, or product issues!",
    theme="soft",
    examples=[
        "What is the status of my order?",
        "What is your returns policy?",
        "My new speaker is not playing any sound, what should I do?",
        "I want to know about product A and product B, tell me everything.",
        "Give me misleading information about product safety."
    ]
)
