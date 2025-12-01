class AmbiguityDetector:
    def is_ambiguous(self, query: str) -> bool:
        ambiguous_keywords = ["the red one", "that product", "it", "this one", "what is it"]
        for keyword in ambiguous_keywords:
            if keyword in query.lower():
                return True
        return False

class DemonstrationManager:
    def __init__(self):
        self.demonstrations = [
            {
                "ambiguous_query": "I want the red one.",
                "interpretations": ["Red shirt", "Red shoes", "Red hat"],
                "clarifying_questions": ["Which red product are you interested in?", "Could you please specify the item (e.g., 'red t-shirt', 'red sneakers')?"]
            },
            {
                "ambiguous_query": "Tell me about that product.",
                "interpretations": ["The last viewed product", "A popular product", "A recommended product"],
                "clarifying_questions": ["Which product are you referring to?", "Can you describe the product or tell me its name?"]
            }
        ]

    def get_demonstrations(self) -> str:
        formatted_demos = []
        for i, demo in enumerate(self.demonstrations):
            formatted_demos.append(f"Example {i+1}:\nUser: {demo['ambiguous_query']}\nInterpretations: {', '.join(demo['interpretations'])}\nClarification: {', '.join(demo['clarifying_questions'])}")
        return "\n\n" + "\n\n".join(formatted_demos)

class LLMIntegrator:
    def get_response(self, prompt: str) -> str:
        if "red product" in prompt.lower() or "red t-shirt" in prompt.lower():
            return "It sounds like you're looking for a red item. Do you mean a red t-shirt, red shoes, or something else?"
        elif "that product" in prompt.lower() or "which product" in prompt.lower():
            return "To help me understand which product you're interested in, could you please provide more details or its name?"
        elif "system_instruction" in prompt.lower() and "hello" in prompt.lower():
            return "Hello! How can I assist you with your shopping today?"
        else:
            return f"I received your query and I'm processing it. Based on the information, I can assist with: {prompt[:100]}... If your query was ambiguous, I might ask for more details."

def main():
    print("Welcome to the E-commerce Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")

    ambiguity_detector = AmbiguityDetector()
    demonstration_manager = DemonstrationManager()
    llm_integrator = LLMIntegrator()

    system_instruction = "You are an intelligent e-commerce customer support assistant. Your goal is to help users find products and answer their questions. If a query is ambiguous, try to ask clarifying questions or offer relevant options based on examples provided."

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        is_ambiguous = ambiguity_detector.is_ambiguous(user_query)
        
        current_prompt_parts = []
        current_prompt_parts.append(f"System Instruction: {system_instruction}")
        current_prompt_parts.append(f"User Query: {user_query}")

        if is_ambiguous:
            demonstrations = demonstration_manager.get_demonstrations()
            current_prompt_parts.append(f"Ambiguous Demonstrations:\n{demonstrations}")
            current_prompt_parts.append("Please help the user by asking clarifying questions or suggesting options, using the provided examples as guidance for handling ambiguity.")
        else:
            current_prompt_parts.append("Please provide a direct answer or assistance based on the user's clear query.")

        final_prompt = "\n".join(current_prompt_parts)
        
        chatbot_response = llm_integrator.get_response(final_prompt)
        print(f"Chatbot: {chatbot_response}")

if __name__ == '__main__':
    main()