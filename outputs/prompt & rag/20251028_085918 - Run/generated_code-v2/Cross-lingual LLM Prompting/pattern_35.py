from transformers import pipeline

class InCLTPromptingModule:
    def __init__(self):
        self.examples = [
            {
                "intent": "cancel_subscription",
                "source_lang": "English",
                "source_query": "I want to cancel my subscription.",
                "source_response": "I can help you with that. Could you please confirm your account details?",
                "target_lang": "Spanish",
                "target_query": "Quiero cancelar mi suscripción.",
                "target_response": "Puedo ayudarte con eso. ¿Podrías confirmar los detalles de tu cuenta?"
            },
            {
                "intent": "product_inquiry",
                "source_lang": "English",
                "source_query": "Tell me about the new product features.",
                "source_response": "Certainly! Our new product includes enhanced AI capabilities and a streamlined user interface.",
                "target_lang": "French",
                "target_query": "Parlez-moi des nouvelles fonctionnalités du produit.",
                "target_response": "Certainement ! Notre nouveau produit comprend des capacités d'IA améliorées et une interface utilisateur simplifiée."
            }
        ]

    def construct_prompt(self, user_query: str, target_language: str) -> str:
        prompt_parts = [
            "Please assist a customer support agent in responding to queries. Below are examples of how to respond to customer inquiries in different languages, leveraging cross-lingual understanding."
        ]

        for i, example in enumerate(self.examples):
            # Example with source language query and response in target language
            prompt_parts.append(f"\nExample {2*i + 1} (From {example['source_lang']} to {example['target_lang']}):")
            prompt_parts.append(f"Customer ({example['source_lang']}): " + example['source_query'])
            prompt_parts.append(f"Agent ({example['target_lang']}): " + example['target_response'])

            # Example with target language query and response in target language
            prompt_parts.append(f"\nExample {2*i + 2} ({example['target_lang']} to {example['target_lang']}):")
            prompt_parts.append(f"Customer ({example['target_lang']}): " + example['target_query'])
            prompt_parts.append(f"Agent ({example['target_lang']}): " + example['target_response'])

        prompt_parts.append(f"\nNow, please respond to the following customer query in {target_language}.\n")
        prompt_parts.append(f"Customer ({target_language}): " + user_query)
        prompt_parts.append("Agent:")

        return "\n".join(prompt_parts)

class LLMIntegrationLayer:
    def __init__(self):
        try:
            self.generator = pipeline("text-generation", model="distilgpt2") # Using a small model for demonstration
        except Exception as e:
            print(f"Warning: Could not load text-generation pipeline. Make sure 'transformers' is installed and a model is available. Error: {e}")
            print("Proceeding with a mock LLM response.")
            self.generator = None

    def get_llm_response(self, prompt: str) -> str:
        if self.generator:
            # Limiting max_new_tokens to avoid excessively long generations for a simple demo
            # and removing the prompt from the generated text.
            response = self.generator(prompt, max_new_tokens=50, num_return_sequences=1, truncation=True)[0]['generated_text']
            # Post-process to remove the input prompt from the generated text
            if response.startswith(prompt):
                return response[len(prompt):].strip()
            return response.strip()
        else:
            # Mock LLM response
            print(f"[MOCK LLM RESPONSE for prompt]: {prompt[:100]}...")
            if "cancel my subscription" in prompt.lower():
                return "As a mock response, I understand you want to cancel your subscription. Please provide your account ID."
            elif "product features" in prompt.lower():
                return "As a mock response, the new product has many exciting features!"
            elif "cancelar mi suscripción" in prompt.lower():
                return "Como respuesta simulada, entiendo que desea cancelar su suscripción. Por favor, proporcione su ID de cuenta."
            else:
                return "As a mock response, I received your query in a multilingual context."

class MultilingualChatbot:
    def __init__(self):
        self.prompting_module = InCLTPromptingModule()
        self.llm_integration = LLMIntegrationLayer()

    def chat(self):
        print("Multilingual Customer Support Chatbot (Type 'exit' to quit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            target_lang = input("In which language do you expect the response (e.g., English, Spanish, French)? ")
            
            prompt = self.prompting_module.construct_prompt(user_input, target_lang)
            response = self.llm_integration.get_llm_response(prompt)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot = MultilingualChatbot()
    chatbot.chat()
