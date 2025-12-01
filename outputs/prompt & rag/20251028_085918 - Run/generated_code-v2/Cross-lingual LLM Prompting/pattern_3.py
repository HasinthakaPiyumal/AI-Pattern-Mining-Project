import os
from openai import OpenAI

class MultilingualChatbot:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.in_clt_examples = [
            {
                "source_language": "Spanish",
                "source_query": "¿Cuál es el estado de mi pedido?",
                "target_equivalent": "What is the status of my order?",
                "agent_response": "Your order is currently being processed and is expected to ship within 2 business days."
            },
            {
                "source_language": "French",
                "source_query": "Je souhaite retourner un article.",
                "target_equivalent": "I want to return an item.",
                "agent_response": "To initiate a return, please visit our returns page and follow the instructions provided."
            },
            {
                "source_language": "German",
                "source_query": "Wie ändere ich meine Lieferadresse?",
                "target_equivalent": "How do I change my delivery address?",
                "agent_response": "You can update your delivery address in your account settings under 'Shipping Information'."
            },
            {
                "source_language": "Spanish",
                "source_query": "¿Tienen este producto en otro color?",
                "target_equivalent": "Do you have this product in another color?",
                "agent_response": "Please specify the product name or ID, and I can check the available colors for you."
            }
        ]

    def _format_in_context_examples(self) -> str:
        formatted_examples = []
        for example in self.in_clt_examples:
            formatted_examples.append(
                f"Customer Query ({example['source_language']}): {example['source_query']}\n"
                f"English Equivalent: {example['target_equivalent']}\n"
                f"Agent Response: {example['agent_response']}"
            )
        return "\n---\n".join(formatted_examples) + "\n---\n"

    def ask_chatbot(self, user_query: str, source_language: str) -> str:
        in_context_prompt = self._format_in_context_examples()

        full_prompt = (
            f"{in_context_prompt}"
            f"Customer Query ({source_language}): {user_query}\n"
            f"English Equivalent:\n"
            f"Agent Response:"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful multilingual customer support assistant for an e-commerce platform. Provide concise and helpful responses based on the provided examples. All responses should be in English."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error communicating with the LLM: {e}"

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        chatbot = MultilingualChatbot(openai_api_key)

        print("Chatbot initialized. Type 'exit' to quit.")

        while True:
            user_input_lang = input("Enter source language (e.g., Spanish, French, German): ").strip()
            if user_input_lang.lower() == 'exit':
                break

            user_input_query = input("Enter your query: ").strip()
            if user_input_query.lower() == 'exit':
                break

            if not user_input_lang or not user_input_query:
                print("Both language and query are required. Please try again.")
                continue

            print(f"\nUser ({user_input_lang}): {user_input_query}")
            agent_response = chatbot.ask_chatbot(user_input_query, user_input_lang)
            print(f"Chatbot (English): {agent_response}\n")