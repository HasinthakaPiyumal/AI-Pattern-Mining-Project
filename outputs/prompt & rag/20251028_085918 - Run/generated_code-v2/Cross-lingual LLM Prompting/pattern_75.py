from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Ensure consistent language detection results
DetectorFactory.seed = 0

class MultilingualChatbot:
    def __init__(self):
        # Simple in-memory Knowledge Base (primarily in English)
        self.knowledge_base = {
            "customer service hours": "Our customer service hours are from 9 AM to 5 PM, Monday to Friday.",
            "how to reset password": "To reset your password, please visit our login page and click on 'Forgot Password'.",
            "shipping information": "Standard shipping takes 3-5 business days. Expedited options are available at checkout.",
            "return policy": "You can return items within 30 days of purchase with a valid receipt."
        }

        # Cross-lingual In-Context Learning (InCLT) Examples
        # These examples demonstrate to the LLM how to infer concepts cross-lingually
        # and respond in the user's language using English KB concepts as an intermediary.
        self.inclt_examples = [
            {
                "source_lang": "es",
                "source_query": "¿Cuál es el horario de atención al cliente?",
                "target_concept_answer": "Customer service hours. Our customer service hours are from 9 AM to 5 PM, Monday to Friday.",
                "response_in_source_lang": "Nuestro horario de atención al cliente es de 9 AM a 5 PM, de lunes a viernes."
            },
            {
                "source_lang": "fr",
                "source_query": "Comment puis-je réinitialiser mon mot de passe?",
                "target_concept_answer": "How to reset password. To reset your password, please visit our login page and click on 'Forgot Password'.",
                "response_in_source_lang": "Pour réinitialiser votre mot de passe, veuillez visiter notre page de connexion et cliquer sur 'Mot de passe oublié'."
            },
            {
                "source_lang": "de",
                "source_query": "Wie sind die Lieferzeiten?",
                "target_concept_answer": "Shipping information. Standard shipping takes 3-5 business days. Expedited options are available at checkout.",
                "response_in_source_lang": "Der Standardversand dauert 3-5 Werktage. Beschleunigte Optionen sind an der Kasse verfügbar."
            }
        ]

        # Load a multilingual LLM (using a T5 variant as a general-purpose LLM example)
        # Note: A truly dedicated multilingual model like mT5 or mBART might be better for production
        # but t5-small is used here for demonstration purposes and quicker loading.
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "en" # Default to English if detection fails

    def retrieve_kb_info(self, query):
        # Simplified KB retrieval: keyword matching
        query_lower = query.lower()
        relevant_info = []
        for key, value in self.knowledge_base.items():
            if any(word in query_lower for word in key.split()): # Simple keyword match
                relevant_info.append(f"Knowledge Base (English): {value}")
        return "\n".join(relevant_info) if relevant_info else "No direct information found in KB."

    def construct_inclt_prompt(self, user_query, detected_lang, relevant_kb_info):
        prompt_parts = []
        prompt_parts.append(f"You are a helpful customer support assistant. Answer the user's question in {detected_lang}.")
        prompt_parts.append("\nHere are some examples of how to answer cross-lingual queries:")

        for example in self.inclt_examples:
            prompt_parts.append(f"User ({example['source_lang']}): {example['source_query']}")
            prompt_parts.append(f"Thought Process (leveraging English KB): The user is asking about the concept: {example['target_concept_answer']}")
            prompt_parts.append(f"Answer ({example['source_lang']}): {example['response_in_source_lang']}\n")

        prompt_parts.append(f"\nHere is some relevant information from our knowledge base:\n{relevant_kb_info}\n")
        prompt_parts.append(f"Now, answer the following query in {detected_lang}:")
        prompt_parts.append(f"User ({detected_lang}): {user_query}")

        return "\n".join(prompt_parts)

    def get_llm_response(self, prompt_text):
        inputs = self.tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs.input_ids, 
            max_new_tokens=100,
            num_beams=5, 
            early_stopping=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to remove potential instruction echoes or leading phrases from T5
        if response.startswith("Answer (") or response.startswith("User (") or response.startswith("Thought Process ("):
            response = response.split(": ", 1)[1] if ": " in response else response

        return response.strip()

    def chat(self):
        print("\nWelcome to the Multilingual Customer Support Chatbot!")
        print("Type 'exit' to end the conversation.")

        while True:
            user_query = input("\nYou: ")
            if user_query.lower() == 'exit':
                print("Goodbye!")
                break

            detected_lang = self.detect_language(user_query)
            print(f"(Detected language: {detected_lang.upper()})")

            relevant_kb_info = self.retrieve_kb_info(user_query)
            prompt = self.construct_inclt_prompt(user_query, detected_lang, relevant_kb_info)
            
            # For debugging: print the full prompt
            # print("\n--- Constructed Prompt ---")
            # print(prompt)
            # print("--------------------------\n")

            llm_response = self.get_llm_response(prompt)
            print(f"Chatbot: {llm_response}")

if __name__ == "__main__":
    chatbot = MultilingualChatbot()
    chatbot.chat()