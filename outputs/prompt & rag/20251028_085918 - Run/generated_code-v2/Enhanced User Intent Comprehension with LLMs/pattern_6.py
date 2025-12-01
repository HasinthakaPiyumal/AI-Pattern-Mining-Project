from sentence_transformers import SentenceTransformer, util
import torch

class IntentUnderstandingChatbot:
    def __init__(self):
        # Load a pre-trained sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Define intents and example phrases
        self.intents = {
            "Order Status": [
                "Where is my order?",
                "What is the status of my recent purchase?",
                "Track my delivery."
            ],
            "Product Inquiry": [
                "Tell me about product X.",
                "Do you have specifications for item Y?",
                "Information on model Z."
            ],
            "Technical Support": [
                "I need help with a technical issue.",
                "My device isn't working.",
                "Troubleshoot my software."
            ],
            "Billing Inquiry": [
                "I have a question about my bill.",
                "Explain my last invoice.",
                "How much do I owe?"
            ],
            "Greeting": [
                "Hello",
                "Hi there",
                "Good morning"
            ],
            "Goodbye": [
                "Goodbye",
                "See you later",
                "Bye for now"
            ]
        }

        self.intent_embeddings = self._get_intent_embeddings()

    def _get_intent_embeddings(self):
        embeddings = {}
        for intent, phrases in self.intents.items():
            embeddings[intent] = self.model.encode(phrases, convert_to_tensor=True)
        return embeddings

    def get_intent(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        best_intent = "Unknown"
        max_similarity = -1

        for intent, intent_phrase_embeddings in self.intent_embeddings.items():
            # Calculate cosine similarity between the query and all phrases for the current intent
            cosine_scores = util.cos_sim(query_embedding, intent_phrase_embeddings)
            
            # Take the maximum similarity score for this intent
            current_max_similarity = torch.max(cosine_scores).item()

            if current_max_similarity > max_similarity:
                max_similarity = current_max_similarity
                best_intent = intent

        # Set a threshold for intent recognition (can be tuned)
        if max_similarity < 0.5: # Example threshold
            return "Ambiguous / Needs Clarification", max_similarity

        return best_intent, max_similarity

    def chat(self):
        print("Chatbot initialized. Type 'exit' to quit.")
        while True:
            user_query = input("You: ")
            if user_query.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            
            intent, similarity = self.get_intent(user_query)
            print(f"Chatbot: I detect your intent as '{intent}' with a similarity score of {similarity:.2f}.")
            
            # Simple response logic based on detected intent
            if intent == "Order Status":
                print("Chatbot: Please provide your order number, and I will check its status for you.")
            elif intent == "Product Inquiry":
                print("Chatbot: Which product are you interested in? I can provide details.")
            elif intent == "Technical Support":
                print("Chatbot: I'm sorry you're experiencing issues. Please describe your problem in more detail, and I can connect you with a specialist if needed.")
            elif intent == "Billing Inquiry":
                print("Chatbot: To assist you with your billing inquiry, could you please provide your account details?")
            elif intent == "Greeting":
                print("Chatbot: Hello! How can I assist you today?")
            elif intent == "Goodbye":
                print("Chatbot: It was a pleasure assisting you. Have a great day!")
            else:
                print("Chatbot: I'm not quite sure I understand. Could you please rephrase or provide more context?")

if __name__ == "__main__":
    chatbot = IntentUnderstandingChatbot()
    chatbot.chat()
