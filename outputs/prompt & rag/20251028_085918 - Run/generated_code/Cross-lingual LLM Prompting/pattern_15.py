import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch # Import torch for device handling

class MultilingualChatbot:
    def __init__(self, model_name="google/flan-t5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.cross_lingual_examples = [
            {
                "english_query": "I cannot log in, what should I do?",
                "spanish_query": "No puedo iniciar sesión, ¿qué debo hacer?",
                "french_query": "Je n'arrive pas à me connecter, que dois-je faire?",
                "response_template": "Please try resetting your password using the 'Forgot Password' link on the login page. If the issue persists, contact support.",
            },
            {
                "english_query": "My internet connection is unstable.",
                "spanish_query": "Mi conexión a internet es inestable.",
                "german_query": "Meine Internetverbindung ist instabil.",
                "response_template": "Ensure your router is properly connected and try restarting it. If the problem continues, there might be an outage in your area or you may need to contact your ISP.",
            },
            {
                "english_query": "How do I upgrade my subscription?",
                "french_query": "Comment puis-je améliorer mon abonnement ?",
                "italian_query": "Come posso aggiornare il mio abbonamento?",
                "response_template": "You can upgrade your subscription from your account settings page under the 'Subscription' section. Follow the prompts to select a new plan.",
            },
        ]

    def _build_prompt(self, user_query: str, query_language: str = "english") -> str:
        prompt_parts = [
            "You are a helpful and empathetic multilingual customer support assistant.",
            "Your goal is to understand customer queries across different languages and provide appropriate responses.",
            "Here are some examples demonstrating how to handle similar issues in various languages, followed by a relevant response:",
            ""
        ]

        # Add in-context examples
        for i, example in enumerate(self.cross_lingual_examples):
            prompt_parts.append(f"### Example {i+1}:")
            
            # Show queries in various languages from the example
            if "english_query" in example:
                prompt_parts.append(f"Query (English): \"{example['english_query']}\"")
            if "spanish_query" in example:
                prompt_parts.append(f"Query (Spanish): \"{example['spanish_query']}\"")
            if "french_query" in example:
                prompt_parts.append(f"Query (French): \"{example['french_query']}\"")
            if "german_query" in example:
                prompt_parts.append(f"Query (German): \"{example['german_query']}\"")
            if "italian_query" in example:
                prompt_parts.append(f"Query (Italian): \"{example['italian_query']}\"")
            
            prompt_parts.append(f"Response: \"{example['response_template']}\"")
            prompt_parts.append("") # Separator

        # Add the current user query
        prompt_parts.append("---")
        prompt_parts.append(f"### Current Customer Query:")
        prompt_parts.append(f"Query ({query_language.capitalize()}): \"{user_query}\"")
        prompt_parts.append(f"Response:")

        return "\n".join(prompt_parts)

    def get_response(self, user_query: str, query_language: str = "english") -> str:
        if not user_query.strip():
            return "Please enter a query."

        full_prompt = self._build_prompt(user_query, query_language)
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=100, 
            num_beams=5, 
            early_stopping=True,
            no_repeat_ngram_size=2 
        )
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response_start_marker = "Response:"
        if response_start_marker in response_text:
            last_response_index = response_text.rfind(response_start_marker)
            extracted_response = response_text[last_response_index + len(response_start_marker):].strip()
            
            # Attempt to remove any potential repetition of the user query if the model echoes it
            if extracted_response.lower().startswith(user_query.lower()):
                extracted_response = extracted_response[len(user_query):].strip()
            return extracted_response
        
        return response_text.strip() # Fallback

# Initialize the chatbot
chatbot = MultilingualChatbot()

# Gradio Interface
iface = gr.Interface(
    fn=chatbot.get_response,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your customer support query here...", label="Your Query"),
        gr.Dropdown(
            choices=["english", "spanish", "french", "german", "italian"],
            label="Query Language",
            value="english"
        )
    ],
    outputs=gr.Textbox(label="Chatbot Response"),
    title="Multilingual Customer Support Chatbot (InCLT Prompting Demo)",
    description="This chatbot demonstrates InCLT Crosslingual Transfer Prompting. It uses examples in multiple languages to enhance cross-lingual understanding for customer support queries. Provide your query and its language to get a response."
)

if __name__ == "__main__":
    iface.launch()