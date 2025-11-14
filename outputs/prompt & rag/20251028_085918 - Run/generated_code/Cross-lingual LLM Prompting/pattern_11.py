
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import random

class InCLTLLMService:
    def __init__(self, model_name="t5-small"):
        # For a real-world multilingual chatbot, consider larger models like:
        # "google/mt5-base", "facebook/mbart-large-50-many-to-many-mmt",
        # or fine-tuned versions of multilingual LLMs.
        # 't5-small' is used here for demonstration purposes due to its smaller size.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.llm_pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=100,
            temperature=0.7, # Added for more varied output
            top_p=0.9 # Added for more varied output
        )
        self.knowledge_base = self._load_dummy_knowledge_base()

    def _load_dummy_knowledge_base(self):
        # A simplified knowledge base containing example questions and answers
        # in English (en), Spanish (es), and French (fr).
        # In a production system, this would be a sophisticated data store
        # (e.g., a vector database, a structured knowledge graph).
        return [
            {
                "en_q": "How do I reset my password?",
                "en_a": "You can reset your password by visiting the account settings page and clicking 'Forgot Password'.",
                "es_q": "¿Cómo reseteo mi contraseña?",
                "es_a": "Puede restablecer su contraseña visitando la página de configuración de la cuenta y haciendo clic en 'Olvidé mi contraseña'.",
                "fr_q": "Comment réinitialiser mon mot de passe?",
                "fr_a": "Vous pouvez réinitialiser votre mot de passe en visitant la page des paramètres du compte et en cliquant sur 'Mot de passe oublié'."
            },
            {
                "en_q": "What are your business hours?",
                "en_a": "Our business hours are Monday to Friday, 9 AM to 5 PM local time.",
                "es_q": "¿Cuáles son sus horas de trabajo?",
                "es_a": "Nuestro horario comercial es de lunes a viernes, de 9 a.m. a 5 p.m. hora local.",
                "fr_q": "Quelles sont vos heures d'ouverture?",
                "fr_a": "Nos heures d'ouverture sont du lundi au vendredi, de 9h à 17h heure locale."
            },
            {
                "en_q": "How can I contact customer support?",
                "en_a": "You can contact customer support via email at support@example.com or call us at 1-800-123-4567.",
                "es_q": "¿Cómo puedo contactar al soporte al cliente?",
                "es_a": "Puede ponerse en contacto con el soporte al cliente por correo electrónico a support@example.com o llamarnos al 1-800-123-4567.",
                "fr_q": "Comment puis-je contacter le service client?",
                "fr_a": "Vous pouvez contacter le service client par e-mail à support@example.com ou nous appeler au 1-800-123-4567."
            }
        ]

    def _get_in_context_examples(self, query_lang: str, source_lang: str = "en", num_examples: int = 1):
        """
        Retrieves and formats in-context examples based on the InCLT pattern.
        This method samples from the knowledge base and creates examples
        covering source-source, source-target, target-source, and target-target language pairs.
        """
        available_examples = self.knowledge_base
        # Ensure we have enough examples; if not, use all available and potentially repeat
        if len(available_examples) < num_examples:
            selected_examples = available_examples * (num_examples // len(available_examples) + 1)
            selected_examples = random.sample(selected_examples, num_examples)
        else:
            selected_examples = random.sample(available_examples, num_examples)

        formatted_examples = []
        for ex in selected_examples:
            # Example 1: Source Language Question, Source Language Answer
            formatted_examples.append(f"Q ({source_lang.upper()}): {ex[f'{source_lang}_q']}\nA ({source_lang.upper()}): {ex[f'{source_lang}_a']}")
            # Example 2: Source Language Question, Target (Query) Language Answer
            formatted_examples.append(f"Q ({source_lang.upper()}): {ex[f'{source_lang}_q']}\nA ({query_lang.upper()}): {ex[f'{query_lang}_a']}")
            # Example 3: Target (Query) Language Question, Source Language Answer
            formatted_examples.append(f"Q ({query_lang.upper()}): {ex[f'{query_lang}_q']}\nA ({source_lang.upper()}): {ex[f'{source_lang}_a']}")
            # Example 4: Target (Query) Language Question, Target (Query) Language Answer
            formatted_examples.append(f"Q ({query_lang.upper()}): {ex[f'{query_lang}_q']}\nA ({query_lang.upper()}): {ex[f'{query_lang}_a']}")
        
        return "\n\n".join(formatted_examples)

    def generate_response(self, query: str, query_lang: str, source_lang: str = "en") -> str:
        """
        Generates a chatbot response by constructing a prompt with InCLT examples
        and feeding it to the multilingual LLM.
        """
        # Ensure target language exists in knowledge base, default to English if not for examples
        if f'{query_lang}_q' not in self.knowledge_base[0]:
            print(f"Warning: No examples found for query language '{query_lang}'. Falling back to '{source_lang}' for example answers.")
            # If query_lang examples are not available, we can only provide S-S and S-T (if T is supported by model)
            # For demonstration, we'll proceed but this is a real-world limitation.
            # In a real scenario, this would trigger a different example retrieval strategy or translation.
            pass # Keep using the provided query_lang as a target for the LLM

        in_context_examples = self._get_in_context_examples(query_lang, source_lang)

        # Construct the final prompt for the LLM
        # The prompt is designed to instruct the LLM to use the in-context examples
        # and respond in the specified query_lang.
        prompt = (
            f"You are a helpful multilingual customer support assistant. "
            f"Your task is to answer the following customer question in {query_lang.upper()}.\n\n"
            f"Here are some relevant examples of questions and answers in various language combinations "
            f"to help you understand cross-lingual contexts and generate an accurate response:\n\n"
            f"{in_context_examples}\n\n"
            f"Now, please answer the customer's question:\n"
            f"Q ({query_lang.upper()}): {query}\n"
            f"A ({query_lang.upper()}):"
        )

        print(f"DEBUG: Generated Prompt:\n{prompt}") # For debugging purposes

        try:
            output = self.llm_pipeline(prompt)
            response = output[0]["generated_text"].strip()
            # The t5-small model might generate additional tokens like "<pad>" or just repeat.
            # A simple cleanup:
            response = response.replace("<pad>", "").strip()
            return response
        except Exception as e:
            print(f"ERROR: Error generating response from LLM: {str(e)}")
            return f"Lo siento, no pude generar una respuesta en este momento. (Error: {str(e)})" # Default error in Spanish
