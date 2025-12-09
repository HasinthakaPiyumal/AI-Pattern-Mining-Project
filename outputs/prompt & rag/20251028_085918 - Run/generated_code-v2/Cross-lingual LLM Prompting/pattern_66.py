
import random

# --- Mock Components (Simulating LLM, Translator, and Knowledge Base) ---

class MockLLM:
    def __init__(self, model_name="Multilingual_LLM_v1"):
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        # Simulate LLM processing time and generate a simple response
        if "Answer the following question" in prompt:
            # Try to extract the question and provide a generic answer
            lines = prompt.split('\n')
            question_line = ""
            for line in reversed(lines):
                if line.startswith("Customer Query (") and ":" in line:
                    question_line = line.split(": ", 1)[1]
                    break
            if question_line:
                return f"[Mock LLM Response in {self.model_name}]: Based on the context, here is an answer to '{question_line}': This is a comprehensive response generated from our knowledge. "
            else:
                return f"[Mock LLM Response in {self.model_name}]: I've processed your request based on the provided examples and context."
        return f"[Mock LLM Response in {self.model_name}]: I received your prompt and am providing a generic reply."

class MockTranslator:
    def __init__(self):
        self.translations = {
            "en": {
                "What is your return policy?": "What is your return policy?",
                "How do I track my order?": "How do I track my order?",
                "Can I change my shipping address?": "Can I change my shipping address?",
                "How long does shipping take?": "How long does shipping take?",
                "What payment methods do you accept?": "What payment methods do you accept?"
            },
            "es": {
                "What is your return policy?": "¿Cuál es su política de devoluciones?",
                "How do I track my order?": "¿Cómo hago seguimiento de mi pedido?",
                "Can I change my shipping address?": "¿Puedo cambiar mi dirección de envío?",
                "How long does shipping take?": "¿Cuánto tarda el envío?",
                "What payment methods do you accept?": "¿Qué métodos de pago aceptan?"
            },
            "fr": {
                "What is your return policy?": "Quelle est votre politique de retour ?",
                "How do I track my order?": "Comment suivre ma commande ?",
                "Can I change my shipping address?": "Puis-je changer mon adresse de livraison ?",
                "How long does shipping take?": "Combien de temps prend l'expédition ?",
                "What payment methods do you accept?": "Quels modes de paiement acceptez-vous ?"
            }
        }
        self.answers = {
            "en": {
                "What is your return policy?": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
                "How do I track my order?": "You can track your order using the tracking number provided in your shipping confirmation email.",
                "Can I change my shipping address?": "Shipping addresses cannot be changed once an order has been shipped. Please contact support immediately for unshipped orders.",
                "How long does shipping take?": "Standard shipping typically takes 5-7 business days, while express shipping takes 2-3 business days.",
                "What payment methods do you accept?": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay."
            },
            "es": {
                "Our return policy allows returns within 30 days of purchase with a valid receipt.": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra con un recibo válido.",
                "You can track your order using the tracking number provided in your shipping confirmation email.": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en el correo electrónico de confirmación de envío.",
                "Shipping addresses cannot be changed once an order has been shipped. Please contact support immediately for unshipped orders.": "Las direcciones de envío no se pueden cambiar una vez que un pedido ha sido enviado. Por favor, contacte a soporte inmediatamente para pedidos no enviados.",
                "Standard shipping typically takes 5-7 business days, while express shipping takes 2-3 business days.": "El envío estándar generalmente toma de 5 a 7 días hábiles, mientras que el envío express toma de 2 a 3 días hábiles.",
                "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.": "Aceptamos Visa, MasterCard, American Express, PayPal y Apple Pay."
            },
            "fr": {
                "Our return policy allows returns within 30 days of purchase with a valid receipt.": "Notre politique de retour permet les retours dans les 30 jours suivant l'achat avec un reçu valide.",
                "You can track your order using the tracking number provided in your shipping confirmation email.": "Vous pouvez suivre votre commande en utilisant le numéro de suivi fourni dans votre e-mail de confirmation d'expédition.",
                "Shipping addresses cannot be changed once an order has been shipped. Please contact support immediately for unshipped orders.": "Les adresses de livraison ne peuvent pas être modifiées une fois qu'une commande a été expédiée. Veuillez contacter le support immédiatement pour les commandes non expédiées.",
                "Standard shipping typically takes 5-7 business days, while express shipping takes 2-3 business days.": "L'expédition standard prend généralement 5 à 7 jours ouvrables, tandis que l'expédition express prend 2 à 3 jours ouvrables.",
                "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.": "Nous acceptons Visa, MasterCard, American Express, PayPal et Apple Pay."
            }
        }

    def translate_question(self, text: str, target_lang: str) -> str:
        # Simulate translation of a question
        if target_lang in self.translations and text in self.translations["en"]:
            return self.translations[target_lang][text]
        return f"[Translated to {target_lang}]: {text}" # Fallback

    def translate_answer(self, text: str, target_lang: str) -> str:
        # Simulate translation of an answer
        if target_lang in self.answers and text in self.answers["en"]:
            return self.answers[target_lang][text]
        return f"[Translated Answer to {target_lang}]: {text}" # Fallback


class KnowledgeBase:
    def __init__(self):
        self.faq_en = {
            "What is your return policy?": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
            "How do I track my order?": "You can track your order using the tracking number provided in your shipping confirmation email.",
            "Can I change my shipping address?": "Shipping addresses cannot be changed once an order has been shipped. Please contact support immediately for unshipped orders.",
            "How long does shipping take?": "Standard shipping typically takes 5-7 business days, while express shipping takes 2-3 business days.",
            "What payment methods do you accept?": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay."
        }

    def get_faq_topics(self):
        return list(self.faq_en.keys())

    def get_answer_en(self, question: str) -> str:
        return self.faq_en.get(question, "I'm sorry, I don't have an answer for that specific question in my knowledge base.")

# --- InCLT Prompting Logic ---

class InCLTCrosslingualPrompter:
    def __init__(self, knowledge_base: KnowledgeBase, translator: MockTranslator):
        self.kb = knowledge_base
        self.translator = translator
        self.source_lang = "en" # Source language of the knowledge base

    def create_icl_prompt(self, customer_query: str, target_lang: str, num_examples: int = 2) -> str:
        prompt_parts = [
            "You are a helpful customer support assistant. Answer the following question in the customer's language based on the provided examples and context.",
            "Below are examples of questions and answers provided in both the source language (English) of our knowledge base and the target language of the customer's query.",
            "Use these examples to improve your cross-lingual understanding and provide an accurate response.",
            "\n---\n"
        ]

        # Select random examples for in-context learning
        available_questions = self.kb.get_faq_topics()
        selected_questions = random.sample(available_questions, min(num_examples, len(available_questions)))

        for q_en in selected_questions:
            a_en = self.kb.get_answer_en(q_en)

            # Translate the question and answer to the target language
            q_target = self.translator.translate_question(q_en, target_lang)
            a_target = self.translator.translate_answer(a_en, target_lang)

            # Add both source and target language examples to the prompt
            prompt_parts.append(f"Example Question (English): {q_en}")
            prompt_parts.append(f"Example Answer (English): {a_en}")
            prompt_parts.append(f"Example Question ({target_lang.upper()}): {q_target}")
            prompt_parts.append(f"Example Answer ({target_lang.upper()}): {a_target}")
            prompt_parts.append("\n---\n")

        # Add the actual customer query
        prompt_parts.append(f"Customer Query ({target_lang.upper()}): {customer_query}")
        prompt_parts.append(f"Answer ({target_lang.upper()}):")

        return "\n".join(prompt_parts)

# --- Chatbot Application --- 

class MultilingualCustomerSupportChatbot:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.translator = MockTranslator()
        self.llm = MockLLM()
        self.prompter = InCLTCrosslingualPrompter(self.kb, self.translator)

    def get_response(self, customer_query: str, target_lang: str) -> str:
        # Step 1: Create the InCLT prompt using both source and target language examples
        icl_prompt = self.prompter.create_icl_prompt(customer_query, target_lang)

        print("\n--- Generated InCLT Prompt ---")
        print(icl_prompt)
        print("------------------------------\n")

        # Step 2: Send the prompt to the (mock) LLM
        llm_response = self.llm.generate_response(icl_prompt)

        return llm_response

# --- Main Execution / Demonstration ---
if __name__ == "__main__":
    chatbot = MultilingualCustomerSupportChatbot()

    print("\n--- Multilingual Customer Support Chatbot Demo ---")

    # Scenario 1: Spanish Customer Query
    spanish_query = "¿Cómo hago seguimiento de mi pedido?"
    print(f"\nCustomer (Spanish): {spanish_query}")
    response_es = chatbot.get_response(spanish_query, "es")
    print(f"Chatbot Response: {response_es}")

    print("\n===================================================")

    # Scenario 2: French Customer Query
    french_query = "Puis-je changer mon adresse de livraison ?"
    print(f"\nCustomer (French): {french_query}")
    response_fr = chatbot.get_response(french_query, "fr")
    print(f"Chatbot Response: {response_fr}")

    print("\n===================================================")

    # Scenario 3: English Customer Query
    english_query = "What is your return policy?"
    print(f"\nCustomer (English): {english_query}")
    response_en = chatbot.get_response(english_query, "en")
    print(f"Chatbot Response: {response_en}")

    print("\n--- Demo End ---")
