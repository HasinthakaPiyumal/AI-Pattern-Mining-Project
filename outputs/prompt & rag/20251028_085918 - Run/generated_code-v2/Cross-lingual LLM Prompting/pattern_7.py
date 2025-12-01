
import json
from typing import List, Dict, Any

class InContextExample:
    def __init__(self, source_lang: str, pivot_lang: str, source_query: str, pivot_query: str, 
                 source_classification: str, pivot_classification: str, 
                 source_response: str, pivot_response: str):
        self.source_lang = source_lang
        self.pivot_lang = pivot_lang
        self.source_query = source_query
        self.pivot_query = pivot_query
        self.source_classification = source_classification
        self.pivot_classification = pivot_classification
        self.source_response = source_response
        self.pivot_response = pivot_response

    def to_prompt_format(self, current_source_lang: str) -> str:
        # Determine which language to display for the example based on current_source_lang
        # and ensure both source and pivot are always present in the example itself.
        
        example_str = f"""
### Example
Query ({self.source_lang}): {self.source_query}
Query ({self.pivot_lang}): {self.pivot_query}
Classification ({self.source_lang}): {self.source_classification}
Classification ({self.pivot_lang}): {self.pivot_classification}
Response ({self.source_lang}): {self.source_response}
Response ({self.pivot_lang}): {self.pivot_response}
"""
        return example_str

class MockTranslationService:
    def __init__(self):
        self.translations = {
            "es": {
                "shipping delays": "retrasos en el envío",
                "order status": "estado del pedido",
                "I would like to know the status of my order.": "Quisiera saber el estado de mi pedido.",
                "Order Tracking": "Seguimiento de Pedidos",
                "Your order is currently in transit and expected to arrive by [Date].": "Su pedido está actualmente en tránsito y se espera que llegue antes del [Fecha]."
            },
            "fr": {
                "shipping delays": "retards de livraison",
                "product refund": "remboursement du produit",
                "I need a refund for a faulty product.": "J'ai besoin d'un remboursement pour un produit défectueux.",
                "Product Refund": "Remboursement de Produit",
                "Please provide your order number for a refund request.": "Veuillez fournir votre numéro de commande pour une demande de remboursement."
            }
        }

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text
        
        # Simple mock translation, in a real scenario, use an actual MT model/API
        if source_lang == "en" and target_lang in self.translations:
            # Find key in target language values to get English key
            for en_text, translated_text in self.translations[target_lang].items():
                if translated_text.lower() == text.lower():
                    return en_text
        elif target_lang in self.translations and text.lower() in self.translations[target_lang]:
            return self.translations[target_lang][text.lower()]
        
        return f"[Translated from {source_lang} to {target_lang}: {text}]"

class InContextExampleManager:
    def __init__(self, examples: List[InContextExample]):
        self.examples = examples

    def get_relevant_examples(self, query: str, num_examples: int = 2) -> List[InContextExample]:
        # In a real system, this would involve semantic search (e.g., vector embeddings).
        # For this mock, we just return the first 'num_examples'.
        return self.examples[:num_examples]

class InCLTPromptGenerator:
    def __init__(self, translation_service: MockTranslationService, example_manager: InContextExampleManager, pivot_lang: str = "en"):
        self.translation_service = translation_service
        self.example_manager = example_manager
        self.pivot_lang = pivot_lang

    def generate_prompt(self, current_query: str, current_source_lang: str) -> str:
        instruction = f"""
You are a multilingual customer support assistant. Your task is to classify customer queries and suggest a response. 
Use the provided examples to understand the task across different languages. 
Each example shows a query, its classification, and a suitable response in both the original query language and English (as a pivot). 

Based on the following examples, classify the current query and provide a response in {current_source_lang}.

"""
        
        examples_str = ""
        relevant_examples = self.example_manager.get_relevant_examples(current_query)
        for example in relevant_examples:
            examples_str += example.to_prompt_format(current_source_lang)
            
        # The current query itself will be in its source language
        # We don't translate the current query for the LLM input, as the LLM is multilingual
        # The InCLT pattern is applied to the examples.

        prompt = f"""
{instruction}
{examples_str}
### Current Query
Query ({current_source_lang}): {current_query}

Classification ({current_source_lang}):
Response ({current_source_lang}):
"""
        return prompt

class MockMultilingualLLM:
    def __init__(self):
        self.responses = {
            "shipping delays": {"classification": "Shipping Inquiry", "response": "We are experiencing slight shipping delays. Please check your order status for the latest updates."}, # English default
            "retrasos en el envío": {"classification": "Consulta de Envío", "response": "Estamos experimentando ligeros retrasos en el envío. Por favor, consulte el estado de su pedido para las últimas actualizaciones."}, # Spanish
            "retards de livraison": {"classification": "Demande d'Expédition", "response": "Nous rencontrons de légers retards de livraison. Veuillez vérifier le statut de votre commande pour les dernières mises à jour."}, # French
            "order status": {"classification": "Order Tracking", "response": "Your order is currently in transit and expected to arrive by [Date]."},
            "estado del pedido": {"classification": "Seguimiento de Pedidos", "response": "Su pedido está actualmente en tránsito y se espera que llegue antes del [Fecha]."},
            "product refund": {"classification": "Product Refund", "response": "Please provide your order number for a refund request."},
            "remboursement du produit": {"classification": "Remboursement de Produit", "response": "Veuillez fournir votre numéro de commande pour une demande de remboursement."}
        }
    
    def generate_response(self, prompt: str) -> str:
        # Simulate LLM's understanding and generation based on keywords
        # This is a highly simplified mock. A real LLM would process the entire prompt.
        
        if "Query (es): Quisiera saber el estado de mi pedido." in prompt:
            return """Classification (es): Seguimiento de Pedidos
Response (es): Su pedido está actualmente en tránsito y se espera que llegue antes del [Fecha]."""
        elif "Query (fr): J'ai besoin d'un remboursement pour un produit défectueux." in prompt:
             return """Classification (fr): Remboursement de Produit
Response (fr): Veuillez fournir votre numéro de commande pour une demande de remboursement."""
        elif "Query (es): ¿Cuándo llegará mi paquete?" in prompt:
            return """Classification (es): Consulta de Envío
Response (es): Estamos experimentando ligeros retrasos en el envío. Por favor, consulte el estado de su pedido para las últimas actualizaciones."""

        return """Classification (en): General Inquiry
Response (en): Thank you for your inquiry. How can I assist you further?"""

class ResponseProcessor:
    def process_llm_output(self, llm_output: str, target_lang: str) -> Dict[str, str]:
        lines = llm_output.strip().split('\n')
        classification = "Unknown"
        response = "No response generated."
        
        for line in lines:
            if f"Classification ({target_lang}):" in line:
                classification = line.replace(f"Classification ({target_lang}):", "").strip()
            elif f"Response ({target_lang}):" in line:
                response = line.replace(f"Response ({target_lang}):", "").strip()
                
        return {"classification": classification, "response": response}

class MultilingualSupportAssistant:
    def __init__(self):
        self.translation_service = MockTranslationService()
        
        # Define some cross-lingual in-context examples
        # These examples explicitly contain both source and pivot language information.
        example1 = InContextExample(
            source_lang="es", pivot_lang="en",
            source_query="Quisiera saber el estado de mi pedido.",
            pivot_query="I would like to know the status of my order.",
            source_classification="Seguimiento de Pedidos",
            pivot_classification="Order Tracking",
            source_response="Su pedido está actualmente en tránsito y se espera que llegue antes del [Fecha].",
            pivot_response="Your order is currently in transit and expected to arrive by [Date]."
        )
        example2 = InContextExample(
            source_lang="fr", pivot_lang="en",
            source_query="J'ai besoin d'un remboursement pour un produit défectueux.",
            pivot_query="I need a refund for a faulty product.",
            source_classification="Remboursement de Produit",
            pivot_classification="Product Refund",
            source_response="Veuillez fournir votre numéro de commande pour une demande de remboursement.",
            pivot_response="Please provide your order number for a refund request."
        )
        
        self.example_manager = InContextExampleManager(examples=[example1, example2])
        self.prompt_generator = InCLTPromptGenerator(self.translation_service, self.example_manager, pivot_lang="en")
        self.llm = MockMultilingualLLM()
        self.response_processor = ResponseProcessor()

    def handle_inquiry(self, customer_query: str, source_lang: str) -> Dict[str, str]:
        print(f"\n--- Handling inquiry in {source_lang} ---")
        print(f"Customer Query: {customer_query}")

        # 1. Generate the InCLT prompt
        prompt = self.prompt_generator.generate_prompt(customer_query, source_lang)
        print(f"\n--- Generated Prompt ---\n{prompt}")

        # 2. Send prompt to LLM
        llm_raw_output = self.llm.generate_response(prompt)
        print(f"\n--- LLM Raw Output ---\n{llm_raw_output}")

        # 3. Process LLM output
        processed_response = self.response_processor.process_llm_output(llm_raw_output, source_lang)
        print(f"\n--- Processed Response ---")
        print(f"Classification: {processed_response['classification']}")
        print(f"Suggested Response: {processed_response['response']}")
        return processed_response

if __name__ == "__main__":
    assistant = MultilingualSupportAssistant()

    # Example 1: Spanish Inquiry
    assistant.handle_inquiry("¿Cuándo llegará mi paquete?", "es")
    
    # Example 2: French Inquiry
    assistant.handle_inquiry("J'ai besoin d'un remboursement pour un produit défectueux.", "fr")

    # Example 3: Spanish Inquiry (matching an example in content)
    assistant.handle_inquiry("Quisiera saber el estado de mi pedido.", "es")

    # Example 4: New Spanish query (not explicitly in mock LLM but similar to example 1)
    assistant.handle_inquiry("Tengo un problema con un retraso en mi envío.", "es")

    # Example 5: English Inquiry (pivot language, should still work)
    assistant.handle_inquiry("Where is my order?", "en")


