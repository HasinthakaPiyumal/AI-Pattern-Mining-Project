import gradio as gr
import torch
import faiss
import numpy as np
from typing import List, Dict, Optional

# Using sentence-transformers for embedding queries for Faiss retrieval
from sentence_transformers import SentenceTransformer


class Example:
    """Represents an in-context learning example, potentially with source and target language versions."""
    def __init__(self, id: int, source_query: str, source_answer: str, target_query: Optional[str] = None, target_answer: Optional[str] = None, language: str = "en"):
        self.id = id
        self.source_query = source_query
        self.source_answer = source_answer
        self.target_query = target_query
        self.target_answer = target_answer
        self.language = language # Target language for cross-lingual examples

    def to_prompt_string(self, query_language: str) -> str:
        """
        Formats the example as a string suitable for an LLM prompt.
        Prioritizes the target language if a cross-lingual example matches the query language.
        """
        if query_language == self.language and self.target_query and self.target_answer:
            # Present the example in the target language
            return (f"Question ({self.language}): {self.target_query}\n"
                    f"Answer ({self.language}): {self.target_answer}")
        else:
            # Otherwise, default to the source language (English in this setup)
            return (f"Question (en): {self.source_query}\n"
                    f"Answer (en): {self.source_answer}")


class MultilingualCustomerSupportChatbot:
    """
    A chatbot leveraging InCLT Crosslingual Transfer Prompting for multilingual customer support.
    It retrieves both source-only and cross-lingual in-context examples to boost LLM performance.
    """
    def __init__(self):
        # Initialize SentenceTransformer for generating embeddings for Faiss
        # 'all-MiniLM-L6-v2' is a good general-purpose multilingual sentence embedder.
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        # In-memory storage for Example objects and Faiss index
        self.examples: List[Example] = []
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.next_example_id = 0

        self._initialize_faiss()
        self._add_dummy_examples()

    def _initialize_faiss(self):
        """Initializes the Faiss index with the appropriate dimension."""
        self.faiss_index = faiss.IndexFlatL2(self.dimension)

    def _embed_text(self, text: str) -> np.ndarray:
        """Generates a numerical embedding for a given text."""
        # The SentenceTransformer encodes text into a fixed-size vector.
        return self.embedding_model.encode(text, convert_to_tensor=False).astype('float32').reshape(1, -1)

    def add_example(self, source_query: str, source_answer: str, target_query: Optional[str] = None, target_answer: Optional[str] = None, language: str = "en"):
        """
        Adds a new example to the knowledge base and indexes its relevant query in Faiss.
        If target_query is provided, it's considered a cross-lingual example.
        """
        example = Example(self.next_example_id, source_query, source_answer, target_query, target_answer, language)
        self.examples.append(example)

        # For retrieval, we embed the target query if available, otherwise the source query.
        # This allows matching queries in the target language.
        text_to_embed = target_query if target_query else source_query
        embedding = self._embed_text(text_to_embed)
        self.faiss_index.add(embedding)
        self.next_example_id += 1

    def _add_dummy_examples(self):
        """Populates the chatbot's knowledge base with predefined examples."""
        # Source-only (English) examples
        self.add_example("Where is my order?", "Your order is currently processing and expected to arrive within 2-3 business days.", language="en")
        self.add_example("How do I return an item?", "Please visit our 'Returns & Refunds' page on our website for detailed instructions.", language="en")
        self.add_example("What payment methods do you accept?", "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.", language="en")

        # Cross-lingual (English-Spanish) examples
        self.add_example(
            source_query="What is your refund policy?",
            source_answer="Our refund policy allows returns within 30 days of purchase with the original receipt.",
            target_query="¿Cuál es su política de reembolso?",
            target_answer="Nuestra política de reembolso permite devoluciones dentro de los 30 días posteriores a la compra con el recibo original.",
            language="es"
        )
        self.add_example(
            source_query="How can I track my package?",
            source_answer="You can track your package using the tracking number provided in your shipping confirmation email.",
            target_query="¿Cómo puedo rastrear mi paquete?",
            target_answer="Puede rastrear su paquete utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío.",
            language="es"
        )
        self.add_example(
            source_query="I received a damaged item, what should I do?",
            source_answer="We apologize for the inconvenience. Please contact customer support with your order details and photos of the damaged item for a replacement or refund.",
            target_query="Recibí un artículo dañado, ¿qué debo hacer?",
            target_answer="Lamentamos el inconveniente. Por favor, póngase en contacto con el servicio de atención al cliente con los detalles de su pedido y fotos del artículo dañado para un reemplazo o reembolso.",
            language="es"
        )
        self.add_example(
            source_query="Can I change my shipping address after placing an order?",
            source_answer="Unfortunately, we cannot change the shipping address once an order has been placed. Please double-check your address before confirming.",
            target_query="¿Puedo cambiar mi dirección de envío después de realizar un pedido?",
            target_answer="Lamentablemente, no podemos cambiar la dirección de envío una vez que se ha realizado un pedido. Por favor, verifique su dirección antes de confirmar.",
            language="es"
        )

    def retrieve_icl_examples(self, user_query: str, target_language: str, num_examples: int = 3) -> str:
        """
        Retrieves relevant in-context learning examples from the knowledge base.
        It prioritizes cross-lingual examples matching the target_language.
        """
        if self.faiss_index.ntotal == 0:
            return ""

        query_embedding = self._embed_text(user_query)
        # Retrieve more examples than needed to allow for filtering and prioritization
        distances, indices = self.faiss_index.search(query_embedding, num_examples * 3)

        retrieved_examples = [self.examples[idx] for idx in indices[0] if idx < len(self.examples)] # Ensure index is valid

        # Separate examples into source-only and cross-lingual for the target language
        source_only_examples = []
        cross_lingual_examples = []

        for ex in retrieved_examples:
            if ex.language == target_language and ex.target_query and ex.target_answer:
                cross_lingual_examples.append(ex)
            else:
                source_only_examples.append(ex)

        # Build the final set of selected examples, prioritizing cross-lingual
        selected_examples: List[Example] = []
        # Try to include at least one cross-lingual example if available
        if cross_lingual_examples and len(selected_examples) < num_examples:
            selected_examples.append(cross_lingual_examples.pop(0))

        # Fill the remaining slots with a mix, prioritizing cross-lingual if more are available
        while len(selected_examples) < num_examples:
            if cross_lingual_examples:
                selected_examples.append(cross_lingual_examples.pop(0))
            elif source_only_examples:
                selected_examples.append(source_only_examples.pop(0))
            else: # No more examples left to add
                break

        # Format the selected examples into a single string for the prompt
        icl_prompt_parts = [ex.to_prompt_string(target_language) for ex in selected_examples]
        return "\n\n".join(icl_prompt_parts)

    def generate_response(self, user_query: str, target_language: str = "es") -> str:
        """
        Generates a response to a user query in the specified target language
        by first retrieving cross-lingual in-context examples and then simulating an LLM response.
        """
        # Step 1: Retrieve relevant ICL examples using the InCLT Crosslingual Transfer Prompting pattern
        icl_examples_str = self.retrieve_icl_examples(user_query, target_language, num_examples=3)

        # Step 2: Construct the full prompt for the (hypothetical) multilingual LLM
        prompt = f"You are a helpful customer support assistant for an international e-commerce platform. "
        prompt += f"Your task is to provide accurate and concise answers in {target_language}. "
        prompt += f"Use the provided examples to understand complex queries and generate relevant responses.\n\n"

        if icl_examples_str:
            prompt += "--- Start of In-Context Learning Examples ---\n"
            prompt += icl_examples_str
            prompt += "\n--- End of In-Context Learning Examples ---\n\n"

        prompt += f"Customer Question ({target_language}): {user_query}\n"
        prompt += f"Assistant Answer ({target_language}):"

        # --- SIMULATED LLM RESPONSE ---
        # In a real-world application, this 'prompt' string would be sent to a powerful multilingual LLM
        # (e.g., a fine-tuned mBART, Llama 2, or a commercial LLM API like Google's PaLM/Gemini, OpenAI's GPT).
        # The LLM would then process the query along with the carefully constructed cross-lingual ICL examples
        # to generate a contextually relevant and accurate response in the target language.
        #
        # For this demonstration within the code generation tool, we simulate a plausible response
        # based on keywords from the user's query and the target language. This allows us to focus on

        # the ICL prompt generation logic, which is the core of the InCLT pattern.

        simulated_answer = ""
        user_query_lower = user_query.lower()

        if target_language == "es":
            if "pedido" in user_query_lower or "orden" in user_query_lower:
                simulated_answer = "Su pedido está en camino y esperamos que llegue pronto. Le enviaremos una actualización de seguimiento por correo electrónico."
            elif "devolver" in user_query_lower or "devolución" in user_query_lower:
                simulated_answer = "Para iniciar una devolución, visite la sección de 'Devoluciones' en nuestro sitio web. Le guiaremos a través del proceso."
            elif "reembolso" in user_query_lower:
                 simulated_answer = "Los reembolsos se procesan generalmente dentro de los 5-7 días hábiles después de que el artículo devuelto sea recibido y verificado."
            elif "dañado" in user_query_lower or "roto" in user_query_lower:
                simulated_answer = "Lamentamos el inconveniente. Por favor, póngase en contacto con nuestro soporte al cliente con los detalles de su pedido para una solución inmediata."
            elif "pago" in user_query_lower:
                 simulated_answer = "Aceptamos varias formas de pago, incluyendo tarjetas de crédito/débito y PayPal. Consulte nuestra página de pagos para más detalles."
            else:
                simulated_answer = f"Gracias por su consulta en español sobre '{user_query}'. Un agente de soporte revisará su pregunta y le responderá en breve."
        elif target_language == "en": # English is often the default/source language
            if "order" in user_query_lower or "shipment" in user_query_lower:
                simulated_answer = "Your order is on its way and expected to arrive soon. We will send you a tracking update via email."
            elif "return" in user_query_lower:
                simulated_answer = "To initiate a return, please visit the 'Returns' section on our website. We will guide you through the process."
            elif "refund" in user_query_lower:
                 simulated_answer = "Refunds are generally processed within 5-7 business days after the returned item is received and verified."
            elif "damaged" in user_query_lower or "broken" in user_query_lower:
                simulated_answer = "We apologize for the inconvenience. Please contact our customer support with your order details for an immediate resolution."
            elif "payment" in user_query_lower:
                 simulated_answer = "We accept various payment methods, including credit/debit cards and PayPal. Please check our payment page for more details."
            else:
                simulated_answer = f"Thank you for your inquiry in English about '{user_query}'. A support agent will review your question and respond shortly."
        else: # Generic fallback for other languages not explicitly handled
            simulated_answer = f"Thank you for your question in {target_language}. We are currently processing it and will get back to you soon."

        return simulated_answer

# Initialize the chatbot
chatbot = MultilingualCustomerSupportChatbot()

# Define the Gradio interface
def chat_interface(user_query: str, target_language: str) -> str:
    """Gradio wrapper function to interact with the chatbot."""
    return chatbot.generate_response(user_query, target_language)

# Create the Gradio Blocks interface for a more structured layout and clear output
with gr.Blocks() as demo:
    gr.Markdown("# Multilingual Customer Support Chatbot (InCLT Crosslingual Transfer Prompting)")
    gr.Markdown("This chatbot demonstrates the 'InCLT Crosslingual Transfer Prompting' pattern by dynamically generating in-context examples in both source and target languages to answer customer queries. The LLM response is simulated for demonstration.")

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Your Question", placeholder="E.g., ¿Cuál es mi política de reembolso? or Where is my order?")
            language_selector = gr.Radio(["es", "en"], label="Target Language", value="es")
            submit_btn = gr.Button("Get Answer")
        with gr.Column():
            output_text = gr.Textbox(label="Chatbot Response", interactive=False)

    submit_btn.click(
        fn=chat_interface,
        inputs=[user_input, language_selector],
        outputs=output_text
    )

    gr.Examples(
        examples=[
            ["¿Cuál es mi política de reembolso?", "es"],
            ["I received a damaged item, what should I do?", "en"],
            ["¿Cómo puedo rastrear mi paquete?", "es"],
            ["Where is my order?", "en"],
            ["¿Puedo cambiar mi dirección de envío después de realizar un pedido?", "es"]
        ],
        inputs=[user_input, language_selector]
    )

if __name__ == "__main__":
    demo.launch()