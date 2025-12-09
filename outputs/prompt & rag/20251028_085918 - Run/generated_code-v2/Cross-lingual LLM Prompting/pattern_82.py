import gradio as gr
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
import torch
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

class MockLLM:
    def invoke(self, prompt_text: str):
        print(f"--- Mock LLM received prompt ---\n{prompt_text}\n--- End Mock LLM Prompt ---")
        if "order status" in prompt_text.lower() and ("FR" in prompt_text or "ES" in prompt_text):
            return "Based on your query, if you are asking about order status, please provide your order number. (Simulated response for cross-lingual query)"
        elif "refund policy" in prompt_text.lower() and ("Politique de remboursement" in prompt_text or "Política de reembolso" in prompt_text):
            return "Our refund policy allows returns within 30 days of purchase with the original receipt. (Simulated response for cross-lingual query)"
        else:
            return "Thank you for your query. We are processing your request. (Simulated generic response)"

mock_llm = MockLLM()

icl_examples = [
    {
        "input_en": "What is the status of my order?",
        "input_fr": "Quel est le statut de ma commande ?",
        "output_en": "To check your order status, please provide your order ID.",
        "output_fr": "Pour vérifier le statut de votre commande, veuillez fournir votre numéro de commande.",
        "context_info": "Customer wants to know about order tracking. Both languages provide clear intent."
    },
    {
        "input_en": "How can I return an item?",
        "input_es": "¿Cómo puedo devolver un artículo?",
        "output_en": "You can return items within 30 days. Please visit our returns page for more details.",
        "output_es": "Puede devolver artículos dentro de los 30 días. Por favor, visite nuestra página de devoluciones para más detalles.",
        "context_info": "Customer needs information about product returns. Cross-lingual understanding is vital for accurate policy retrieval."
    },
    {
        "input_fr": "J'ai un problème avec le paiement.",
        "input_en": "I have an issue with the payment.",
        "output_fr": "Veuillez contacter notre support technique avec les détails de votre problème de paiement.",
        "output_en": "Please contact our technical support with details of your payment issue.",
        "context_info": "Customer has a payment problem. The LLM should understand the core issue regardless of the input language."
    }
]

example_template = """
User Query (English): {input_en}
User Query (French): {input_fr}
User Query (Spanish): {input_es}
Expected Response (English): {output_en}
Expected Response (French): {output_fr}
Expected Response (Spanish): {output_es}
Context for understanding: {context_info}
"""
example_prompt = PromptTemplate.from_template(example_template)

few_shot_prompt = FewShotPromptTemplate(
    examples=icl_examples,
    example_prompt=example_prompt,
    prefix="""You are a multilingual customer support assistant. Below are examples of customer queries in different languages and the appropriate cross-lingual responses. Pay close attention to how information across languages helps in generating the correct response.

""",
    suffix="""
Now, based on the examples above, answer the following customer query:

User Query: {user_query}
Detected Language: {detected_language}
Target Language for Response (Inferred): {target_response_language}
Assistant Response: """,
    input_variables=["user_query", "detected_language", "target_response_language"],
)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def get_chatbot_response(user_query: str):
    detected_lang = detect_language(user_query)

    target_response_lang = detected_lang if detected_lang != "en" and detected_lang != "unknown" else "en"

    full_prompt_text = few_shot_prompt.format(
        user_query=user_query,
        detected_language=detected_lang.upper(),
        target_response_language=target_response_lang.upper()
    )

    simulated_response = mock_llm.invoke(full_prompt_text)

    return f"**Detected Language:** {detected_lang.upper()}\n\n**Full Prompt sent to LLM:**\n```\n{full_prompt_text}\n```\n\n**Simulated Assistant Response:**\n{simulated_response}"

iface = gr.Interface(
    fn=get_chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here in English, French, or Spanish...", label="Your Query"),
    outputs=gr.Markdown(label="Chatbot Response"),
    title="Multilingual E-commerce Support Chatbot (InCLT Prompting Demo)",
    description="This chatbot demonstrates InCLT Crosslingual Transfer Prompting by leveraging examples in both source and target languages to improve understanding and response generation. The full prompt sent to the LLM is displayed for transparency."
)

if __name__ == "__main__":
    iface.launch()