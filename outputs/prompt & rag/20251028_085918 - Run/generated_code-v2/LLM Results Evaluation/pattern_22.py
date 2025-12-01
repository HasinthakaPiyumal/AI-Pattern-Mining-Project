"""A mock LLM service for demonstrating response generation."""

import time
import random

def mock_llm_generate(prompt_text: str) -> str:
    """Generates a mock LLM response based on the prompt text.

    This function simulates an LLM by returning a predefined response
    with slight variations, based on keywords in the prompt.
    """
    print(f"[Mock LLM] Processing prompt: {prompt_text[:70]}...")
    time.sleep(0.5)  # Simulate LLM processing time

    if "technical issue" in prompt_text.lower() or "error" in prompt_text.lower():
        responses = [
            "I understand you're facing a technical issue. Please try restarting your device and checking your internet connection.",
            "It sounds like a technical problem. Could you provide more details about the error message you're seeing?",
            "For technical assistance, please describe the steps you've already taken."
        ]
    elif "billing" in prompt_text.lower() or "invoice" in prompt_text.lower():
        responses = [
            "Regarding your billing inquiry, please check your account's 'Billing History' section. If you still have questions, provide your invoice number.",
            "I can help with billing questions. Could you confirm your account ID and the specific charge you're inquiring about?",
            "For billing discrepancies, we recommend reviewing your recent statements. If the issue persists, our finance team can assist further."
        ]
    elif "product" in prompt_text.lower() or "feature" in prompt_text.lower():
        responses = [
            "Thank you for your interest in our product. What specific feature are you curious about?",
            "Our product offers many features. You can find detailed information in our knowledge base.",
            "To learn more about product features, please visit our website's 'Features' page."
        ]
    else:
        responses = [
            "Thank you for reaching out. How can I assist you further?",
            "I've received your query. Please let me know how I can help.",
            "We appreciate you contacting us. What specific assistance do you require today?"
        ]
    
    response = random.choice(responses)
    print(f"[Mock LLM] Generated response: {response[:70]}...")
    return response
