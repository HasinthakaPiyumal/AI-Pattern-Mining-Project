from transformers import pipeline
import gradio as gr

# 1. Load the Language Model
# Using a smaller model like 'distilgpt2' for demonstration purposes.
# For better performance in a real-world scenario, a larger, instruction-tuned model would be used.
model_name = "distilgpt2"
generator = pipeline("text-generation", model=model_name)

# Predefined categories for classification
CATEGORIES = ["Billing Inquiry", "Technical Support", "Product Information", "General Question"]

# 2. Prompt Engineering Module
def create_classification_prompt(query):
    categories_str = ", ".join(CATEGORIES)
    prompt = f"Classify the following customer query into one of these categories: {categories_str}.\nQuery: {query}\nCategory: "
    return prompt

def create_response_generation_prompt(query, category):
    if category == "Billing Inquiry":
        instruction = "Provide a polite and informative initial response regarding a billing inquiry. Suggest checking their account or contacting the billing department."
    elif category == "Technical Support":
        instruction = "Provide a helpful initial response for technical support, suggesting basic troubleshooting steps or directing them to a support agent."
    elif category == "Product Information":
        instruction = "Give a concise and informative initial response providing product information. Suggest checking the product page or FAQ."
    else: # General Question
        instruction = "Provide a helpful and friendly initial response to a general question. Ask for more details if necessary."

    prompt = f"Instruction: {instruction}\nCustomer Query: {query}\nInitial Response: "
    return prompt

# 3. Core AI Logic
def customer_support_bot(user_query):
    # Zero-shot Classification
    classification_prompt = create_classification_prompt(user_query)
    # The LLM will complete the prompt, effectively classifying the query.
    # We limit max_new_tokens to get just the category name.
    # Using num_return_sequences=1 for simplicity
    classification_output = generator(classification_prompt, max_new_tokens=10, num_return_sequences=1, do_sample=False, truncation=True)
    
    # Extract the predicted category. This requires careful parsing as LLMs might generate extra text.
    # For distilgpt2, it's less direct than instruction-tuned models. We'll look for keywords.
    predicted_text = classification_output[0]['generated_text'][len(classification_prompt):].strip()
    
    classified_category = "General Question" # Default if no match
    for category in CATEGORIES:
        if category.lower() in predicted_text.lower():
            classified_category = category
            break

    # Zero-shot Response Generation
    response_prompt = create_response_generation_prompt(user_query, classified_category)
    response_output = generator(response_prompt, max_new_tokens=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, truncation=True)
    
    # Extract the generated response
    generated_response = response_output[0]['generated_text'][len(response_prompt):].strip()
    
    full_response = f"Detected Category: {classified_category}\n\nBot Response: {generated_response}"
    return full_response

# 4. Gradio Interface
iface = gr.Interface(
    fn=customer_support_bot,
    inputs=gr.Textbox(lines=5, placeholder="Enter your customer query here..."),
    outputs="text",
    title="AI-Powered Zero-Shot Customer Support Bot",
    description="This bot classifies customer queries and generates initial responses using Zero-Shot Prompting, without explicit examples."
)

iface.launch()