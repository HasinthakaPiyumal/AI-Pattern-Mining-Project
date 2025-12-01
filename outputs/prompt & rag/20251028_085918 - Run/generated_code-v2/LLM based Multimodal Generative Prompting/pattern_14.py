import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import os

# --- Configuration ---
# Set your OpenAI API key
# It's recommended to load this from an environment variable for security
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY") 

# --- Image Processing and Captioning Module ---
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_image_caption(raw_image: Image.Image) -> str:
    inputs = processor(raw_image.convert("RGB"), return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# --- Large Language Model (LLM) Interaction Module ---
def get_llm_recommendations(prompt: str) -> str:
    try:
        response = openai.Completion.create(
            model="text-davinci-003",  # You can use gpt-3.5-turbo via ChatCompletion.create as well
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error communicating with LLM: {e}"

# --- Main Application Logic ---
def recommend_products(image: Image.Image, user_preferences: str) -> str:
    if image is None:
        return "Please upload an image." 

    # Generate image caption
    image_caption = generate_image_caption(image)

    # Prompt Engineering Module
    full_prompt = (
        f"I'm looking for product recommendations. Here's a description of an item "
        f"I'm interested in: '{image_caption}'.\n"
        f"My specific preferences are: '{user_preferences}'.\n"
        f"Based on this, please suggest similar e-commerce products, including their category and a brief reason why they are a good match. "
        f"Format your recommendations as a bulleted list.\n"
    )

    # Get LLM recommendations
    recommendations = get_llm_recommendations(full_prompt)

    return recommendations

# --- User Interface (UI) with Gradio ---
iface = gr.Interface(
    fn=recommend_products,
    inputs=[
        gr.Image(type="pil", label="Upload Product Image"),
        gr.Textbox(label="Your Product Preferences (e.g., 'under $100, men's section, durable')")
    ],
    outputs=gr.Textbox(label="Product Recommendations"),
    title="AI-Powered E-commerce Product Recommender",
    description=(
        "Upload an image of a product or style you like, and tell us your preferences. "
        "Our AI will generate a description from the image and use it with your text "
        "to find similar product recommendations." 
    )
)

iface.launch()