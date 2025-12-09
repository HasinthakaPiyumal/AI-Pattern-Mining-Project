from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
import torch

def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def generate_image_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def create_llm_prompt(image_caption, product_name, product_category, key_features):
    prompt = f"Generate a compelling e-commerce product description for '{product_name}'.\n"
    prompt += f"Category: {product_category}\n"
    prompt += f"Key Features: {', '.join(key_features)}\n"
    prompt += f"Visual Description: {image_caption}\n\n"
    prompt += "Product Description:"
    return prompt


def generate_product_description(prompt, tokenizer, model, max_length=150, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt itself from the generated text
    return generated_text[len(prompt):].strip()


if __name__ == "__main__":
    # --- Configuration ---
    image_path = "./product_image.jpg"  # Replace with the actual path to your product image
    # Create a dummy image for demonstration if it doesn't exist
    try:
        Image.new('RGB', (60, 30), color = 'red').save(image_path)
        print(f"Dummy image created at {image_path}")
    except Exception as e:
        print(f"Could not create dummy image: {e}. Please ensure you have a 'product_image.jpg' in the current directory.")
        print("Exiting for now. Please create or provide a valid image path.")
        exit()

    # Simulate product metadata
    product_name = "Ergonomic Office Chair"
    product_category = "Furniture"
    key_features = [
        "Adjustable lumbar support",
        "Breathable mesh back",
        "3D armrests",
        "Smooth-rolling casters",
        "High-density foam seat"
    ]

    # --- Load Models ---
    print("Loading BLIP image captioning model...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("BLIP model loaded.")

    print("Loading GPT-2 language model...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Add a padding token if it's not present, often needed for generation
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    print("GPT-2 model loaded.")

    # --- Main Workflow ---
    print(f"\nProcessing image: {image_path}")
    try:
        product_image = load_image(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Please check the path.")
        exit()

    print("Generating image caption...")
    caption = generate_image_caption(product_image, blip_processor, blip_model)
    print(f"Generated Image Caption: {caption}")

    print("Creating LLM prompt...")
    llm_prompt = create_llm_prompt(caption, product_name, product_category, key_features)
    print(f"\nLLM Prompt:\n{llm_prompt}")

    print("Generating product description...")
    product_description = generate_product_description(llm_prompt, gpt2_tokenizer, gpt2_model)
    print("\n--- Generated Product Description ---")
    print(product_description)
    print("-------------------------------------")

    # Clean up dummy image
    import os
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Cleaned up dummy image at {image_path}")
