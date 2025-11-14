import gradio as gr
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# --- Model Loading ---
# BLIP-2 for image captioning (Visual-to-Textual Conversion)
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
blip_model.to("cuda") # Move to GPU if available

# Stable Diffusion for Text-to-Image and Image-to-Image Generation
# Using a smaller, faster model for demonstration, consider larger models for production
sd_model_id = "runwayml/stable-diffusion-v1-5"
sd_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
sd_pipeline.to("cuda") # Move to GPU if available
sd_img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
sd_img2img_pipeline.to("cuda") # Move to GPU if available

def generate_design(text_prompt: str, input_image: Image.Image = None, negative_prompt: str = "" "", strength: float = 0.8) -> Image.Image:
    """
    Generates a fashion design based on text and an optional input image.
    """
    full_prompt = text_prompt

    if input_image is not None:
        # Convert visual information into textual descriptions
        # BLIP-2 expects an image and a question, or just an image for captioning
        # For simpler use, we'll caption it.
        pixel_values = blip_processor(images=input_image, return_tensors="pt").pixel_values.to("cuda", torch.float16)
        generated_ids = blip_model.generate(pixel_values=pixel_values, max_new_tokens=50)
        image_description = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        print(f"Generated image description: {image_description}")
        
        # Combine user prompt with image description for richer context
        full_prompt = f"{text_prompt}, {image_description}"

        # Image-to-image transformation
        generated_image = sd_img2img_pipeline(prompt=full_prompt, image=input_image, strength=strength, negative_prompt=negative_prompt).images[0]
    else:
        # Text-to-image generation
        generated_image = sd_pipeline(prompt=full_prompt, negative_prompt=negative_prompt).images[0]

    return generated_image

# --- Gradio Interface ---
iface = gr.Interface(
    fn=generate_design,
    inputs=[
        gr.Textbox(label="Design Description (e.g., 'a minimalist summer dress with a floral pattern')"),
        gr.Image(type="pil", label="Upload Mood Board or Existing Garment (Optional)"),
        gr.Textbox(label="Negative Prompt (e.g., 'no ruffles, avoid bold stripes')", value="", placeholder="Enter elements to avoid..."),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.05, label="Image Similarity Strength (for Image-to-Image, 0.1=more creative, 1.0=more similar)")
    ],
    outputs=gr.Image(type="pil", label="Generated Fashion Design"),
    title="AI-powered Fashion Design Assistant",
    description="Generate and modify garment designs using advanced multimodal prompting. Upload an image for inspiration or describe your design with text. Use negative prompts for precise control."
)

if __name__ == "__main__":
    iface.launch(share=False) # Set share=True to get a public link (careful with sensitive data)
