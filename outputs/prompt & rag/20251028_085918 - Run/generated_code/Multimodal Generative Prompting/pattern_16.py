import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

# Load BLIP for image-to-text
# Ensure you have an internet connection to download the model weights the first time.
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load Stable Diffusion for text-to-image
# For better quality, consider larger models like "runwayml/stable-diffusion-v1-5" or custom fine-tuned models.
# For first use, you may need to run `huggingface-cli login` in your terminal
# if the model requires authentication (e.g., stable-diffusion-v1-5).
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
if torch.cuda.is_available():
    pipe.to("cuda")

def generate_fashion_design(inspiration_image: Image.Image | None, positive_prompt: str, negative_prompt: str) -> Image.Image:
    """
    Generates a fashion design based on an inspiration image, positive prompt, and negative prompt.
    """
    generated_prompt = ""

    if inspiration_image:
        # Image-to-text conversion using BLIP
        inputs = blip_processor(inspiration_image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = blip_model.generate(**inputs)
        image_description = blip_processor.decode(out[0], skip_special_tokens=True)
        generated_prompt += f"{{image_description}}, "

    generated_prompt += positive_prompt

    print(f"Final Positive Prompt: {generated_prompt}")
    print(f"Negative Prompt: {negative_prompt}")

    # Text-to-image generation using Stable Diffusion
    # The negative_prompt parameter directly handles negative weighting.
    image = pipe(
        prompt=generated_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,  # Number of steps for generation (higher = better quality, slower)
        guidance_scale=7.5    # How much to rely on the prompt (higher = more adherence)
    ).images[0]

    return image

# Gradio Interface
iface = gr.Interface(
    fn=generate_fashion_design,
    inputs=[
        gr.Image(type="pil", label="Upload Inspiration Image (Optional)"),
        gr.Textbox(label="Positive Prompt (e.g., 'a elegant silk evening gown', 'bohemian style summer dress')"),
        gr.Textbox(label="Negative Prompt (e.g., 'no floral patterns', 'avoid dull colors', 'no wrinkles')")
    ],
    outputs=gr.Image(type="pil", label="Generated Fashion Design"),
    title="AI-Powered Fashion Design Assistant",
    description="Generate unique fashion designs using multimodal prompts. Upload an image for inspiration, describe your desired design, and specify elements to avoid."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()