import gradio as gr
from PIL import Image
import io
import base64
import torch

# Conditional imports for diffusers and transformers
try:
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
    from transformers import BlipProcessor, BlipForConditionalGeneration
    AI_MODELS_AVAILABLE = True
except ImportError:
    print("Warning: 'diffusers' or 'transformers' libraries not found. AI features will be disabled.")
    print("Please install them with: pip install diffusers transformers accelerate torch")
    AI_MODELS_AVAILABLE = False

# --- Global Models (Load once for efficiency) ---
text_to_image_pipeline = None
img_to_img_pipeline = None
blip_processor = None
blip_model = None

if AI_MODELS_AVAILABLE and torch.cuda.is_available():
    device = "cuda"
elif AI_MODELS_AVAILABLE:
    device = "cpu"
    print("Warning: CUDA not available. Running AI models on CPU, which might be slow.")
else:
    device = "cpu" # Default if models are not available or libraries missing

def load_ai_models():
    global text_to_image_pipeline, img_to_img_pipeline, blip_processor, blip_model

    if not AI_MODELS_AVAILABLE:
        print("AI models not loaded due to missing libraries.")
        return

    print(f"Loading AI models to {device}...")

    # Text-to-Image Model (Stable Diffusion v1-5 for faster demo, can be SDXL)
    try:
        print("Loading Stable Diffusion Text-to-Image pipeline...")
        text_to_image_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print("Stable Diffusion Text-to-Image model loaded.")
    except Exception as e:
        print(f"Error loading Stable Diffusion Text-to-Image model: {e}")
        text_to_image_pipeline = None

    # Image-to-Image Model (Stable Diffusion v1-5 for faster demo)
    try:
        print("Loading Stable Diffusion Image-to-Image pipeline...")
        img_to_img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print("Stable Diffusion Image-to-Image model loaded.")
    except Exception as e:
        print(f"Error loading Stable Diffusion Image-to-Image model: {e}")
        img_to_img_pipeline = None

    # BLIP Image Captioning Model
    try:
        print("Loading BLIP Image Captioning model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", 
                                                                 torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        print("BLIP Image Captioning model loaded.")
    except Exception as e:
        print(f"Error loading BLIP model: {e}")
        blip_processor, blip_model = None, None

# --- AI Function Implementations (Simulating FastAPI endpoints) ---

def generate_product_image(prompt: str, negative_prompt: str = "") -> tuple[Image.Image, str]:
    """
    Generates a product image from a textual description using a diffusion model.
    In a full FastAPI setup, this would be an endpoint like /generate_product_image.
    """
    if not AI_MODELS_AVAILABLE or text_to_image_pipeline is None:
        return Image.new("RGB", (512, 512), color = 'red'), "AI models not loaded or available. Check console for errors."

    if not prompt.strip():
        return Image.new("RGB", (512, 512), color = 'red'), "Please provide a product description."

    print(f"Generating image for prompt: '{prompt}', negative: '{negative_prompt}'")
    try:
        image = text_to_image_pipeline(prompt, negative_prompt=negative_prompt).images[0]
        return image, "Image generated successfully."
    except Exception as e:
        return Image.new("RGB", (512, 512), color = 'red'), f"Error generating image: {e}"

def describe_image(image: Image.Image) -> str:
    """
    Converts an uploaded image to a textual description.
    In a full FastAPI setup, this would be an endpoint like /describe_image.
    """
    if not AI_MODELS_AVAILABLE or blip_processor is None or blip_model is None:
        return "AI models not loaded or available. Check console for errors."

    if image is None:
        return "Please upload an image."

    print("Describing image...")
    try:
        # Preprocess the image and generate caption
        inputs = blip_processor(image, "a photography of a ", return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error describing image: {e}"

def transform_image_style(input_image: Image.Image, transformation_prompt: str) -> tuple[Image.Image, str]:
    """
    Transforms the style of an input image based on a textual prompt.
    In a full FastAPI setup, this would be an endpoint like /transform_image_style.
    """
    if not AI_MODELS_AVAILABLE or img_to_img_pipeline is None:
        return Image.new("RGB", (512, 512), color = 'red'), "AI models not loaded or available. Check console for errors."

    if input_image is None or not transformation_prompt.strip():
        return Image.new("RGB", (512, 512), color = 'red'), "Please upload an image and provide a transformation prompt."

    print(f"Transforming image with prompt: '{transformation_prompt}'")
    try:
        # Resize input image to a common dimension for Stable Diffusion
        input_image = input_image.resize((512, 512), Image.LANCZOS)
        image = img_to_img_pipeline(prompt=transformation_prompt, image=input_image, strength=0.75, guidance_scale=7.5).images[0]
        return image, "Image transformed successfully."
    except Exception as e:
        return Image.new("RGB", (512, 512), color = 'red'), f"Error transforming image: {e}"

def perform_virtual_try_on(product_image: Image.Image, person_image: Image.Image) -> tuple[Image.Image, str]:
    """
    Performs a simplified virtual try-on by overlaying the product image onto a person's image.
    In a full FastAPI setup, this would be an endpoint like /virtual_try_on.
    """
    if product_image is None or person_image is None:
        return Image.new("RGB", (512, 512), color = 'red'), "Please upload both product and person images."

    print("Performing simplified virtual try-on...")
    try:
        person_width, person_height = person_image.size
        product_width, product_height = product_image.size

        # A very basic heuristic: scale product image to about 40% of person's width
        target_product_width = int(person_width * 0.4)
        # Maintain aspect ratio for height
        target_product_height = int(product_height * (target_product_width / product_width))

        if target_product_width == 0 or target_product_height == 0: # Avoid division by zero or invalid resize
             return Image.new("RGB", (512, 512), color = 'red'), "Cannot resize product image to 0 dimensions. Try different images."

        resized_product = product_image.resize((target_product_width, target_product_height), Image.LANCZOS)

        # Create a blank canvas the size of the person image
        combined_image = person_image.copy()

        # Simple placement: center horizontally, place near top 1/3 vertically
        x_offset = (person_width - target_product_width) // 2
        y_offset = int(person_height * 0.25) # Adjust for desired vertical placement

        # Ensure offsets are not negative
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

        # Paste the product image onto the person image.
        # If product image has alpha channel, use it for transparent overlay.
        if resized_product.mode == 'RGBA':
            combined_image.paste(resized_product, (x_offset, y_offset), resized_product)
        else:
            # If product is RGB, we can convert it to RGBA and apply a global alpha for blending
            # This is a basic form of blending. For complex scenarios, mask generation is needed.
            product_with_alpha = resized_product.convert('RGBA')
            # Example: 70% opacity
            alpha = Image.new('L', resized_product.size, 178) # 178 out of 255 is ~70%
            product_with_alpha.putalpha(alpha)
            combined_image.paste(product_with_alpha, (x_offset, y_offset), product_with_alpha)

        return combined_image, "Virtual try-on simulated successfully."
    except Exception as e:
        return Image.new("RGB", (512, 512), color = 'red'), f"Error during virtual try-on: {e}"

# --- Gradio Interface ---

if __name__ == "__main__":
    print("Starting application...")
    load_ai_models() # Load models when the script starts

    with gr.Blocks(title="Multimodal Product Customization & Virtual Try-on") as demo:
        gr.Markdown(
            """
            # AI-powered Product Customization and Virtual Try-on Platform
            Customize products and try them on virtually using multimodal prompts!
            **Note:** This demo directly calls AI functions. In a real-world scenario, 
            a FastAPI backend would expose these functionalities as REST endpoints,
            and this Gradio interface would interact with that backend via HTTP requests.
            Models are loaded to CUDA if available, otherwise CPU (can be slow).
            """
        )

        with gr.Tab("1. Product Generation (Text-to-Image)"):
            with gr.Row():
                with gr.Column():
                    text_prompt = gr.Textbox(label="Product Description (Positive Prompt)",
                                             placeholder="e.g., A stylish blue dress with floral patterns, elegant, high quality")
                    negative_text_prompt = gr.Textbox(label="Undesired Elements (Negative Prompt)",
                                                      placeholder="e.g., ugly, low resolution, blurry, poor quality, ruffles, polka dots")
                    generate_button = gr.Button("Generate Product Image")
                with gr.Column():
                    generated_product_output = gr.Image(label="Generated Product", type="pil")
                    text_gen_status = gr.Textbox(label="Status", interactive=False)
            generate_button.click(
                fn=generate_product_image,
                inputs=[text_prompt, negative_text_prompt],
                outputs=[generated_product_output, text_gen_status]
            )

        with gr.Tab("2. Image-to-Text Description"):
            with gr.Row():
                with gr.Column():
                    image_input_desc = gr.Image(label="Upload Image to Describe", type="pil")
                    describe_button = gr.Button("Describe Image")
                with gr.Column():
                    image_description_output = gr.Textbox(label="Image Description", interactive=False)
            describe_button.click(
                fn=describe_image,
                inputs=[image_input_desc],
                outputs=[image_description_output]
            )

        with gr.Tab("3. Image-to-Image Transformation"):
            with gr.Row():
                with gr.Column():
                    img_transform_input = gr.Image(label="Upload Product Image for Transformation", type="pil")
                    transform_prompt = gr.Textbox(label="Transformation Prompt",
                                                  placeholder="e.g., change long sleeves to short sleeves, make it silk, add geometric patterns")
                    transform_button = gr.Button("Transform Image")
                with gr.Column():
                    transformed_image_output = gr.Image(label="Transformed Product", type="pil")
                    img_trans_status = gr.Textbox(label="Status", interactive=False)
            transform_button.click(
                fn=transform_image_style,
                inputs=[img_transform_input, transform_prompt],
                outputs=[transformed_image_output, img_trans_status]
            )

        with gr.Tab("4. Virtual Try-on (Simplified)"):
            gr.Markdown("---")
            gr.Markdown("## Simplified Virtual Try-on")
            gr.Markdown("Upload a generated product image and a photo of a person to see a basic overlay.")
            with gr.Row():
                with gr.Column():
                    try_on_product_img = gr.Image(label="Product Image (e.g., from generation/transformation)", type="pil")
                    try_on_person_img = gr.Image(label="Photo of a Person (for try-on)", type="pil")
                    try_on_button = gr.Button("Perform Virtual Try-on")
                with gr.Column():
                    try_on_result_img = gr.Image(label="Virtual Try-on Result", type="pil")
                    try_on_status = gr.Textbox(label="Status", interactive=False)
            try_on_button.click(
                fn=perform_virtual_try_on,
                inputs=[try_on_product_img, try_on_person_img],
                outputs=[try_on_result_img, try_on_status]
            )

    demo.launch()