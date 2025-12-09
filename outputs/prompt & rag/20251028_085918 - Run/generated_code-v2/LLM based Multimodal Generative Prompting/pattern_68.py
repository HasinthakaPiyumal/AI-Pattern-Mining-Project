import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import io

class PromptDrivenSegmentationModel:
    def __init__(self):
        # In a real application, this would load a pre-trained segmentation model
        # like SAM (Segment Anything Model) or a fine-tuned medical segmentation model
        # capable of understanding text prompts.
        pass

    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        """
        Simulates prompt-driven segmentation. In a real scenario, this would involve
        feeding the image and prompt to a sophisticated deep learning model.
        For this demonstration, it generates a simple mask based on a placeholder logic.
        """
        width, height = image.size
        mask = Image.new("L", (width, height), 0) # Black mask
        draw = ImageDraw.Draw(mask)

        # Simple placeholder logic based on prompt keywords
        prompt_lower = prompt.lower()
        if "tumor" in prompt_lower or "lesion" in prompt_lower or "abnormality" in prompt_lower:
            # Simulate a central elliptical or circular region
            center_x, center_y = width // 2, height // 2
            radius_x = int(width * 0.2)
            radius_y = int(height * 0.15)
            draw.ellipse((center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y), fill=255)
        elif "lung" in prompt_lower or "chest" in prompt_lower:
            # Simulate a larger, more general region for organs
            draw.rectangle((width * 0.1, height * 0.1, width * 0.9, height * 0.9), fill=255)
        elif "fracture" in prompt_lower or "bone" in prompt_lower:
            # Simulate a linear or small rectangular region
            draw.rectangle((width * 0.4, height * 0.45, width * 0.6, height * 0.55), fill=255)
        else:
            # Default: a small central circle if no specific keyword
            center_x, center_y = width // 2, height // 2
            draw.ellipse((center_x - 30, center_y - 30, center_x + 30, center_y + 30), fill=255)
        
        return mask


def segment_image_with_prompt(image_file, prompt: str):
    if image_file is None:
        return None, None

    # Convert bytes to PIL Image
    input_image = Image.open(io.BytesIO(image_file))
    
    model = PromptDrivenSegmentationModel()
    segmentation_mask = model.predict(input_image, prompt)

    # Create an RGBA version of the original image for overlay
    original_rgba = input_image.convert("RGBA")
    
    # Create a colored overlay from the mask
    # Use a semi-transparent red color for the segmented region
    red_overlay = Image.new("RGBA", original_rgba.size, (0, 0, 0, 0)) # Fully transparent
    draw_overlay = ImageDraw.Draw(red_overlay)
    
    # Iterate through the mask to apply the overlay color
    mask_np = np.array(segmentation_mask)
    for y in range(mask_np.shape[0]):
        for x in range(mask_np.shape[1]):
            if mask_np[y, x] > 0: # If it's part of the segmented region
                draw_overlay.point((x, y), fill=(255, 0, 0, 128)) # Red with 50% transparency

    # Composite the original image and the overlay
    segmented_image = Image.alpha_composite(original_rgba, red_overlay)

    return input_image, segmented_image.convert("RGB") # Return as RGB for consistent output


# Gradio Interface
iface = gr.Interface(
    fn=segment_image_with_prompt,
    inputs=[
        gr.File(type="bytes", label="Upload Medical Image (e.g., X-ray, MRI)"),
        gr.Textbox(label="Segmentation Prompt (e.g., 'segment the tumor', 'highlight the lung')")
    ],
    outputs=[
        gr.Image(type="pil", label="Original Image"),
        gr.Image(type="pil", label="Segmented Image")
    ],
    title="Medical Image Analysis with Prompt-Driven Segmentation (Simulated)",
    description="Upload a medical image and provide a text prompt to simulate segmentation of regions of interest. This is a conceptual demonstration; the segmentation logic is simplified."
)

if __name__ == "__main__":
    iface.launch()