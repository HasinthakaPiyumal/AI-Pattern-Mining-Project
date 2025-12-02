import gradio as gr
import cv2
import numpy as np
from PIL import Image

def segment_image_with_prompt(image_np, prompt_text):
    height, width, _ = image_np.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    prompt_text_lower = prompt_text.lower()

    if "tumor" in prompt_text_lower or "lesion" in prompt_text_lower:
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        cv2.circle(mask, (center_x, center_y), radius, 1, -1)
    elif "liver" in prompt_text_lower:
        mask[height // 2:, width // 2:] = 1
    elif "ventricle" in prompt_text_lower or "heart" in prompt_text_lower:
        mask[:height // 2, :width // 2] = 1
    else:
        mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = 1

    return mask * 255

def process_segmentation(image, prompt):
    if image is None:
        return None, "Please upload an image."
    if not prompt:
        return None, "Please enter a segmentation prompt."

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    segmentation_mask = segment_image_with_prompt(image_np, prompt)

    red_mask = np.zeros_like(image_np)
    red_mask[:, :, 2] = segmentation_mask
    
    alpha = 0.4
    segmented_image = cv2.addWeighted(image_np, 1, red_mask, alpha, 0)
    
    segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(segmented_image_rgb), "Segmentation successful based on prompt: " + prompt

iface = gr.Interface(
    fn=process_segmentation,
    inputs=[
        gr.Image(type="pil", label="Upload Medical Image (MRI/CT)"),
        gr.Textbox(label="Segmentation Prompt", placeholder="e.g., segment the tumor in the liver, highlight the left ventricle, outline the lesion")
    ],
    outputs=[
        gr.Image(type="pil", label="Segmented Image"),
        gr.Textbox(label="Status")
    ],
    title="AI-powered Medical Image Segmentation Assistant",
    description="Upload a medical image and provide a text prompt to segment specific anatomical structures or anomalies. (Simplified demonstration of prompt-driven segmentation)"
)

if __name__ == "__main__":
    iface.launch()