import gradio as gr
import numpy as np
import cv2
import torch

def perform_segmentation(image_np, prompt_text):
    h, w, _ = image_np.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if "tumor" in prompt_text.lower():
        # Simulate a circular tumor in the center
        center_x, center_y = w // 2, h // 2
        radius = min(h, w) // 5
        cv2.circle(mask, (center_x, center_y), radius, 1, -1)
    elif "lung" in prompt_text.lower():
        # Simulate a rectangular lung region
        start_x, start_y = w // 4, h // 4
        end_x, end_y = 3 * w // 4, 3 * h // 4
        cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 1, -1)
    else:
        # Default: a small central square if no specific prompt keyword
        square_size = min(h, w) // 8
        start_x, start_y = (w - square_size) // 2, (h - square_size) // 2
        cv2.rectangle(mask, (start_x, start_y), (start_x + square_size, start_y + square_size), 1, -1)

    # In a real scenario, this is where a PyTorch model would process image_np and prompt_text
    # to generate a more sophisticated mask.
    # For this conceptual example, we just return the dummy numpy mask.
    return mask

def overlay_mask(image_np, mask):
    # Convert mask to 3 channels for overlay
    mask_3_channel = np.stack([mask * 255, np.zeros_like(mask), np.zeros_like(mask)], axis=-1) # Red mask
    
    # Convert original image to uint8 if it's not already
    if image_np.dtype != np.uint8:
        image_np = cv2.convertScaleAbs(image_np)

    # Create an alpha channel for blending
    alpha = (mask * 0.5).astype(np.float32) # 50% transparency
    alpha = np.stack([alpha, alpha, alpha], axis=-1)

    # Blend the mask with the original image
    overlaid_image = (image_np * (1 - alpha) + mask_3_channel * alpha).astype(np.uint8)
    return overlaid_image

def process_image_and_prompt(image, prompt):
    if image is None:
        return None
    
    # Convert Gradio image (PIL or numpy array) to OpenCV format (numpy array)
    image_np = np.array(image)
    
    # Ensure image is in BGR format if it's RGB (Gradio provides RGB usually)
    if image_np.shape[-1] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    mask = perform_segmentation(image_np, prompt)
    overlaid_image = overlay_mask(image_np, mask)
    
    # Convert back to RGB for Gradio display
    overlaid_image_rgb = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
    return overlaid_image_rgb

# Gradio Interface
iface = gr.Interface(
    fn=process_image_and_prompt,
    inputs=[
        gr.Image(type="numpy", label="Upload Medical Image (e.g., X-ray, CT Scan)"),
        gr.Textbox(lines=1, label="Segmentation Prompt (e.g., 'segment tumor', 'highlight lung')")
    ],
    outputs=gr.Image(type="numpy", label="Segmented Image"),
    title="Medical Image Analysis Assistant (Prompt-driven Segmentation)",
    description="Upload a medical image and provide a text prompt to segment specific regions of interest. (Conceptual SAM-like model)"
)

iface.launch()