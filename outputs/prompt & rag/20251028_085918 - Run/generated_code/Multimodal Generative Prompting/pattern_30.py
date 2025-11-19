from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def image_to_text_description(
    image_path: str,
) -> str:
    """
    Converts an image into a detailed text description using a BLIP model.
    This demonstrates the ImageasText Prompting pattern for visual search and recommendations.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: A generated textual description of the image.
    """
    try:
        # Load image
        raw_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        return f"Error loading image {image_path}: {e}"

    # Load pre-trained BLIP model and processor
    # Using a smaller model for faster inference; a larger model might give better descriptions
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    except Exception as e:
        print(f"Error loading BLIP model or processor: {e}")
        return "Error: Could not load vision-language model."

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Preprocess image and generate caption
    inputs = processor(raw_image, return_tensors="pt").to(device)
    
    # Generate a more descriptive text by varying parameters or using a more advanced prompting strategy
    # For simple captioning, we can directly generate. For 'ImageasText Prompting', we are turning the image
    # *into* a textual prompt for other systems or for detailed search.
    out = model.generate(**inputs, max_new_tokens=50, min_length=10, num_beams=5, early_stopping=True)
    
    description = processor.decode(out[0], skip_special_tokens=True)
    print(f"Generated description for {image_path}: {description}")
    return description

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Create a dummy image for demonstration
    dummy_image = Image.new("RGB", (200, 200), color = 'green')
    from PIL import ImageDraw
    d = ImageDraw.Draw(dummy_image)
    d.ellipse([(50, 50), (150, 150)], fill="blue")
    d.text((10, 10), "Blue Circle on Green", fill=(255,255,255))
    dummy_image_path = "dummy_visual_search_item.png"
    dummy_image.save(dummy_image_path)

    print(f"Created dummy image: {dummy_image_path}")

    # Generate text description for the dummy image
    text_description = image_to_text_description(dummy_image_path)
    print(f"\nImage description: {text_description}")

    # In a real application, this text_description would be used for:
    # 1. Populating search queries for product discovery.
    # 2. Feeding into other generative models for design variations (e.g., 'design a shirt with {text_description}').
    # 3. Enhancing product recommendations based on visual similarity converted to text attributes.
