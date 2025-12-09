import base64
from PIL import Image, ImageDraw
import io

def generate_dummy_image(text, width=400, height=300):
    """Generates a dummy image with the given text."""
    img = Image.new('RGB', (width, height), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), text, fill=(255, 255, 0))
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def chain_of_images_process(product_idea: str) -> list[str]:
    """Simulates the Chain of Images process for product customization.

    Args:
        product_idea: The initial idea for the product (e.g., "cat lover t-shirt").

    Returns:
        A list of base64 encoded image strings representing the CoI steps.
    """
    print(f"Starting Chain of Images for: {product_idea}")
    
    images = []

    # Step 1: Generic product image
    prompt_step_1 = f"A plain t-shirt, simple design. Idea: {product_idea}"
    img_1_b64 = generate_dummy_image(f"Step 1: Generic Product\n({product_idea})")
    images.append(f"data:image/png;base64,{img_1_b64}")
    print(f"Generated image for Step 1: {prompt_step_1}")

    # Step 2: Add core design element
    core_element = "a cute cat silhouette"
    prompt_step_2 = f"A t-shirt with a {core_element} on it. Idea: {product_idea}"
    img_2_b64 = generate_dummy_image(f"Step 2: Add Core Element\n({core_element})")
    images.append(f"data:image/png;base64,{img_2_b64}")
    print(f"Generated image for Step 2: {prompt_step_2}")

    # Step 3: Refine with text/patterns
    refinement = "with 'Meow' text and paw print patterns"
    prompt_step_3 = f"A t-shirt with a cat silhouette, {refinement}. Idea: {product_idea}"
    img_3_b64 = generate_dummy_image(f"Step 3: Refine Design\n({refinement})")
    images.append(f"data:image/png;base64,{img_3_b64}")
    print(f"Generated image for Step 3: {prompt_step_3}")

    # Step 4: Contextualize (e.g., on a model, different color)
    context = "worn by a happy person, light blue t-shirt"
    prompt_step_4 = f"A {context} showing a t-shirt with a cat silhouette and 'Meow' text. Idea: {product_idea}"
    img_4_b64 = generate_dummy_image(f"Step 4: Contextualize\n({context})")
    images.append(f"data:image/png;base64,{img_4_b64}")
    print(f"Generated image for Step 4: {prompt_step_4}")

    # Step 5: Final high-fidelity render
    final_details = "high resolution, photorealistic, professional product shot"
    prompt_step_5 = f"A final customized product visualization of a t-shirt with a cat silhouette and 'Meow' text, {final_details}. Idea: {product_idea}"
    img_5_b64 = generate_dummy_image(f"Step 5: Final Render\n({final_details})")
    images.append(f"data:image/png;base64,{img_5_b64}")
    print(f"Generated image for Step 5: {prompt_step_5}")

    print("Chain of Images process completed.")
    return images
