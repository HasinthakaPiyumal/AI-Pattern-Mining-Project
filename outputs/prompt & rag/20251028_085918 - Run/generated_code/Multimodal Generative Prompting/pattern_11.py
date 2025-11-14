
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import io
import base64

# --- Helper Functions (Mock AI Services) ---

def img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_img(base64_string):
    return Image.open(io.BytesIO(base64.b64decode(base64_string)))

def create_placeholder_image(text="Generated Image", size=(512, 512)):
    img = Image.new('RGB', size, color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((size[0]/4, size[1]/2), text, fill=(255,255,255))
    return img

def mock_generate_image(
    prompt: str,
    negative_prompt: str = None,
    input_image: Image.Image = None,
    pattern_image: Image.Image = None,
    mask_image: Image.Image = None
) -> Image.Image:
    """Mock function for image generation service."""
    st.spinner("Generating image...")
    output_text = f"Generated based on: '{prompt}'"
    if negative_prompt:
        output_text += f" (Negative: '{negative_prompt}')"
    if input_image and pattern_image:
        # Simulate pattern transfer
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        if pattern_image.mode != "RGB":
            pattern_image = pattern_image.convert("RGB")
        
        # Resize pattern to fit a portion of the input image
        pattern_resized = pattern_image.resize((input_image.width // 2, input_image.height // 2))
        
        # Create a new image to paste on, preserving original input_image for background
        combined_img = input_image.copy()
        
        # Simple overlay for demonstration
        # You'd use more advanced techniques like ControlNet, img2img with specific weights
        combined_img.paste(pattern_resized, (input_image.width // 4, input_image.height // 4))
        return combined_img

    elif input_image and mask_image:
        # Simulate segmentation-driven editing (e.g., color change)
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        if mask_image.mode != "L": # Ensure mask is grayscale
            mask_image = mask_image.convert("L")

        # Create a tinted version for the masked area
        tint_color = (100, 100, 255) # A blue tint
        tint_layer = Image.new('RGB', input_image.size, tint_color)
        
        # Composite using the mask
        result_img = Image.composite(tint_layer, input_image, mask_image)
        return result_img

    return create_placeholder_image(output_text)

def mock_caption_image(image: Image.Image) -> str:
    """Mock function for image captioning service."""
    st.spinner("Captioning image...")
    # Simulate basic analysis based on image properties
    width, height = image.size
    mode = image.mode
    if width > 500 and height > 500:
        return f"A detailed image ({width}x{height}, mode: {mode}) likely depicting a complex object or scene. Looks like a hand-drawn sketch with various lines and shapes."
    else:
        return f"A smaller image ({width}x{height}, mode: {mode}), possibly a simple design or icon. Appears to be a basic sketch."

def mock_segment_image(image: Image.Image, text_prompt: str = None) -> Image.Image:
    """Mock function for segmentation service, returning a simple mask."""
    st.spinner("Performing segmentation...")
    # For demonstration, create a simple mask (e.g., an ellipse in the center)
    mask = Image.new('L', image.size, 0) # Black background
    draw = ImageDraw.Draw(mask)
    
    # Simulate a rough segmentation for a central object
    cx, cy = image.width // 2, image.height // 2
    radius_x, radius_y = image.width // 4, image.height // 3
    draw.ellipse((cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y), fill=255)

    # Add some text to the mask to indicate it's a segmentation
    d = ImageDraw.Draw(mask)
    d.text((50, 50), f"Mask for: {text_prompt or 'object'}", fill=128)

    return mask

def mock_customize_3d(
    text_prompt: str,
    image_input: Image.Image = None,
    _3d_object_input: str = None, # Simulate a path to a 3D model file or a simple ID
    annotations: str = None # Simulate textual annotations like measurements
) -> str:
    """Mock function for 3D customization."""
    st.spinner("Customizing 3D model...")
    response = f"3D model customization initiated with: '{text_prompt}'."
    if image_input:
        response += " Image reference provided."
    if _3d_object_input:
        response += f" Base 3D model: {_3d_object_input}."
    if annotations:
        response += f" Annotations: {annotations}."
    response += " (Conceptual: This would generate/modify a 3D model and render an image.)"
    return response

def mock_try_on_3d(
    user_3d_scan: str = None, # Simulate a path to a user's 3D body scan
    product_3d_model: str = None, # Simulate a path to a product's 3D model
    annotations: str = None
) -> str:
    """Mock function for virtual 3D try-on."""
    st.spinner("Performing 3D virtual try-on...")
    response = "3D virtual try-on initiated."
    if user_3d_scan:
        response += f" User 3D scan: {user_3d_scan}."
    if product_3d_model:
        response += f" Product 3D model: {product_3d_model}."
    if annotations:
        response += f" Annotations: {annotations}."
    response += " (Conceptual: This would fit and render 3D models for try-on.)"
    return response

# --- Streamlit Frontend ---
st.set_page_config(layout="wide", page_title="Multimodal Product Customization")
st.title("🛍️ Intelligent Product Customization and Virtual Try-On Platform")
st.markdown("--- Generative AI with Advanced Multimodal Prompting --- ")


# --- 1. Image Customization (Text-to-Image with Negative Prompt) ---
st.header("1. Image Customization (Text-to-Image with Negative Prompt)")
st.write("Generate product designs with precise control using positive and negative prompts.")

text_prompt_t2i = st.text_input(
    "Enter your design idea (e.g., 'a red dragon flying on a t-shirt'):",
    "a sleek black sports car with neon blue stripes"
)
negative_prompt_t2i = st.text_input(
    "Enter elements to avoid (e.g., 'no fire, no aggressive teeth'):",
    "no spoilers, no sharp edges"
)

if st.button("Generate Design"):  # Added a unique key for the button
    if text_prompt_t2i:
        generated_image = mock_generate_image(text_prompt_t2i, negative_prompt_t2i)
        st.image(generated_image, caption="Generated Product Design", use_column_width=True)
    else:
        st.warning("Please enter a design idea.")

st.markdown("--- ")

# --- 2. Virtual Try-On (Image-to-Image with Pattern Transfer) ---
st.header("2. Virtual Try-On (Image-to-Image with Pattern Transfer)")
st.write("Upload your photo and a pattern to see how it looks on a virtual garment.")

col1, col2 = st.columns(2)

user_photo = col1.file_uploader("Upload your photo for try-on", type=["png", "jpg", "jpeg"], key="user_photo_uploader")
pattern_image = col2.file_uploader("Upload a pattern to apply", type=["png", "jpg", "jpeg"], key="pattern_uploader")

try_on_button = st.button("Perform Virtual Try-On")

if try_on_button:
    if user_photo and pattern_image:
        user_img = Image.open(user_photo)
        pattern_img = Image.open(pattern_image)
        st.subheader("Original Images:")
        st.image(user_img, caption="Your Photo", width=250)
        st.image(pattern_img, caption="Pattern to Apply", width=250)
        
        try_on_result = mock_generate_image(prompt="virtual try-on with pattern", input_image=user_img, pattern_image=pattern_img)
        st.subheader("Virtual Try-On Result:")
        st.image(try_on_result, caption="Your Photo with New Pattern", use_column_width=True)
    else:
        st.warning("Please upload both your photo and a pattern image.")

st.markdown("--- ")

# --- 3. Converting Visual Information into Textual Descriptions ---
st.header("3. Sketch-to-Text Description")
st.write("Upload a sketch or image to get a detailed textual description.")

sketch_image_uploader = st.file_uploader("Upload a sketch or image", type=["png", "jpg", "jpeg"], key="sketch_uploader")

if st.button("Generate Description"): # Added a unique key for the button
    if sketch_image_uploader:
        sketch_img = Image.open(sketch_image_uploader)
        st.image(sketch_img, caption="Uploaded Sketch", width=250)
        description = mock_caption_image(sketch_img)
        st.subheader("Generated Textual Description:")
        st.info(description)
    else:
        st.warning("Please upload a sketch or image.")

st.markdown("--- ")

# --- 4. Segmentation-driven Customization ---
st.header("4. Segmentation-driven Customization")
st.write("Upload a product image and specify a part to customize (e.g., change stitching color).")

product_image_seg = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"], key="product_seg_uploader")
seg_custom_prompt = st.text_input("What do you want to change (e.g., 'change stitching color to navy blue')?", "change the main body color to green")

if st.button("Apply Segmented Customization"): # Added a unique key for the button
    if product_image_seg and seg_custom_prompt:
        prod_img = Image.open(product_image_seg)
        st.subheader("Original Product Image:")
        st.image(prod_img, caption="Product", width=250)

        # Mock segmentation
        mask = mock_segment_image(prod_img, seg_custom_prompt)
        st.subheader("Generated Segmentation Mask (Conceptual):")
        st.image(mask, caption="Mask", width=250)

        # Apply customization using the mask
        customized_image = mock_generate_image(
            prompt=seg_custom_prompt,
            input_image=prod_img,
            mask_image=mask
        )
        st.subheader("Customized Product Image:")
        st.image(customized_image, caption="Customized", use_column_width=True)
    else:
        st.warning("Please upload a product image and provide a customization prompt.")

st.markdown("--- ")

# --- 5. 3D Customization and Virtual Try-On (Conceptual) ---
st.header("5. 3D Customization and Virtual Try-On (Conceptual)")
st.write("Simulate advanced 3D product customization and virtual try-on.")

st.subheader("5.1. 3D Product Customization")
text_prompt_3d = st.text_input("Describe the 3D product customization:", "create a ring with a sapphire gemstone and intricate filigree pattern")
image_ref_3d = st.file_uploader("Upload a reference image for 3D customization (e.g., pattern)", type=["png", "jpg", "jpeg"], key="3d_ref_uploader")

if st.button("Customize 3D Product"): # Added a unique key for the button
    img_3d_input = Image.open(image_ref_3d) if image_ref_3d else None
    result_3d_custom = mock_customize_3d(text_prompt_3d, image_input=img_3d_input, _3d_object_input="base_ring_model.obj")
    st.info(result_3d_custom)
    if img_3d_input:
        st.image(img_3d_input, caption="3D Reference Image", width=200)

st.subheader("5.2. 3D Virtual Try-On")
user_3d_scan_input = st.text_input("Simulate user 3D body scan file (e.g., 'user_wrist_scan.obj'):", "user_wrist_scan.obj")
product_3d_model_input = st.text_input("Simulate product 3D model file (e.g., 'watch_model.obj'):", "watch_model.obj")
annotations_3d_try_on = st.text_input("Annotations (e.g., 'wrist circumference: 18cm'):", "wrist circumference: 18cm, snug fit")

if st.button("Perform 3D Virtual Try-On"): # Added a unique key for the button
    result_3d_try_on = mock_try_on_3d(user_3d_scan_input, product_3d_model_input, annotations_3d_try_on)
    st.info(result_3d_try_on)
    st.image(create_placeholder_image("3D Try-On Render"), caption="Conceptual 3D Try-On Result", use_column_width=True)
