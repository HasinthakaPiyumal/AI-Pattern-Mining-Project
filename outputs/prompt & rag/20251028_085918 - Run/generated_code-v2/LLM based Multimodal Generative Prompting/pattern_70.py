import streamlit as st
from PIL import Image

# Placeholder for actual image generation
def generate_image(prompt):
    # In a real application, this would call a generative AI model (e.g., Stable Diffusion)
    # and return the generated image.
    # For this example, we return a simple dummy image.
    st.warning(f"Generating image with prompt: \"{prompt}\". (Using a dummy image for demonstration.)")
    return Image.new('RGB', (512, 512), color='red')

def generate_multimodal_content(base_prompt, modifiers):
    full_prompt_parts = [base_prompt]
    for category, value in modifiers.items():
        if value and value != "None":
            full_prompt_parts.append(f"{category} ({value})")
    
    full_prompt = ", ".join(full_prompt_parts)
    
    # For this MVP, we only handle image generation
    generated_content = generate_image(full_prompt)
    return generated_content, full_prompt

st.set_page_config(layout="centered", page_title="Prompt Modifiers Story Creator")
st.title("Social Media Story Creator with Prompt Modifiers")

st.markdown("Generate image posts and fine-tune their aesthetics using prompt modifiers.")

# User Input
base_prompt = st.text_area("Enter your base prompt here:", "A majestic lion roaring in the savanna")

# Prompt Modifiers
st.subheader("Fine-tune your content with modifiers:")

col1, col2 = st.columns(2)

with col1:
    medium = st.selectbox(
        "Medium",
        ["None", "watercolor painting", "oil on canvas", "digital art", "photography", "sketch"]
    )
    lighting = st.selectbox(
        "Lighting",
        ["None", "golden hour sunlight", "dramatic studio light", "moonlight", "soft natural light", "neon glow"]
    )

with col2:
    mood = st.selectbox(
        "Mood",
        ["None", "whimsical", "dreamy", "vibrant", "mysterious", "serene", "epic"]
    )
    style = st.selectbox(
        "Style",
        ["None", "impressionistic", "surrealistic", "cartoon", "realistic", "abstract", "cyberpunk"]
    )

modifiers = {
    "Medium": medium,
    "Lighting": lighting,
    "Mood": mood,
    "Style": style,
}

if st.button("Generate Content"):
    if base_prompt:
        with st.spinner("Generating your content..."):
            generated_content, full_prompt_used = generate_multimodal_content(base_prompt, modifiers)
            st.success("Content generated!")
            st.write(f"**Generated with prompt:** `{full_prompt_used}`")
            st.image(generated_content, caption="Your Generated Image", use_column_width=True)
    else:
        st.warning("Please enter a base prompt to generate content.")
