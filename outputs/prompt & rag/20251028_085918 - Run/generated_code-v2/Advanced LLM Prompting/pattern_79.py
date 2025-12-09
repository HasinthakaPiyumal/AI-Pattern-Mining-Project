import streamlit as st
from transformers import pipeline

# Load the pre-trained distilgpt2 model
@st.cache_resource
def load_model():
    generator = pipeline('text-generation', model='distilgpt2')
    return generator

generator = load_model()

st.title("AI-Powered E-commerce Product Description Generator")
st.write("Generate engaging product descriptions with custom style and tone!")

# User Inputs
product_name = st.text_input("Product Name", "")
key_features = st.text_area("Key Features (comma-separated)", "")

styles = ["formal", "casual", "humorous", "luxurious", "concise", "descriptive"]
tones = ["informative", "persuasive", "excited", "calm", "witty", "professional"]
genres = ["bullet points", "paragraph", "short & punchy", "blog post excerpt"]

desired_style = st.selectbox("Desired Style", styles)
desired_tone = st.selectbox("Desired Tone", tones)
desired_genre = st.selectbox("Desired Genre", genres)

if st.button("Generate Description"):
    if product_name and key_features:
        # Construct the prompt using Style Prompting pattern
        prompt = f"Product: {product_name}\nKey Features: {key_features}\n\nGenerate a product description for '{product_name}' with the following characteristics:\n- Style: {desired_style}\n- Tone: {desired_tone}\n- Genre: {desired_genre}\n\nProduct Description:"

        with st.spinner("Generating description..."):
            # Generate text using the loaded model
            # max_new_tokens controls the length of the generated text
            # The generated_text will include the prompt, so we need to trim it.
            generated_output = generator(prompt, max_new_tokens=200, num_return_sequences=1)[0]['generated_text']

            # Extract only the generated description part by removing the prompt and potential leading/trailing whitespace
            # This is a simple heuristic; more robust parsing might be needed for complex prompts/models
            description_start_index = generated_output.find("Product Description:") + len("Product Description:")
            generated_description = generated_output[description_start_index:].strip()

            st.subheader("Generated Product Description:")
            st.write(generated_description)
    else:
        st.warning("Please fill in the Product Name and Key Features to generate a description.")