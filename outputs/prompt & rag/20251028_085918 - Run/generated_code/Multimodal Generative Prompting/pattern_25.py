import streamlit as st
from text_to_image_model import generate_garment_design
from virtual_tryon_model import perform_virtual_tryon
from image_to_text_model import describe_image_style
from _3d_generation_model import apply_3d_prompt
from segmentation_model import segment_image_for_tryon
import cv2
import numpy as np
from PIL import Image

st.title("Intelligent Product Customization and Virtual Try-On Platform")

st.header("1. Garment Design (Negative Prompting & Prompt Modifiers)")
positive_prompt = st.text_input("Describe your desired garment (e.g., 'a sleek minimalist dress, evening wear, high slit, royal blue, silk texture')", "a sleek minimalist dress")
negative_prompt = st.text_input("What to exclude? (e.g., 'no red, no polka dots, not too casual')", "no red, no polka dots")

if st.button("Generate 2D Garment Design"):
    if positive_prompt:
        with st.spinner("Generating 2D garment design..."):
            generated_image = generate_garment_design(positive_prompt, negative_prompt)
            if generated_image is not None:
                st.image(generated_image, caption="Generated 2D Garment Design", use_column_width=True)
                st.session_state['generated_garment'] = generated_image
            else:
                st.error("Failed to generate image. Please check your prompts.")

st.header("2. Image as Text Prompting")
st.write("Upload an inspiration image to get text descriptions for new designs.")
inspiration_image_upload = st.file_uploader("Upload Inspiration Image", type=["png", "jpg", "jpeg"], key="inspiration_image_upload")
if inspiration_image_upload is not None:
    inspiration_image = Image.open(inspiration_image_upload)
    st.image(inspiration_image, caption="Inspiration Image", use_column_width=True)
    if st.button("Describe Image Style"):
        with st.spinner("Analyzing image and generating text description..."):
            description = describe_image_style(inspiration_image)
            st.write(f"**Image Style Description:** {description}")
            st.session_state['image_style_description'] = description

st.header("3. Segmentation for Virtual Try-On Prep")
st.write("Upload your own photo for segmentation before virtual try-on.")
user_image_upload = st.file_uploader("Upload Your Photo", type=["png", "jpg", "jpeg"], key="user_image_upload")
if user_image_upload is not None:
    user_image_pil = Image.open(user_image_upload)
    user_image_np = np.array(user_image_pil)
    st.image(user_image_pil, caption="Your Uploaded Photo", use_column_width=True)

    if st.button("Perform Segmentation"):
        with st.spinner("Segmenting body and potential garment..."):
            segmented_image_pil, mask = segment_image_for_tryon(user_image_pil) # Assuming it returns PIL image and a mask
            if segmented_image_pil:
                st.image(segmented_image_pil, caption="Segmented Image (Body/Garment Delineation)", use_column_width=True)
                st.session_state['user_image_segmented'] = user_image_pil # Store original for try-on
                st.session_state['user_mask'] = mask # Store mask
            else:
                st.error("Segmentation failed.")

st.header("4. Virtual Try-On (PairedImage Prompting)")
st.write("Combine your segmented photo with a generated or uploaded garment for virtual try-on.")

tryon_garment_upload = st.file_uploader("Upload Garment for Try-On (Optional, uses generated if not uploaded)", type=["png", "jpg", "jpeg"], key="tryon_garment_upload")

if tryon_garment_upload is not None:
    tryon_garment_pil = Image.open(tryon_garment_upload)
    st.image(tryon_garment_pil, caption="Garment for Try-On", use_column_width=True)
    st.session_state['tryon_garment'] = tryon_garment_pil
elif 'generated_garment' in st.session_state:
    st.image(st.session_state['generated_garment'], caption="Using Previously Generated Garment", use_column_width=True)
    st.session_state['tryon_garment'] = st.session_state['generated_garment']

if st.button("Perform Virtual Try-On"):
    if 'user_image_segmented' in st.session_state and 'tryon_garment' in st.session_state:
        with st.spinner("Performing virtual try-on..."):
            tryon_result = perform_virtual_tryon(
                st.session_state['user_image_segmented'], 
                st.session_state['tryon_garment'], 
                st.session_state.get('user_mask')
            )
            if tryon_result is not None:
                st.image(tryon_result, caption="Virtual Try-On Result", use_column_width=True)
            else:
                st.error("Virtual try-on failed. Ensure both user image and garment are available.")
    else:
        st.warning("Please upload/generate a garment and upload/segment your photo first.")

st.header("5. 3D Product Exploration (3D Prompting Conceptual)")
st.write("Interact with a conceptual 3D model of the garment using text prompts.")
_3d_prompt = st.text_input("Describe 3D interaction (e.g., 'rotate 90 degrees, change fabric to denim, add a belt')", "show dress in 3D, rotate, change fabric to denim")

if st.button("Apply 3D Prompt"):
    if 'tryon_garment' in st.session_state or 'generated_garment' in st.session_state:
        with st.spinner("Applying 3D prompt (conceptual)..."):
            _3d_action_description = apply_3d_prompt(_3d_prompt)
            st.info(f"Conceptual 3D action applied: {_3d_action_description}")
            st.write("In a real application, this would update a 3D viewer with the garment reflecting these changes.")
    else:
        st.warning("Please generate or upload a garment first to interact in 3D.")
