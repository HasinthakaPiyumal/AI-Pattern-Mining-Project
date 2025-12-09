import streamlit as st
import io
import base64
from PIL import Image
import numpy as np
import cv2
import pydicom
import nibabel as nib


def load_medical_image_backend(file_content: bytes, filename: str):
    try:
        if filename.lower().endswith(".dcm"):
            ds = pydicom.dcmread(io.BytesIO(file_content))
            image = ds.pixel_array
            image = (image - image.min()) / (image.max() - image.min()) * 255
            return Image.fromarray(image.astype(np.uint8))
        elif filename.lower().endswith((".nii", ".nii.gz")):
            img = nib.load(io.BytesIO(file_content))
            data = img.get_fdata()
            slice_idx = data.shape[2] // 2
            image_slice = data[:, :, slice_idx]
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
            return Image.fromarray(image_slice.astype(np.uint8))
        else:
            return Image.open(io.BytesIO(file_content))
    except Exception:
        return Image.open(io.BytesIO(file_content))

def simulate_segmentation_model_backend(image: Image.Image, prompt: str):
    np_image = np.array(image.convert("L"))
    height, width = np_image.shape

    mask = np.zeros((height, width), dtype=np.uint8)

    if "lung" in prompt.lower() or "nodule" in prompt.lower():
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    elif "liver" in prompt.lower() or "tumor" in prompt.lower():
        cv2.rectangle(mask, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), 255, -1)
    elif "bone" in prompt.lower() or "fracture" in prompt.lower():
        cv2.line(mask, (width // 4, height // 2), (3 * width // 4, height // 2 + 20), 255, 10)
    else:
        cv2.rectangle(mask, (width // 3, height // 3), (2 * width // 3, 2 * height // 3), 255, -1)

    return mask

def overlay_mask_on_image_backend(original_image: Image.Image, mask: np.ndarray):
    original_np = np.array(original_image.convert("RGB"))
    mask_colored = np.zeros_like(original_np)
    mask_colored[mask > 0] = [0, 255, 0]

    alpha = 0.5
    overlaid_image = cv2.addWeighted(original_np, 1 - alpha, mask_colored, alpha, 0)
    return Image.fromarray(overlaid_image)

def process_segmentation_request_backend(image_data: bytes, filename: str, prompt: str):
    try:
        original_image = load_medical_image_backend(image_data, filename)
        processed_image = original_image.convert("RGB")

        mask = simulate_segmentation_model_backend(processed_image, prompt)
        overlaid_image = overlay_mask_on_image_backend(processed_image, mask)

        buffer_original = io.BytesIO()
        original_image.save(buffer_original, format="PNG")
        original_b64 = base64.b64encode(buffer_original.getvalue()).decode("utf-8")

        buffer_overlaid = io.BytesIO()
        overlaid_image.save(buffer_overlaid, format="PNG")
        overlaid_b64 = base64.b64encode(buffer_overlaid.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "original_image_b64": original_b64,
            "segmented_image_b64": overlaid_b64,
            "message": "Segmentation processed successfully."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("Medical Image Segmentation with Prompting")

    st.sidebar.header("Upload Medical Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image (DICOM, NIfTI, PNG, JPG)", type=["dcm", "nii", "nii.gz", "png", "jpg", "jpeg"])

    st.sidebar.header("Segmentation Prompt")
    prompt_text = st.sidebar.text_area("Enter your segmentation prompt:", "e.g., segment the lung nodules")

    if uploaded_file is not None and prompt_text:
        st.info("Processing image...")
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        response = process_segmentation_request_backend(file_bytes, filename, prompt_text)

        if response["status"] == "success":
            st.success(response["message"])

            original_b64 = response["original_image_b64"]
            segmented_b64 = response["segmented_image_b64"]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                original_image_bytes = base64.b64decode(original_b64)
                st.image(original_image_bytes, use_column_width=True)

            with col2:
                st.subheader("Segmented Image")
                segmented_image_bytes = base64.b64decode(segmented_b64)
                st.image(segmented_image_bytes, use_column_width=True)
        else:
            st.error(f"Error: {response['message']}")
    elif uploaded_file is None:
        st.info("Please upload a medical image.")
    elif not prompt_text:
        st.info("Please enter a segmentation prompt.")

if __name__ == "__main__":
    run_streamlit_app()