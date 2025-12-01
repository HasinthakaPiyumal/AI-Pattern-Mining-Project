import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import json

# --- Configuration ---
IMAGE_DIR = "images"
ANNOTATIONS_FILE = "medical_annotations.json"

# Create directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Helper Functions ---
def load_annotations():
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_annotations(annotations):
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations, f, indent=4)

def save_uploaded_image(uploaded_file):
    file_path = os.path.join(IMAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def get_available_images():
    return [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Medical Image Annotator")
st.title("Human-in-the-Loop Medical Image Annotation")

# Load existing annotations
all_annotations = load_annotations()

# --- Sidebar for Image Management ---
st.sidebar.header("Image Management")

uploaded_file = st.sidebar.file_uploader("Upload a new medical image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    file_path = save_uploaded_image(uploaded_file)
    st.sidebar.success(f"Image '{uploaded_file.name}' uploaded successfully!")

available_images = get_available_images()
if not available_images:
    st.sidebar.info("No images available. Please upload one.")
    selected_image = None
else:
    selected_image_name = st.sidebar.selectbox("Select an image to annotate", available_images)
    selected_image_path = os.path.join(IMAGE_DIR, selected_image_name)
    selected_image = Image.open(selected_image_path)

# --- Main Annotation Area ---
if selected_image:
    st.subheader(f"Annotating: {selected_image_name}")

    # Display image and canvas
    st.write("Draw bounding boxes or masks on the image below.")
    
    # Ensure image is not too large for display within canvas
    max_width = 800
    max_height = 600
    img_width, img_height = selected_image.size

    if img_width > max_width or img_height > max_height:
        ratio = min(max_width / img_width, max_height / img_height)
        display_width = int(img_width * ratio)
        display_height = int(img_height * ratio)
    else:
        display_width = img_width
        display_height = img_height

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Orange semi-transparent
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=selected_image,
        height=display_height,
        width=display_width,
        drawing_mode="rect",  # Can also be "polygon", "point", "freedraw"
        key="canvas",
    )

    st.subheader("Annotation Details")
    
    current_image_annotations = all_annotations.get(selected_image_name, [])
    
    with st.form("annotation_form", clear_on_submit=True):
        st.write("Add details for your drawn annotations.")
        
        # If there are drawings, show them for selection
        if canvas_result and canvas_result.json_data and canvas_result.json_data["objects"]:
            st.info(f"{len(canvas_result.json_data['objects'])} drawing(s) detected. Please fill in details for each new drawing.")
            
            # For simplicity, we'll assume the user is annotating the *last* drawn object or manually linking later.
            # A more complex UI would let users select a specific drawing to annotate.
            # For now, we'll just capture the *current state* of all drawings on submit.
            drawn_objects_json = canvas_result.json_data["objects"]
        else:
            st.warning("No drawings detected on the canvas. Please draw on the image.")
            drawn_objects_json = []

        # Annotation Input Fields
        classification_label = st.text_input("Classification Label (e.g., 'Tumor', 'Fracture', 'Normal')")
        confidence_level = st.slider("Confidence Level", 0, 100, 75)
        potential_ambiguities = st.text_area("Potential Ambiguities/Challenges")
        differential_diagnoses = st.text_area("Differential Diagnoses (comma-separated)")
        notes = st.text_area("Additional Notes")

        submit_button = st.form_submit_button("Save Annotation")

        if submit_button and drawn_objects_json and classification_label:
            new_annotation = {
                "drawings": drawn_objects_json,
                "label": classification_label,
                "confidence": confidence_level,
                "ambiguities": potential_ambiguities,
                "differential_diagnoses": [dd.strip() for dd in differential_diagnoses.split(',') if dd.strip()],
                "notes": notes,
                "timestamp": str(st.session_state.get('last_annotation_time', '')) # Placeholder for actual time stamp if needed
            }
            
            # Append new annotations to the image's list
            current_image_annotations.append(new_annotation)
            all_annotations[selected_image_name] = current_image_annotations
            save_annotations(all_annotations)
            st.success("Annotation saved successfully!")
            st.session_state['last_annotation_time'] = st.session_state.get('last_annotation_time', 0) + 1 # Simple way to re-render for new annotations
            st.experimental_rerun()
        elif submit_button:
            st.error("Please draw on the image and provide a classification label to save.")

    st.subheader("Existing Annotations for this Image")
    if current_image_annotations:
        for i, annotation in enumerate(current_image_annotations):
            st.json(annotation)
    else:
        st.info("No annotations saved for this image yet.")
else:
    st.info("Please upload or select an image from the sidebar to begin annotating.")

# --- DICOM Support (Placeholder) ---
# To add DICOM support, you would integrate pydicom here.
# For example:
# try:
#     import pydicom
#     # Function to load DICOM and convert to PIL Image
# except ImportError:
#     st.sidebar.warning("pydicom not installed. DICOM files are not supported.")
