import streamlit as st
from PIL import Image
import os
import json
import numpy as np

# --- Configuration --- #
IMAGE_DIR = "./medical_images"
ANNOTATION_DIR = "./annotations"
PATHOLOGY_CATEGORIES = ["Tumor", "Cyst", "Inflammation", "Fracture", "Nodule", "Lesion", "Other"]
SEVERITY_LEVELS = ["Mild", "Moderate", "Severe", "Critical"]

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ANNOTATION_DIR, exist_ok=True)

# --- Helper Functions --- #
def load_images_from_dir(directory):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    image_files.sort()
    return image_files

def save_annotation(image_filename, annotation_data):
    annotation_filename = os.path.join(ANNOTATION_DIR, f"{os.path.splitext(image_filename)[0]}.json")
    with open(annotation_filename, 'w') as f:
        json.dump(annotation_data, f, indent=4)
    st.success(f"Annotation saved for {image_filename}!")

def load_annotation(image_filename):
    annotation_filename = os.path.join(ANNOTATION_DIR, f"{os.path.splitext(image_filename)[0]}.json")
    if os.path.exists(annotation_filename):
        with open(annotation_filename, 'r') as f:
            return json.load(f)
    return None

# --- Streamlit App --- #
st.set_page_config(layout="wide", page_title="Medical Image Annotator")
st.title("Medical Image Annotation and Diagnosis Support System")

# Initialize session state
if "image_files" not in st.session_state:
    st.session_state.image_files = load_images_from_dir(IMAGE_DIR)
    if not st.session_state.image_files:
        st.warning(f"No medical images found in '{IMAGE_DIR}'. Please add some images (e.g., .png, .jpg) to this directory.")
        st.stop()
    st.session_state.current_image_index = 0
    st.session_state.annotations = []

image_files = st.session_state.image_files
current_image_index = st.session_state.current_image_index

if image_files:
    current_image_filename = image_files[current_image_index]
    current_image_path = os.path.join(IMAGE_DIR, current_image_filename)

    # --- Sidebar for Navigation --- #
    st.sidebar.header("Navigation")
    if st.sidebar.button("Previous Image"):
        if current_image_index > 0:
            st.session_state.current_image_index -= 1
            st.experimental_rerun()
        else:
            st.sidebar.info("This is the first image.")

    st.sidebar.write(f"Image {current_image_index + 1} of {len(image_files)}")
    st.sidebar.selectbox(
        "Select Image",
        options=range(len(image_files)),
        format_func=lambda x: image_files[x],
        index=current_image_index,
        key="image_selector",
        on_change=lambda: setattr(st.session_state, "current_image_index", st.session_state.image_selector)
    )

    if st.sidebar.button("Next Image"):
        if current_image_index < len(image_files) - 1:
            st.session_state.current_image_index += 1
            st.experimental_rerun()
        else:
            st.sidebar.info("This is the last image.")

    st.sidebar.subheader("Current Image: ")
    st.sidebar.write(f"**{current_image_filename}**")

    # --- Main Content Area --- #
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Medical Image")
        try:
            image = Image.open(current_image_path)
            st.image(image, use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image {current_image_filename}: {e}")

        st.subheader("Simulated AI Diagnosis")
        st.info("AI suggests: Possible 'Nodule' in upper-left lobe with 78% confidence.")
        # Placeholder for AI segmentation overlay if drawing tools were integrated

    with col2:
        st.subheader("Annotation Tools")
        
        # Load existing annotation for the current image if available
        existing_annotation = load_annotation(current_image_filename)
        if existing_annotation:
            st.info("Existing annotation loaded.")
        
        with st.form("annotation_form", clear_on_submit=False):
            st.write("**Annotate Region of Interest**")
            pathology_category = st.selectbox(
                "Pathology Category",
                PATHOLOGY_CATEGORIES,
                index=PATHOLOGY_CATEGORIES.index(existing_annotation['pathology_category']) if existing_annotation and 'pathology_category' in existing_annotation else 0
            )
            
            # Simplified bounding box input
            st.write("Bounding Box (e.g., 10,20,100,150 for x1,y1,x2,y2)")
            bbox_input = st.text_input(
                "",
                value=existing_annotation['bbox'] if existing_annotation and 'bbox' in existing_annotation else ""
            )

            text_description = st.text_area(
                "Detailed Description",
                value=existing_annotation['description'] if existing_annotation and 'description' in existing_annotation else ""
            )
            confidence_score = st.slider(
                "Confidence Score (0-100%)",
                min_value=0, max_value=100, value=int(existing_annotation['confidence_score']) if existing_annotation and 'confidence_score' in existing_annotation else 75
            )
            
            st.write("**Auxiliary Annotations**")
            severity = st.radio(
                "Severity Level",
                SEVERITY_LEVELS,
                index=SEVERITY_LEVELS.index(existing_annotation['severity']) if existing_annotation and 'severity' in existing_annotation else 0,
                horizontal=True
            )
            differential_diagnosis = st.text_input(
                "Differential Diagnosis (optional)",
                value=existing_annotation['differential_diagnosis'] if existing_annotation and 'differential_diagnosis' in existing_annotation else ""
            )

            st.write("**AI Diagnosis Feedback**")
            ai_feedback = st.radio(
                "Do you agree with the AI's preliminary diagnosis?",
                ("Fully Agree", "Partially Agree (Needs Refinement)", "Disagree"),
                index=("Fully Agree", "Partially Agree (Needs Refinement)", "Disagree").index(existing_annotation['ai_feedback']) if existing_annotation and 'ai_feedback' in existing_annotation else 0,
                key=f"ai_feedback_{current_image_filename}"
            )

            if st.form_submit_button("Save Annotation"):
                try:
                    bbox_coords = [int(coord.strip()) for coord in bbox_input.split(',')] if bbox_input else []
                    if bbox_input and len(bbox_coords) != 4:
                        st.error("Bounding box input must be 4 comma-separated integers (x1,y1,x2,y2).")
                    else:
                        annotation_data = {
                            "image_filename": current_image_filename,
                            "pathology_category": pathology_category,
                            "bbox": bbox_input, # Store as string for simplicity, parse if needed
                            "description": text_description,
                            "confidence_score": confidence_score,
                            "severity": severity,
                            "differential_diagnosis": differential_diagnosis,
                            "ai_feedback": ai_feedback
                        }
                        save_annotation(current_image_filename, annotation_data)
                        # Reload annotations after saving
                        st.session_state.annotations = annotation_data # Update current session annotation
                        st.experimental_rerun() # Rerun to show updated state

                except ValueError:
                    st.error("Invalid bounding box format. Please enter comma-separated integers.")

else:
    st.info("Please add medical images to the 'medical_images' directory to start annotating.")
