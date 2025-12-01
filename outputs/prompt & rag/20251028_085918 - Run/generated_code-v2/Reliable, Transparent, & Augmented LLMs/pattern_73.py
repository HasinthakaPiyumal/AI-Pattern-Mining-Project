import streamlit as st
from PIL import Image, ImageDraw
import json
import os

# --- Configuration --- #
IMAGES_DIR = "images"
ANNOTATIONS_FILE = "annotations.json"

# Ensure directories and files exist
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)
if not os.path.exists(ANNOTATIONS_FILE):
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump({}, f)

# --- Helper Functions --- #
def load_annotations():
    with open(ANNOTATIONS_FILE, "r") as f:
        return json.load(f)

def save_annotations(annotations):
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations, f, indent=4)

def get_image_files():
    return [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

def display_image_with_annotations(image_path, annotations_for_image=None):
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        if annotations_for_image:
            for ann in annotations_for_image:
                bbox = ann.get("bbox")
                label = ann.get("label")
                if bbox and len(bbox) == 4:
                    draw.rectangle(bbox, outline="red", width=2)
                    # Optionally draw label text
                    if label:
                        draw.text((bbox[0] + 5, bbox[1] + 5), label, fill="red")

        st.image(image, caption=os.path.basename(image_path), use_column_width=True)
        return image.width, image.height
    except Exception as e:
        st.error(f"Error loading or displaying image: {e}")
        return None, None

# --- Streamlit App --- #
st.set_page_config(layout="wide", page_title="Medical Image Annotation and Validation")
st.title("🩺 Medical Image Annotation and Validation Platform")

# Sidebar for image selection/upload
st.sidebar.header("Image Management")
image_files = get_image_files()

selected_image_name = st.sidebar.selectbox(
    "Select an image for annotation:",
    ["Upload New Image"] + image_files
)

uploaded_file = st.sidebar.file_uploader("Upload a new medical image", type=['png', 'jpg', 'jpeg'])

current_image_path = None
if selected_image_name == "Upload New Image" and uploaded_file is not None:
    file_path = os.path.join(IMAGES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded {uploaded_file.name}")
    current_image_path = file_path
    st.experimental_rerun()
elif selected_image_name != "Upload New Image":
    current_image_path = os.path.join(IMAGES_DIR, selected_image_name)


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Image View and Annotation")
    current_image_annotations = {} # Default empty
    if current_image_path and os.path.exists(current_image_path):
        all_annotations = load_annotations()
        current_image_annotations = all_annotations.get(os.path.basename(current_image_path), [])
        img_width, img_height = display_image_with_annotations(current_image_path, current_image_annotations)

    else:
        st.info("Please upload or select an image to begin annotation.")
        img_width, img_height = None, None

with col2:
    st.header("Annotation Tools")
    if current_image_path and img_width and img_height:
        with st.form("annotation_form", clear_on_submit=True):
            st.subheader("Add New Annotation")
            label = st.text_input("Label (e.g., 'Pneumonia', 'Fracture')", key="label_input")

            st.markdown(f"*Image Dimensions: {img_width}x{img_height}*", help="Enter bounding box coordinates relative to image size.")
            col_bbox1, col_bbox2 = st.columns(2)
            with col_bbox1:
                x1 = st.number_input("Bounding Box X1", min_value=0, max_value=img_width, value=0, key="x1_input")
                y1 = st.number_input("Bounding Box Y1", min_value=0, max_value=img_height, value=0, key="y1_input")
            with col_bbox2:
                x2 = st.number_input("Bounding Box X2", min_value=0, max_value=img_width, value=min(img_width, 100), key="x2_input")
                y2 = st.number_input("Bounding Box Y2", min_value=0, max_value=img_height, value=min(img_height, 100), key="y2_input")

            confidence = st.slider("Confidence Level", min_value=0.0, max_value=1.0, value=0.9, step=0.05, key="confidence_slider")
            notes = st.text_area("Clinical Notes / Description", key="notes_text_area")

            submit_annotation = st.form_submit_button("Save Annotation")

            if submit_annotation:
                if label and x1 < x2 and y1 < y2:
                    new_annotation = {
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "notes": notes
                    }
                    all_annotations = load_annotations()
                    image_key = os.path.basename(current_image_path)
                    if image_key not in all_annotations:
                        all_annotations[image_key] = []
                    all_annotations[image_key].append(new_annotation)
                    save_annotations(all_annotations)
                    st.success("Annotation saved successfully!")
                    st.experimental_rerun() # Rerun to update image with new annotation
                else:
                    st.error("Please provide a label and valid bounding box coordinates (X1 < X2, Y1 < Y2).")
    elif not current_image_path:
        st.info("Select or upload an image to enable annotation tools.")

    st.markdown("--- Other Annotations for this image ---")
    if current_image_annotations:
        for i, ann in enumerate(current_image_annotations):
            st.write(f"**Annotation {i+1}:**")
            st.json(ann)
    else:
        st.write("No annotations yet for this image.")


st.markdown("--- --- --- --- --- --- --- --- --- --- --- --- ")
st.header("Annotation Validation (Simplified)")
if current_image_path:
    st.subheader(f"Reviewing annotations for: {os.path.basename(current_image_path)}")
    st.write("Please compare the following two sets of annotations and indicate your preference.")

    # Dummy annotations for comparison
    ai_suggestion = {
        "label": "Pneumonia",
        "bbox": [50, 50, 200, 200],
        "confidence": 0.95,
        "notes": "AI identified diffuse opacity in the right lung."
    }
    human_correction = {
        "label": "Bacterial Pneumonia",
        "bbox": [45, 45, 210, 210],
        "confidence": 0.98,
        "notes": "Human corrected label and refined bounding box. More specific diagnosis."
    }

    col_ai, col_human = st.columns(2)

    with col_ai:
        st.subheader("AI Suggestion")
        st.json(ai_suggestion)

    with col_human:
        st.subheader("Human Correction")
        st.json(human_correction)

    preference = st.radio(
        "Which annotation set do you prefer?",
        ('AI Suggestion', 'Human Correction', 'Neither is perfect'),
        key="preference_radio"
    )

    if st.button("Submit Validation Feedback", key="submit_validation"):
        st.success(f"Thank you for your feedback! You preferred: {preference}")
        # In a real application, this feedback would be stored or sent to a model retraining pipeline.
else:
    st.info("Select an image to view the validation interface.")


# --- Instructions/Dummy Image --- #
if not image_files and uploaded_file is None:
    st.sidebar.markdown("**How to use:**")
    st.sidebar.markdown(
        "1. **Upload an image** using the file uploader."
        "2. **Select the uploaded image** from the dropdown."
        "3. **Annotate** using the tools on the right."
        "4. **Validate** existing annotations below."
    )
    st.info("No images found. Please upload a medical image to get started.")

    # Optionally create a dummy image for first-time users
    if not os.path.exists(os.path.join(IMAGES_DIR, "dummy_xray.png")):
        try:
            dummy_img = Image.new('RGB', (600, 400), color = 'white')
            d = ImageDraw.Draw(dummy_img)
            d.text((20,20), "Dummy X-Ray Image", fill=(0,0,0))
            d.ellipse((100, 100, 300, 300), fill=(200,200,200), outline="black")
            d.rectangle((350, 150, 550, 250), fill=(150,150,150), outline="black")
            d.text((150,180), "Lung Area", fill=(0,0,0))
            d.text((400,200), "Heart Shadow", fill=(0,0,0))
            dummy_img.save(os.path.join(IMAGES_DIR, "dummy_xray.png"))
            st.sidebar.success("A dummy 'dummy_xray.png' has been created for testing. Refresh the page or re-run to see it in the dropdown.")
            st.experimental_rerun()
        except Exception as e:
            st.warning(f"Could not create dummy image: {e}. Please upload an image manually.")
