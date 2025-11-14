
import tensorflow as tf
import numpy as np
import cv2
import gradio as gr
import shap
import matplotlib.pyplot as plt

# --- 1. Deep Learning Model (Placeholder) ---
# In a real application, you would load your fine-tuned model here.
# For demonstration, we create a dummy CNN model.
image_size = (224, 224)
num_classes = 5  # e.g., No DR, Mild, Moderate, Severe, Proliferative DR

def create_dummy_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    # The model would typically be trained. For SHAP, we need a compiled model.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_dummy_model()
# Load pre-trained weights if available, otherwise it's just a random model
# try:
#     model.load_weights("path/to/your/dr_model_weights.h5")
# except:
#     print("Dummy model created. No pre-trained weights loaded.")

# --- 2. Data Preprocessing ---
def preprocess_image(image):
    if image is None:
        return None
    # Convert Gradio image (PIL or numpy) to expected format (numpy array)
    if isinstance(image, str): # Handle file path input if needed (Gradio usually passes numpy array)
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, image_size)
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postprocess_prediction(predictions):
    class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    predicted_class_idx = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    prediction_str = f"Predicted Class: {predicted_class_name} (Confidence: {confidence:.2f}%)\n"
    prediction_str += "\nAll Class Probabilities:\n"
    for i, prob in enumerate(predictions[0]):
        prediction_str += f"  {class_names[i]}: {prob*100:.2f}%\n"
    return prediction_str

# --- 3. Local Interpretability Module ---
# SHAP Explanations
# For DeepExplainer, a background dataset is often used. For simplicity, we use a single background image.
# In a real scenario, this should be a representative sample of your training data.
explainer = shap.DeepExplainer(model, np.zeros((1, image_size[0], image_size[1], 3)))

def get_shap_explanation(preprocessed_image):
    if preprocessed_image is None:
        return None, ""
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(preprocessed_image)
    
    # SHAP values for image classification are usually per-class.
    # We'll visualize for the predicted class.
    predicted_class_idx = np.argmax(model.predict(preprocessed_image))
    shap_for_predicted_class = shap_values[predicted_class_idx][0]

    # Normalize SHAP values for visualization
    max_val = np.max(np.abs(shap_for_predicted_class))
    if max_val == 0:
        normalized_shap = np.zeros_like(shap_for_predicted_class)
    else:
        normalized_shap = shap_for_predicted_class / max_val
    
    # Overlay SHAP values on the original image
    original_image_display = (preprocessed_image[0] * 255).astype(np.uint8)
    
    # Create a heatmap from SHAP values
    cmap = plt.cm.RdBu # Red for positive, Blue for negative
    heatmap = cmap(normalized_shap[:,:,0]) # Use one channel for heatmap
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8) # Convert to RGB
    heatmap = cv2.resize(heatmap, (image_size[0], image_size[1]), interpolation=cv2.INTER_LINEAR)

    # Blend heatmap with original image
    # Alpha blend: result = alpha * heatmap + (1 - alpha) * image
    alpha = 0.5
    blended_image = cv2.addWeighted(original_image_display, 1 - alpha, heatmap, alpha, 0)
    
    return blended_image, "SHAP values highlight regions contributing positively (reddish) or negatively (bluish) to the predicted class." 

# Counterfactual Explanations (Conceptual)
def get_counterfactual_explanation(image):
    explanation = (
        "**Counterfactual Explanation (Conceptual):**\n\n"
        "Imagine a 'what-if' scenario. If certain features in this retinal image were slightly different, how would the diagnosis change?\n"
        "For instance, if the microaneurysms were less prominent, or hemorrhages smaller, the model might predict 'Mild DR' instead of 'Moderate DR'.\n"
        "Implementing this would involve an optimization process to find the minimal change to the input image that flips the model's prediction to a desired (e.g., lower severity) class, while ensuring the new image remains realistic. This is a complex research area."
    )
    return explanation

# --- 4. Global Interpretability Module (Conceptual) ---
def get_permutation_feature_importance():
    explanation = (
        "**Permutation Feature Importance (Conceptual):**\n\n"
        "Globally, which 'features' (e.g., presence of exudates, hemorrhages, neovascularization) are most important for the model's overall predictions across the entire dataset?\n"
        "Conceptually, we could randomly shuffle (permute) the values of a specific feature across the validation set and observe how much the model's accuracy drops. A large drop indicates high importance.\n"
        "For image data, this often involves abstracting features (e.g., regions detected by another model, handcrafted features) rather than individual pixels.\n"
        "*Example (hypothetical):* Permuting 'presence of severe hemorrhages' might lead to a significant drop in accuracy for detecting Severe DR, indicating its high importance."
    )
    return explanation

def get_pdp_ice_plots():
    explanation = (
        "**Partial Dependence Plots (PDP) / Individual Conditional Expectation (ICE) Plots (Conceptual):**\n\n"
        "How does the model's prediction change as a specific feature varies, while all other features are held constant?\n"
        "**PDP:** Shows the average effect of a feature on the prediction (e.g., as the 'density of microaneurysms' increases, the average predicted probability of 'Mild DR' might increase linearly, then plateau). This gives a global average view.\n"
        "**ICE:** Shows the effect for individual instances (e.g., for patient A, increasing 'optic disc blurring' makes the model more confident in 'Glaucoma' (if applicable), but for patient B, the effect is opposite due to other features). This reveals heterogeneous effects.\n"
        "For images, these are usually applied to high-level semantic features extracted from the image or latent representations, not raw pixels directly."
    )
    return explanation

# --- 5. Interactive Dashboard (Gradio) ---
def diagnose_and_explain(
    image_input, 
    selected_age_group, 
    selected_gender, 
    show_counterfactual,
    show_global_pfi,
    show_global_pdp_ice
):
    preprocessed_img = preprocess_image(image_input)
    
    if preprocessed_img is None:
        return "Please upload an image.", None, "", "", "", ""

    # --- Prediction ---
    predictions = model.predict(preprocessed_img)
    prediction_text = postprocess_prediction(predictions)
    
    # --- SHAP Explanation ---
    shap_image, shap_text = get_shap_explanation(preprocessed_img)

    # --- Counterfactual Explanation ---
    counterfactual_text = get_counterfactual_explanation(image_input) if show_counterfactual else ""
    
    # --- Global Insights ---
    global_pfi_text = get_permutation_feature_importance() if show_global_pfi else ""
    global_pdp_ice_text = get_pdp_ice_plots() if show_global_pdp_ice else ""

    # --- Bias Exploration (Conceptual) ---
    bias_exploration_text = f"\n**Bias Exploration (Conceptual for {selected_age_group}, {selected_gender}):**\n\n"
    bias_exploration_text += "In a real system, we would compare model performance and explanations across different demographic groups.\n"
    if selected_age_group == "65+" and selected_gender == "Female":
        bias_exploration_text += "*Hypothetical:* Model might show slightly lower accuracy for \'Severe DR\' in elderly female patients due to under-representation in training data, or different presentation of disease. Local explanations could reveal if it relies on different visual cues for this subgroup.\n"
    elif selected_age_group == "18-30" and selected_gender == "Male":
        bias_exploration_text += "*Hypothetical:* Model might be overconfident for \'No DR\' in young male patients, potentially missing subtle early signs if this demographic has atypical DR progression or features less prominent in the training data.\n"
    else:
        bias_exploration_text += "*Hypothetical:* For this demographic, the model generally performs well, but further investigation could involve checking specific cases where its certainty is unusually low or explanations are less coherent.\n"
    bias_exploration_text += "This interactive component would allow clinicians to probe for such disparities, identify sensitive subgroups, and guide model retraining or fairness interventions."

    return (
        prediction_text, 
        shap_image, 
        shap_text, 
        counterfactual_text, 
        global_pfi_text, 
        global_pdp_ice_text,
        bias_exploration_text
    )

# --- Gradio Interface Setup ---

# Input components
image_input_component = gr.Image(type="numpy", label="Upload Retinal Image")
age_group_dropdown = gr.Dropdown(
    ["0-17", "18-30", "31-45", "46-64", "65+"], label="Patient Age Group (Conceptual)", value="46-64"
)
gender_dropdown = gr.Dropdown(
    ["Male", "Female", "Other", "Prefer not to say"], label="Patient Gender (Conceptual)", value="Male"
)
counterfactual_checkbox = gr.Checkbox(label="Show Counterfactual Explanation (Conceptual)", value=False)
global_pfi_checkbox = gr.Checkbox(label="Show Permutation Feature Importance (Conceptual)", value=False)
global_pdp_ice_checkbox = gr.Checkbox(label="Show PDP/ICE Plots (Conceptual)", value=False)

# Output components
output_prediction = gr.Textbox(label="Model Prediction")
output_shap_image = gr.Image(type="numpy", label="SHAP Explanation (Regions Contributing to Prediction)")
output_shap_text = gr.Markdown(label="SHAP Explanation Details")
output_counterfactual = gr.Markdown(label="Counterfactual Explanation")
output_global_pfi = gr.Markdown(label="Global Permutation Feature Importance")
output_global_pdp_ice = gr.Markdown(label="Global PDP/ICE Plots")
output_bias_exploration = gr.Markdown(label="Bias Exploration for Selected Demographics")


interface = gr.Interface(
    fn=diagnose_and_explain,
    inputs=[
        image_input_component, 
        age_group_dropdown, 
        gender_dropdown, 
        counterfactual_checkbox, 
        global_pfi_checkbox, 
        global_pdp_ice_checkbox
    ],
    outputs=[
        output_prediction, 
        output_shap_image, 
        output_shap_text, 
        output_counterfactual, 
        output_global_pfi, 
        output_global_pdp_ice,
        output_bias_exploration
    ],
    title="AI-powered Diabetic Retinopathy Detection with Interpretability & Debugging",
    description=(
        "Upload a retinal image to get a DR diagnosis and various interpretability insights. "
        "This demonstration includes conceptual explanations for advanced interpretability methods and bias exploration." 
        "Note: The model is a dummy for demonstration purposes; actual diagnostic accuracy is not implied."
    ),
    live=False
)

if __name__ == "__main__":
    interface.launch()
