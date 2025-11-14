
import streamlit as st
from multimodal_processor import MultimodalProcessor
from reasoning_engine import ReasoningEngine
from output_generator import OutputGenerator

def main():
    st.title("Medical Diagnostic Assistant")
    st.write("Upload patient data (text and images) for diagnosis.")

    # Input for textual data
    patient_history = st.text_area("Patient History & Symptoms", height=150)
    lab_reports_text = st.text_area("Lab Reports (textual summary)", height=100)
    doctor_notes = st.text_area("Doctor's Notes", height=100)

    # Input for visual data
    uploaded_images = st.file_uploader("Upload Medical Images (X-rays, MRIs, etc.)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if st.button("Get Diagnosis"):
        if not (patient_history or lab_reports_text or doctor_notes or uploaded_images):
            st.warning("Please provide some patient data to proceed.")
            return

        st.info("Processing patient data and applying multimodal structured reasoning...")

        # 1. Initialize Components
        multimodal_processor = MultimodalProcessor()
        reasoning_engine = ReasoningEngine()
        output_generator = OutputGenerator()

        # 2. Process Multimodal Inputs
        text_data = {
            "patient_history": patient_history,
            "lab_reports_text": lab_reports_text,
            "doctor_notes": doctor_notes
        }
        processed_text_features, processed_image_features, intermediate_visuals = multimodal_processor.process_inputs(text_data, uploaded_images)

        # Display intermediate visual outputs if any (mock for now)
        if intermediate_visuals:
            st.subheader("Intermediate Visual Interpretations (Simulated)")
            for img_name, img_data in intermediate_visuals.items():
                st.image(img_data, caption=f"Simulated: {img_name}", use_column_width=True)

        # 3. Engage Reasoning Engine
        final_diagnosis, reasoning_steps, thought_graph_representation = reasoning_engine.reason(processed_text_features, processed_image_features)

        # 4. Generate and Display Output
        output_generator.display_results(final_diagnosis, reasoning_steps, thought_graph_representation)

if __name__ == "__main__":
    main()
