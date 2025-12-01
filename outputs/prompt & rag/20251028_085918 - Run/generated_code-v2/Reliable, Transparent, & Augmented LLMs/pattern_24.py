import gradio as gr
import random

class CoreAIDiagnosticModel:
    def diagnose(self, symptoms, lab_results, patient_history):
        # Simulate diagnostic outputs
        if "fever" in symptoms.lower() and "cough" in symptoms.lower():
            return {
                "diagnosis_probabilities": {"Common Cold": 0.6, "Influenza": 0.3, "Pneumonia": 0.1},
                "feature_importances": {"fever": 0.4, "cough": 0.3, "fatigue": 0.1, "lab_result_x": 0.05},
                "confidence_score": 0.75
            }
        elif "headache" in symptoms.lower() and "nausea" in symptoms.lower():
            return {
                "diagnosis_probabilities": {"Migraine": 0.7, "Tension Headache": 0.2, "Other": 0.1},
                "feature_importances": {"headache": 0.5, "nausea": 0.2, "light sensitivity": 0.1},
                "confidence_score": 0.80
            }
        else:
            return {
                "diagnosis_probabilities": {"Undetermined": 0.9, "General Malaise": 0.1},
                "feature_importances": {"symptoms_general": 0.6, "history_vague": 0.2},
                "confidence_score": 0.45
            }

class OODDetectionModule:
    def detect_ood(self, symptoms, lab_results, patient_history):
        # Simulate OOD detection
        if "rare genetic mutation" in patient_history.lower() or "unknown pathogen" in symptoms.lower():
            return True, "Input contains highly unusual medical terms not in typical training data."
        if random.random() < 0.15:  # 15% chance of being OOD randomly
            return True, "Data characteristics significantly deviate from common patterns in training dataset."
        return False, ""

class ReferenceManagementModule:
    def __init__(self):
        self.references = {
            "Common Cold": [
                "CDC: About Common Cold - https://www.cdc.gov/cold/index.html",
                "Mayo Clinic: Common Cold - https://www.mayoclinic.org/diseases-conditions/common-cold/symptoms-causes/syc-20351605"
            ],
            "Influenza": [
                "WHO: Influenza (Seasonal) - https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)",
                "CDC: Flu Symptoms & Complications - https://www.cdc.gov/flu/symptoms/symptoms.htm"
            ],
            "Migraine": [
                "NIH: Migraine Information - https://www.ninds.nih.gov/health-information/disorders/migraine",
                "American Migraine Foundation - https://americanmigrainefoundation.org/"
            ],
            "Undetermined": [
                "General Medical Diagnostics Guidelines (example.com/guidelines)"
            ]
        }

    def get_references(self, diagnosis_name):
        return self.references.get(diagnosis_name, [])

class MDATSystemLogic:
    def __init__(self):
        self.diagnostic_model = CoreAIDiagnosticModel()
        self.ood_detector = OODDetectionModule()
        self.reference_manager = ReferenceManagementModule()

    def process_diagnosis(self, symptoms, lab_results, patient_history):
        model_output = self.diagnostic_model.diagnose(symptoms, lab_results, patient_history)
        is_ood, ood_reason = self.ood_detector.detect_ood(symptoms, lab_results, patient_history)

        diagnosis_probabilities = model_output["diagnosis_probabilities"]
        feature_importances = model_output["feature_importances"]
        confidence_score = model_output["confidence_score"]

        output_md = []

        # OOD Warning
        if is_ood:
            output_md.append(f"<p style=\"color:red; font-weight:bold;\">🚨 **WARNING: Out-of-Distribution Case!** 🚨</p>")
            output_md.append(f"<p style=\"color:red;\">The provided case presents characteristics significantly different from our training data. The diagnosis should be interpreted with extreme caution. Reason: {ood_reason}</p>")
            output_md.append("<p>Model's confidence in this specific scenario may be lower than indicated due to data novelty.</p>")
        else:
            output_md.append("<p>This AI provides diagnostic support based on its training data. Always use clinical judgment.</p>")

        output_md.append("<h3><br>Diagnostic Probabilities:</h3>")
        sorted_diagnoses = sorted(diagnosis_probabilities.items(), key=lambda item: item[1], reverse=True)
        for diag, prob in sorted_diagnoses:
            color = "green" if prob > 0.6 else ("orange" if prob > 0.3 else "red")
            output_md.append(f"<p style=\"color:{color};\">• **{diag}**: {prob:.1%} likelihood</p>")

        output_md.append("<h3><br>Model Confidence:</h3>")
        confidence_color = "green" if confidence_score > 0.7 else ("orange" if confidence_score > 0.5 else "red")
        output_md.append(f"<p style=\"color:{confidence_color};\">Overall Model Confidence: **{confidence_score:.1%}**</p>")
        output_md.append("<p><i>Higher confidence indicates that the model found strong patterns matching its training data. Lower confidence suggests more ambiguity.</i></p>")

        output_md.append("<h3><br>Key Influencing Factors:</h3>")
        sorted_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
        if sorted_features:
            for feature, importance in sorted_features:
                output_md.append(f"<p>• **{feature.replace('_', ' ').title()}**: Contributed {importance:.1%} to the primary diagnosis.</p>")
        else:
            output_md.append("<p>No specific feature importances available for this case.</p>")

        # Get references for the primary diagnosis (highest probability)
        primary_diagnosis = sorted_diagnoses[0][0]
        references = self.reference_manager.get_references(primary_diagnosis)
        if references:
            output_md.append(f"<h3><br>Traceable References for {primary_diagnosis}:</h3>")
            for ref in references:
                output_md.append(f"<p>• {ref}</p>")
        else:
            output_md.append(f"<h3><br>No specific traceable references found for {primary_diagnosis}.</h3>")

        output_md.append("<h3><br>Important Considerations:</h3>")
        output_md.append("<ul>")
        output_md.append("<li><b>Limitations:</b> This AI is a supplementary tool and does not replace professional medical advice. It may struggle with rare conditions, incomplete data, or nuanced patient presentations outside its training distribution.</li>")
        output_md.append("<li><b>Verification:</b> Always cross-reference AI-generated information with clinical guidelines, up-to-date research, and your own expert judgment.</li>")
        output_md.append("<li><b>Probabilistic Nature:</b> Diagnoses are presented as probabilities, reflecting inherent uncertainty.</li>")
        output_md.append("</ul>")

        return "".join(output_md)

mda_t_system = MDATSystemLogic()

iface = gr.Interface(
    fn=mda_t_system.process_diagnosis,
    inputs=[
        gr.Textbox(lines=5, label="Symptoms (e.g., 'fever, cough, fatigue')"),
        gr.Textbox(lines=3, label="Lab Results (e.g., 'WBC: 12.0, CRP: 50')"),
        gr.Textbox(lines=5, label="Patient History (e.g., '45-year-old male, no significant past medical history')")
    ],
    outputs=gr.Markdown(label="Transparent Diagnostic Output"),
    title="Medical Diagnosis Assistant with Transparency (MDA-T)",
    description="Enter patient details to receive a diagnostic suggestion with transparency features to aid critical evaluation. Always use clinical judgment."
)

iface.launch()