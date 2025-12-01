import streamlit as st
from transformers import pipeline

# Initialize the text generation pipeline
# For demonstration purposes, we'll use a simple text generation model.
# In a real application, you would use a more powerful and medical-domain-specific LLM.
generator = pipeline("text-generation", model="distilgpt2")

def get_diagnosis_and_reasoning(symptoms, medical_history, test_results):
    prompt = f"""Based on the following patient information, provide a medical diagnosis and explain your reasoning in a markdown table format. The table should have the following columns: 'Symptom/Finding', 'Medical Condition/Hypothesis', 'Reasoning/Evidence', 'Confidence Level', 'Next Steps/Further Investigation'.

Patient Symptoms: {symptoms}
Medical History: {medical_history}
Test Results: {test_results}

Diagnosis and Reasoning:
"""

    # Generate text from the LLM
    # The max_new_tokens is set to a reasonable number to allow for table generation.
    # adjust as needed based on the expected output length.
    # The actual output will depend heavily on the model's capabilities and training.
    response = generator(prompt, max_new_tokens=500, num_return_sequences=1)[0]["generated_text"]

    # Extracting the part after 