import gradio as gr
import logging
import re
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Constitutional Principles Module ---
class ConstitutionalPrinciples:
    principles = [
        "Factuality: All medical information must be clinically accurate and evidence-based. Avoid speculative or unproven claims.",
        "Harmlessness: Never provide advice that could potentially harm the user or contradict professional medical guidance. Always recommend consulting a doctor for diagnosis and treatment.",
        "Non-discrimination/Bias: Ensure recommendations are not biased based on protected characteristics (e.g., race, gender, age, socioeconomic status).",
        "Privacy: Maintain strict confidentiality of user health data. Do not reveal or ask for personally identifiable information.",
        "Helpfulness: Provide clear, concise, and easy-to-understand information. Offer actionable, general health advice without diagnosing."
    ]

# --- 4. Data Management and Retrieval Mocks ---
# In a real system, these would interact with actual databases (Chroma, PostgreSQL, etc.)
def knowledge_base_lookup(query: str) -> str:
    logging.info(f"Simulating knowledge base lookup for: {query}")
    # Simulate retrieving relevant medical facts
    if "headache" in query.lower():
        return "Headaches are common and can be caused by stress, dehydration, or other factors. Severe or persistent headaches warrant medical attention."
    if "diabetes" in query.lower():
        return "Diabetes is a chronic condition that affects how your body turns food into energy. Management often involves diet, exercise, and medication. Regular monitoring is crucial."
    if "exercise" in query.lower():
        return "Regular physical activity is beneficial for overall health, including cardiovascular health, weight management, and mood improvement. Aim for at least 150 minutes of moderate-intensity activity per week."
    return "General health information: Always consult a healthcare professional for diagnosis and treatment."

def user_profile_lookup(user_id: str) -> Dict[str, Any]:
    logging.info(f"Simulating user profile lookup for user_id: {user_id}")
    # Simulate retrieving anonymized user health data
    if user_id == "user123": # Example user
        return {"age": 35, "gender": "female", "medical_history": ["hypertension"], "allergies": []}
    return {}

# --- 1. Core Language Models (LLMs) Mocks ---
class MockGeneratorLLM:
    def __init__(self, model_name: str = "Mock-Llama-2"):
        self.model_name = model_name
        logging.info(f"Initialized Generator LLM: {self.model_name}")

    def generate(self, prompt: str, context: str = "") -> str:
        logging.info(f"Generator LLM generating response for prompt: {prompt[:50]}...")
        # Simple placeholder generation logic
        if "headache" in prompt.lower():
            response = f"Based on your query about headaches and general health knowledge: {context} Rest, hydration, and over-the-counter pain relievers can often help. If headaches are severe or persistent, please consult a doctor."
        elif "diabetes" in prompt.lower():
            response = f"Regarding diabetes management: {context} A balanced diet, regular exercise, and adherence to prescribed medication are key. Always follow your doctor's advice."
        elif "exercise" in prompt.lower():
            response = f"For exercise recommendations: {context} Incorporating activities like walking, jogging, or swimming regularly can significantly improve your well-being. Start gradually and listen to your body."
        elif "personalized recommendation" in prompt.lower() and "hypertension" in context.lower():
            response = f"Considering your profile indicates hypertension: {context} Moderate aerobic exercise, a low-sodium diet, and stress reduction techniques are generally beneficial. It's vital to discuss any new exercise regimen or dietary changes with your physician due to your specific condition."
        else:
            response = f"Hello! I can provide general health information. {context} For any specific medical concerns, it is always best to consult with a qualified healthcare professional."
        return response

    def revise(self, original_response: str, critique: str) -> str:
        logging.info(f"Generator LLM revising response based on critique: {critique[:50]}...")
        # Simple placeholder revision logic
        if "Factuality" in critique or "Harmlessness" in critique:
            return original_response + "\n(Revision based on factuality/harmlessness critique: Always consult a medical professional for accurate diagnosis and personalized treatment plans.)"
        if "Non-discrimination" in critique:
            return re.sub(r"(male|female|age-group)", "individual", original_response, flags=re.IGNORECASE) + "\n(Revision: Removed potentially biased language.)"
        if "Privacy" in critique:
            # A more robust system would filter out sensitive info, here we just add a disclaimer
            return original_response + "\n(Revision: Ensured no personal identifying information is used or requested.)"
        if "Helpfulness" in critique:
            return original_response + "\n(Revision: Clarified information for better understanding.)"
        return original_response + "\n(Revision: Applied general feedback for improvement.)"


class MockCriticLLM:
    def __init__(self, model_name: str = "Mock-GPT-Critic"):
        self.model_name = model_name
        logging.info(f"Initialized Critic LLM: {self.model_name}")

    def evaluate_and_critique(self, response: str, principles: List[str], factual_context: str, user_profile: Dict[str, Any]) -> str:
        logging.info(f"Critic LLM evaluating response: {response[:50]}...")
        critique_feedback = []

        # Simulate Factuality check
        if ("cure all" in response.lower() or "guarantee results" in response.lower()) and "professional" not in response.lower():
            critique_feedback.append("Factuality: Response contains overly strong or unproven claims without sufficient caution or reference to professional advice.")

        # Simulate Harmlessness check
        if ("ignore medication" in response.lower() or "self-diagnose" in response.lower()):
            critique_feedback.append("Harmlessness: Response encourages potentially harmful actions or contradicts professional medical advice.")
        if "consult a doctor" not in response.lower() and any(word in response.lower() for word in ["diagnosis", "treatment"]):
             critique_feedback.append("Harmlessness: Response discusses diagnosis/treatment without sufficiently emphasizing the need to consult a medical professional.")

        # Simulate Non-discrimination/Bias check
        if re.search(r"(men over 50 should|women should only)", response.lower()):
            critique_feedback.append("Non-discrimination/Bias: Response appears to make generalized recommendations based on protected characteristics.")

        # Simulate Privacy check (simplified)
        if re.search(r"(your social security|your exact address)", response.lower()):
            critique_feedback.append("Privacy: Response attempts to extract or reveals sensitive personal identifying information.")
        if "medical_history" in user_profile and user_profile["medical_history"] and not f"your profile indicates {user_profile['medical_history'][0].lower()}" in response.lower():
             # If medical history is available but not used respectfully as context, or if it's overshared
             pass # This is a subtle check, for a mock we'll keep it simple: no overt asking

        # Simulate Helpfulness check (very basic)
        if len(response.split()) < 20:
            critique_feedback.append("Helpfulness: Response might be too brief or lack sufficient detail to be truly helpful.")

        return "\n".join(critique_feedback) if critique_feedback else ""

# --- 3. Iterative Self-Correction Loop & 6. Ethical Guardrails ---
class ConstitutionalAISystem:
    def __init__(self):
        self.generator_llm = MockGeneratorLLM()
        self.critic_llm = MockCriticLLM()
        self.constitutional_principles = ConstitutionalPrinciples.principles

    def _apply_safety_filters(self, text: str) -> str:
        logging.info("Applying safety filters...")
        # Example: Simple regex-based filter for inappropriate language
        # In a real system, guardrails-ai or similar would be used
        filtered_text = re.sub(r"(badword|offensivephrase)", "[FILTERED_CONTENT]", text, flags=re.IGNORECASE)
        return filtered_text

    def get_medical_recommendation(self, user_query: str, user_id: str = "default_user") -> str:
        logging.info(f"Processing user query: '{user_query}' for user_id: {user_id}")

        # Retrieve context
        factual_context = knowledge_base_lookup(user_query)
        user_profile = user_profile_lookup(user_id)
        profile_context = f"Your profile: {user_profile}. " if user_profile else ""

        initial_prompt = f"Provide medical information or a health recommendation based on the following query: '{user_query}'. {profile_context}Context from knowledge base: {factual_context}"
        current_response = self.generator_llm.generate(initial_prompt, context=profile_context + factual_context)

        max_iterations = 3
        iteration_history = [f"Initial Response:\n{current_response}"]

        for i in range(max_iterations):
            logging.info(f"Constitutional AI Loop - Iteration {i+1}")
            critique = self.critic_llm.evaluate_and_critique(current_response, self.constitutional_principles, factual_context, user_profile)

            if not critique:
                logging.info("Response passed constitutional review.")
                break
            else:
                logging.warning(f"Critique received:\n{critique}")
                revised_response = self.generator_llm.revise(current_response, critique)
                current_response = revised_response
                iteration_history.append(f"\n--- Revision {i+1} ---\nCritique: {critique}\nRevised Response:\n{current_response}")
            time.sleep(0.5) # Simulate processing time

        final_response = self._apply_safety_filters(current_response)
        logging.info(f"Final response after {i+1} iterations:\n{final_response}")

        full_output = "\n\n".join(iteration_history) + f"\n\n--- Final Ethically Aligned Response ---\n{final_response}"
        return full_output

# --- 5. User Interface (Gradio) ---
constitutional_ai_system = ConstitutionalAISystem()

def predict_recommendation(query: str) -> str:
    # In a real app, user_id would come from authenticated session
    return constitutional_ai_system.get_medical_recommendation(query, user_id="user123")

if __name__ == "__main__":
    logging.info("Starting Constitutional AI Medical System Gradio interface...")
    iface = gr.Interface(
        fn=predict_recommendation,
        inputs=gr.Textbox(lines=5, label="Enter your health-related query"),
        outputs=gr.Textbox(label="Ethically Aligned Medical Information / Recommendation", lines=20),
        title="Constitutional AI for Ethical Medical Information",
        description="This system provides general health information and recommendations, aligned with ethical principles (factuality, harmlessness, non-discrimination, privacy, helpfulness). Always consult a healthcare professional for diagnosis and treatment. This is a simulated environment."
    )
    iface.launch(share=False) # Set share=True to get a public link, but be cautious with medical info
