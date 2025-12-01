import streamlit as st
import re
import random
from typing import List, Dict, Any

class SimulatedLLM:
    def generate_response(self, query: str) -> str:
        responses = [
            f"Here is some information about '{query}': It is a complex topic requiring careful consideration.",
            f"Regarding '{query}', studies suggest various factors are at play. Consult a specialist for personalized advice.",
            f"Preliminary data on '{query}' indicates potential approaches, but more research is needed.",
            f"For '{query}', common knowledge includes: it's important to be aware of the implications.",
            f"Concerning '{query}', general guidelines often recommend a cautious approach and professional consultation."
        ]
        return random.choice(responses)

class MedicalKnowledgeBase:
    def __init__(self):
        self.facts = {
            "diabetes": "Diabetes is a chronic metabolic disease characterized by high blood sugar levels. Type 1 is autoimmune, Type 2 is insulin resistance. Management involves diet, exercise, and medication.",
            "hypertension": "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle changes and medication are common treatments.",
            "fever": "A fever is a temporary increase in your body temperature, often due to an illness. It's usually a sign that your body is fighting off an infection. Most fevers resolve on their own, but high fevers require medical attention.",
            "aspirin": "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain. It can also be used to prevent blood clots.",
            "penicillin": "Penicillin is an antibiotic used to treat bacterial infections. Allergic reactions are common and can be severe.",
            "covid-19": "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Symptoms range from mild to severe, and vaccination is highly recommended for prevention."
        }
        self.harmful_terms = [
            "ingest bleach", "self-diagnose cancer without tests", "ignore medical advice",
            "stop medication without doctor", "try untested home remedy for serious illness"
        ]

    def get_info(self, topic: str) -> str:
        topic_lower = topic.lower()
        for key, value in self.facts.items():
            if key in topic_lower:
                return value
        return ""

    def contains_harmful_advice(self, text: str) -> bool:
        return any(term in text.lower() for term in self.harmful_terms)

CONSTITUTION_PRINCIPLES = {
    "factuality": "All medical information must be accurate, up-to-date, and reference reliable scientific sources.",
    "harmlessness": "Never suggest treatments or advice that could be harmful or go against established medical guidelines. Always emphasize consulting a qualified healthcare professional.",
    "bias_reduction": "Avoid any discriminatory language or recommendations based on race, gender, age, socioeconomic status, etc.",
    "privacy_protection": "Do not solicit or store personal identifiable health information.",
    "clarity": "Information should be presented clearly, adapting complexity for both medical professionals and patients."
}

class FactualityChecker:
    def __init__(self, kb: MedicalKnowledgeBase):
        self.kb = kb

    def critique(self, response: str, query: str) -> Dict[str, Any]:
        info = self.kb.get_info(query)
        if info and info not in response:
            return {"principle": "Factuality", "pass": False, "reason": "Response lacks specific factual information from knowledge base.", "suggested_addition": info}
        return {"principle": "Factuality", "pass": True, "reason": "Response appears factually consistent or general enough."}

class HarmlessnessChecker:
    def __init__(self, kb: MedicalKnowledgeBase):
        self.kb = kb

    def critique(self, response: str) -> Dict[str, Any]:
        if self.kb.contains_harmful_advice(response):
            return {"principle": "Harmlessness", "pass": False, "reason": "Response contains potentially harmful advice."}
        if "consult a specialist" not in response.lower() and "medical professional" not in response.lower():
            return {"principle": "Harmlessness", "pass": False, "reason": "Response does not explicitly advise consulting a healthcare professional for medical advice."
}
        return {"principle": "Harmlessness", "pass": True, "reason": "Response seems harmless and advises professional consultation."}

class BiasDetector:
    def critique(self, response: str) -> Dict[str, Any]:
        biased_terms = ["only for men", "women should avoid", "elderly are always", "poor people cannot afford"]
        if any(term in response.lower() for term in biased_terms):
            return {"principle": "Bias Reduction", "pass": False, "reason": "Response contains potentially biased language."}
        return {"principle": "Bias Reduction", "pass": True, "reason": "Response appears unbiased."}

class PrivacyGuard:
    def critique(self, response: str, query: str) -> Dict[str, Any]:
        pii_patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"
        ]
        if any(re.search(pattern, response) for pattern in pii_patterns) or \
           any(re.search(pattern, query) for pattern in pii_patterns):
            return {"principle": "Privacy Protection", "pass": False, "reason": "Response or query contains potential PII."}
        return {"principle": "Privacy Protection", "pass": True, "reason": "No PII detected."}

class ClarityAssessor:
    def critique(self, response: str, user_type: str) -> Dict[str, Any]:
        word_count = len(response.split())
        if user_type == "patient" and word_count > 100:
            return {"principle": "Clarity", "pass": False, "reason": "Response might be too long or complex for a patient.", "suggested_action": "Simplify and shorten."}
        if user_type == "doctor" and word_count < 30:
            return {"principle": "Clarity", "pass": False, "reason": "Response might be too brief for a doctor.", "suggested_action": "Add more detail."}
        return {"principle": "Clarity", "pass": True, "reason": "Response clarity seems appropriate for the user type."}

class AICritic:
    def __init__(self, kb: MedicalKnowledgeBase):
        self.factuality_checker = FactualityChecker(kb)
        self.harmlessness_checker = HarmlessnessChecker(kb)
        self.bias_detector = BiasDetector()
        self.privacy_guard = PrivacyGuard()
        self.clarity_assessor = ClarityAssessor()

    def critique(self, response: str, query: str, user_type: str) -> List[Dict[str, Any]]:
        critiques = []
        critiques.append(self.factuality_checker.critique(response, query))
        critiques.append(self.harmlessness_checker.critique(response))
        critiques.append(self.bias_detector.critique(response))
        critiques.append(self.privacy_guard.critique(response, query))
        critiques.append(self.clarity_assessor.critique(response, user_type))
        return critiques

class AIRevisor:
    def revise(self, original_response: str, critiques: List[Dict[str, Any]]) -> str:
        revised_response = original_response
        for critique in critiques:
            if not critique["pass"]:
                principle = critique["principle"]
                reason = critique["reason"]
                if principle == "Factuality" and "suggested_addition" in critique:
                    if critique["suggested_addition"] not in revised_response:
                        revised_response += f" {critique['suggested_addition']}"
                elif principle == "Harmlessness":
                    if "consult a specialist" not in revised_response.lower():
                        revised_response += " Always consult a qualified healthcare professional for medical advice."
                elif principle == "Bias Reduction":
                    revised_response = revised_response.replace("only for men", "for all genders").replace("women should avoid", "everyone should consider")
                elif principle == "Privacy Protection":
                    pii_patterns = [
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
                        r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"
                    ]
                    for pattern in pii_patterns:
                        revised_response = re.sub(pattern, "[REDACTED_PII]", revised_response)
                elif principle == "Clarity" and "suggested_action" in critique:
                    if "Simplify" in critique["suggested_action"]:
                        words = revised_response.split()
                        revised_response = " ".join([w for w in words if len(w) < 10 or w.lower() in ["the", "a", "is", "of", "and"]]) + " (Simplified for clarity.)"
                    elif "Add more detail" in critique["suggested_action"]:
                        revised_response += " Additional details can be found in peer-reviewed medical journals."
        return revised_response

class AIFeedbackLoop:
    def record_feedback(self, original_response: str, critiques: List[Dict[str, Any]], revised_response: str):
        print("\n--- AI Feedback Recorded ---")
        print(f"Original: {original_response}")
        print(f"Critiques: {critiques}")
        print(f"Revised: {revised_response}")
        print("----------------------------\n")

def process_medical_query(query: str, user_type: str) -> Dict[str, Any]:
    llm_engine = SimulatedLLM()
    medical_kb = MedicalKnowledgeBase()
    ai_critic = AICritic(medical_kb)
    ai_revisor = AIRevisor()
    ai_feedback_loop = AIFeedbackLoop()

    initial_response = llm_engine.generate_response(query)

    critiques = ai_critic.critique(initial_response, query, user_type)

    revised_response = ai_revisor.revise(initial_response, critiques)

    ai_feedback_loop.record_feedback(initial_response, critiques, revised_response)

    return {
        "query": query,
        "user_type": user_type,
        "initial_response": initial_response,
        "critiques": critiques,
        "revised_response": revised_response
    }

st.title("Constitutional AI-powered Medical Information Assistant")
st.write("This assistant provides medical information aligned with ethical principles.")

user_query = st.text_area("Enter your medical query:")
user_type_option = st.selectbox("Are you a:", ("patient", "doctor"))

if st.button("Get Medical Information"):
    if user_query:
        with st.spinner("Processing your query..."):
            results = process_medical_query(user_query, user_type_option)

            st.subheader("Initial LLM Response:")
            st.info(results["initial_response"])

            st.subheader("Constitutional AI Critiques:")
            for critique in results["critiques"]:
                if critique["pass"]:
                    st.success(f"**{critique['principle']}:** PASSED - {critique['reason']}")
                else:
                    st.warning(f"**{critique['principle']}:** FAILED - {critique['reason']}")

            st.subheader("Revised (Constitutionally Aligned) Response:")
            st.success(results["revised_response"])
    else:
        st.warning("Please enter a medical query.")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class MedicalQueryRequest(BaseModel):
    query: str
    user_type: str = "patient"

fastapi_app = FastAPI(
    title="Constitutional AI Medical Assistant API",
    description="API for fetching constitutionally aligned medical information."
)

@fastapi_app.post("/medical_info")
async def get_medical_info_api(request: MedicalQueryRequest):
    try:
        results = process_medical_query(request.query, request.user_type)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))