import streamlit as st
from abc import ABC, abstractmethod
from typing import List

# 1. Constitutional Principles
CONSTITUTIONAL_PRINCIPLES = [
    "Factuality: All information provided must be evidence-based and medically accurate.",
    "Harmlessness: Avoid providing advice that could be dangerous or detrimental to the user's health.",
    "Non-discrimination: Ensure advice and information are inclusive and do not exhibit bias based on demographics, socio-economic status, or pre-existing conditions.",
    "Privacy Protection: Emphasize the importance of consulting a medical professional and avoid asking for or storing sensitive personal health information unless absolutely necessary and with explicit user consent.",
    "Helpfulness: Provide clear, concise, and easy-to-understand explanations and actionable advice."
]

# 2. Data Management: Medical Knowledge Base (Simplified)
class MedicalKnowledgeBase:
    def retrieve(self, query: str) -> str:
        if "diabetes" in query.lower():
            return "Diabetes is a chronic condition that affects how your body turns food into energy. It requires careful management of blood sugar levels, often through diet, exercise, and medication. Always consult a doctor for diagnosis and treatment plan."
        elif "headache" in query.lower():
            return "Headaches are common and usually not serious. They can be caused by stress, dehydration, or eye strain. Persistent or severe headaches should be evaluated by a medical professional."
        elif "fever" in query.lower():
            return "A fever is a temporary increase in your body temperature, often due to an illness. For adults, a fever is typically a temperature of 100.4 F (38 C) or higher. Rest and hydration are key, but consult a doctor if it's high or persistent."
        return "For any medical concern, it's crucial to consult a qualified healthcare professional. Generic information found online should not replace professional medical advice."

# 1. Core Components: Large Language Model (Abstract and Dummy Implementation)
class CoreLLM(ABC):
    @abstractmethod
    def generate_response(self, query: str, context: str) -> str:
        pass

class DummyCoreLLM(CoreLLM):
    def generate_response(self, query: str, context: str) -> str:
        st.session_state.initial_response_count += 1
        return f"Based on your query \"{query}\" and medical context: {context}. Here is some initial information, but remember to consult a professional for personalized advice. Always prioritize advice from a doctor."

# 2. AI Feedback Loop for Alignment: Critique Model (Abstract and Dummy Implementation)
class CritiqueModel(ABC):
    @abstractmethod
    def critique(self, response: str, principles: List[str]) -> List[str]:
        pass

class DummyCritiqueModel(CritiqueModel):
    def critique(self, response: str, principles: List[str]) -> List[str]:
        violations = []
        lower_response = response.lower()

        if "buy this" in lower_response or "take this specific brand" in lower_response:
            violations.append("Harmlessness: Potentially recommending specific products without professional context.")

        if "cure all" in lower_response or "guaranteed to work" in lower_response:
            violations.append("Factuality: Overstating efficacy or making unsubstantiated claims.")

        if "your symptoms indicate" in lower_response and "consult a doctor" not in lower_response:
            violations.append("Privacy Protection: Directly diagnosing without proper disclaimers.")
        
        if "only for young people" in lower_response or "not for certain groups" in lower_response and "medical reason" not in lower_response:
            violations.append("Non-discrimination: Potential for biased advice.")

        if "complicated medical jargon" in lower_response and "easy to understand" not in lower_response:
             violations.append("Helpfulness: Response is not clear or easy to understand.")

        return violations

# 2. AI Feedback Loop for Alignment: Revision Model (Abstract and Dummy Implementation)
class RevisionModel(ABC):
    @abstractmethod
    def revise(self, original_response: str, critiques: List[str]) -> str:
        pass

class DummyRevisionModel(RevisionModel):
    def revise(self, original_response: str, critiques: List[str]) -> str:
        revised_response = original_response

        if any("Harmlessness" in c for c in critiques):
            revised_response = revised_response.replace("buy this", "consider discussing options with your doctor")
            revised_response = revised_response.replace("take this specific brand", "explore various treatment options with a healthcare provider")
            if "Always consult a doctor" not in revised_response:
                revised_response += " It is crucial to always consult a medical professional before making any health decisions."
        
        if any("Factuality" in c for c in critiques):
            revised_response = revised_response.replace("cure all", "may help manage symptoms")
            revised_response = revised_response.replace("guaranteed to work", "has shown promise in some cases")
            if "evidence-based" not in revised_response:
                revised_response += " Ensure any information is verified with evidence-based medical sources."

        if any("Privacy Protection" in c for c in critiques):
            if "Always consult a doctor" not in revised_response:
                 revised_response = "Please remember, this information is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition." + revised_response
            revised_response = revised_response.replace("your symptoms indicate", "information related to your symptoms suggests")
        
        if any("Non-discrimination" in c for c in critiques):
            revised_response = revised_response.replace("only for young people", "relevant for diverse age groups, consult your doctor for suitability")
            revised_response = revised_response.replace("not for certain groups", "applicable to a wide range of individuals, discuss with a healthcare provider for personal relevance")
            if "inclusive advice" not in revised_response:
                revised_response += " The advice aims to be inclusive and applies generally; individual situations may vary."

        if any("Helpfulness" in c for c in critiques):
            revised_response = revised_response.replace("complicated medical jargon", "clear and easy-to-understand explanations")
            if "clear, concise, and easy-to-understand" not in revised_response:
                revised_response += " We strive to provide clear, concise, and easy-to-understand explanations. If anything is unclear, please ask for clarification."

        return revised_response

# Streamlit UI and Application Logic
st.title("Constitutional AI for Health Information")
st.markdown("This system provides personalized health information aligned with ethical principles.")

medical_kb = MedicalKnowledgeBase()
core_llm = DummyCoreLLM()
critique_model = DummyCritiqueModel()
revision_model = DummyRevisionModel()

if "initial_response_count" not in st.session_state:
    st.session_state.initial_response_count = 0

user_query = st.text_input("Ask a health-related question:", "Tell me about diabetes and how to cure it.")

if st.button("Get Health Advice"):
    if user_query:
        st.subheader("--- Processing Request ---")
        
        # 1. RAG: Retrieve context
        medical_context = medical_kb.retrieve(user_query)
        st.write(f"**Retrieved Medical Context:** {medical_context}")

        # 2. Core LLM: Generate initial response
        initial_response = core_llm.generate_response(user_query, medical_context)
        st.write(f"**Initial LLM Response:** {initial_response}")

        current_response = initial_response
        st.markdown("**Applying Constitutional Alignment (up to 3 iterations):**")
        
        for i in range(3):
            st.write(f"\n--- Iteration {i+1} ---")
            
            # 3. Critique Model: Evaluate response
            critiques = critique_model.critique(current_response, CONSTITUTIONAL_PRINCIPLES)
            
            if critiques:
                st.warning("Critiques identified:")
                for critique_msg in critiques:
                    st.write(f"- {critique_msg}")
                
                # 4. Revision Model: Revise response
                revised_response = revision_model.revise(current_response, critiques)
                st.info(f"**Revised Response:** {revised_response}")
                current_response = revised_response
            else:
                st.success("No violations found in this iteration. Response is constitutionally aligned.")
                break
        
        st.subheader("--- Final Aligned Advice ---")
        st.success(current_response)

        # 5. SLAIF Placeholder
        st.sidebar.markdown("### SLAIF (Supervised Learning from AI Feedback)")
        st.sidebar.info("In a real system, the aligned responses would be used to fine-tune or guide the Core LLM to directly generate better responses over time, reducing the need for extensive critique/revision loops.")

st.sidebar.markdown("### Constitutional Principles")
for principle in CONSTITUTIONAL_PRINCIPLES:
    st.sidebar.markdown(f"- {principle}")

st.sidebar.write(f"Initial LLM responses generated: {st.session_state.initial_response_count}")
