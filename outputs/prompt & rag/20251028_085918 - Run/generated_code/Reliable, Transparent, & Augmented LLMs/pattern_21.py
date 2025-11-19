"""
medchat_assistant.py

This file contains the core logic for the MedChat AI assistant, including
integration with mock external medical tools (knowledge base, diagnostic algorithm)
and a mock LLM. It handles processing queries, generating responses with reasoning
and confidence scores, and implementing progressive disclosure.
"""

import time
from typing import Dict, Any, List

class MedicalKnowledgeBase:
    """
    A mock medical knowledge base to simulate searching for health information.
    In a real application, this would connect to actual medical databases.
    """
    def __init__(self):
        self.knowledge = {
            "hypertension": "High blood pressure (hypertension) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
            "diabetes": "Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.",
            "common cold": "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way.",
            "symptoms_hypertension": ["headaches", "shortness of breath", "nosebleeds"],
            "treatment_hypertension": ["lifestyle changes", "medication (diuretics, ACE inhibitors)"],
            "symptoms_diabetes": ["frequent urination", "increased thirst", "unexplained weight loss"],
            "treatment_diabetes": ["dietary management", "exercise", "medication (insulin, metformin)"],
        }

    def search(self, query: str) -> str:
        """
        Simulates searching the knowledge base for information related to the query.
        """
        query_lower = query.lower()
        if "hypertension" in query_lower:
            return self.knowledge.get("hypertension", "") + " Symptoms: " + ", ".join(self.knowledge.get("symptoms_hypertension", [])) + ". Treatment: " + ", ".join(self.knowledge.get("treatment_hypertension", []))
        elif "diabetes" in query_lower:
            return self.knowledge.get("diabetes", "") + " Symptoms: " + ", ".join(self.knowledge.get("symptoms_diabetes", [])) + ". Treatment: " + ", ".join(self.knowledge.get("treatment_diabetes", []))
        elif "cold" in query_lower:
            return self.knowledge.get("common cold", "")
        return "No specific information found for: " + query

class DiagnosticAlgorithm:
    """
    A mock diagnostic algorithm to simulate processing symptoms and suggesting a diagnosis.
    In a real application, this would be a more complex model or rule-based system.
    """
    def diagnose(self, symptoms: List[str]) -> str:
        """
        Simulates making a diagnosis based on a list of symptoms.
        """
        symptoms_lower = [s.lower() for s in symptoms]
        if "headache" in symptoms_lower and "shortness of breath" in symptoms_lower:
            return "Potential diagnosis: Hypertension. Please consult a doctor for confirmation."
        if "frequent urination" in symptoms_lower and "increased thirst" in symptoms_lower:
            return "Potential diagnosis: Diabetes. Please consult a doctor for confirmation."
        if "runny nose" in symptoms_lower and "sore throat" in symptoms_lower:
            return "Potential diagnosis: Common Cold. Rest and symptomatic treatment are usually sufficient."
        return "Cannot make a definitive diagnosis based on provided symptoms. Further investigation is needed."

class MockLLM:
    """
    A mock Large Language Model (LLM) to simulate generating responses.
    This class is a placeholder for actual LLM integrations (e.g., OpenAI, Gemini, etc.).
    It simulates reasoning paths and self-rated confidence scores.
    """
    def __init__(self):
        pass

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Simulates LLM processing and generating a response with reasoning and confidence.
        """
        if "diagnose" in prompt.lower() and "symptoms" in prompt.lower():
            diagnosis_result = self._get_mock_diagnosis_response(prompt)
            return {
                "medical_info": diagnosis_result["medical_info"],
                "reasoning": diagnosis_result["reasoning"],
                "confidence_score": diagnosis_result["confidence_score"]
            }
        elif "medical information" in prompt.lower() or "knowledge about" in prompt.lower() or "tell me about" in prompt.lower():
            info_result = self._get_mock_info_response(prompt)
            return {
                "medical_info": info_result["medical_info"],
                "reasoning": info_result["reasoning"],
                "confidence_score": info_result["confidence_score"]
            }
        else:
            return {
                "medical_info": "I understand your query. Please specify if you need information, a potential diagnosis, or treatment options.",
                "reasoning": "The query was too general, prompting for more specific input.",
                "confidence_score": 0.6
            }

    def _get_mock_diagnosis_response(self, prompt: str) -> Dict[str, Any]:
        """
        Provides mock diagnostic responses based on keywords in the prompt.
        """
        if "headache and shortness of breath" in prompt.lower():
            return {
                "medical_info": "Based on the symptoms of headache and shortness of breath, a potential diagnosis could be Hypertension. It's crucial to confirm this with a healthcare professional through blood pressure measurements and other tests.",
                "reasoning": "Symptoms align with known indicators of hypertension, as cross-referenced with internal medical guidelines and diagnostic patterns. The diagnostic algorithm tool was consulted.",
                "confidence_score": 0.85
            }
        elif "frequent urination and increased thirst" in prompt.lower():
            return {
                "medical_info": "The symptoms of frequent urination and increased thirst are characteristic of Diabetes. A blood glucose test is highly recommended for definitive diagnosis.",
                "reasoning": "These are classic symptoms of diabetes mellitus, derived from epidemiological data and clinical diagnostic criteria. The diagnostic algorithm tool was consulted.",
                "confidence_score": 0.92
            }
        else:
            return {
                "medical_info": "Based on the provided symptoms, further information or diagnostic tests would be required for a more specific diagnosis. Please provide more details or consult a medical professional.",
                "reasoning": "Insufficient distinct symptoms provided to narrow down a specific condition with high confidence, even after consulting the diagnostic algorithm.",
                "confidence_score": 0.55
            }

    def _get_mock_info_response(self, prompt: str) -> Dict[str, Any]:
        """
        Provides mock information responses based on keywords in the prompt.
        """
        if "hypertension" in prompt.lower():
            return {
                "medical_info": "Hypertension, or high blood pressure, significantly increases the risk of heart disease and stroke. It's often called a 'silent killer' because it usually has no warning signs or symptoms. Lifestyle changes and medication can manage it.",
                "reasoning": "Extracted core definition, common risks, and general management from the medical knowledge base.",
                "confidence_score": 0.95
            }
        elif "diabetes" in prompt.lower():
            return {
                "medical_info": "Diabetes is a chronic metabolic disease characterized by elevated levels of blood glucose (blood sugar), which over time leads to serious damage to the heart, blood vessels, eyes, kidneys and nerves. Management involves diet, exercise, and often medication.",
                "reasoning": "Summarized the definition, long-term complications, and management from standard medical texts found in the knowledge base.",
                "confidence_score": 0.93
            }
        else:
            return {
                "medical_info": "I can provide general medical information. Could you please specify a condition or topic?",
                "reasoning": "The information query was too broad, requiring refinement to access specific knowledge effectively.",
                "confidence_score": 0.7
            }


class MedChatAssistant:
    """
    The main class for the MedChat AI diagnostic assistant.
    It orchestrates interactions between the LLM and external medical tools,
    and structures the output for trustworthiness and explainability.
    """
    def __init__(self):
        self.llm = MockLLM()  # Replace with actual LLM integration (e.g., Langchain, OpenAI SDK) in a real app
        self.knowledge_base = MedicalKnowledgeBase()
        self.diagnostic_algorithm = DiagnosticAlgorithm()

    def _call_external_tools(self, query: str) -> Dict[str, Any]:
        """
        Determines which external tools to call based on the query and retrieves information.
        """
        tool_results = {}
        query_lower = query.lower()

        # Check for diagnostic intent and extract symptoms
        if "diagnose" in query_lower or "symptoms" in query_lower:
            symptoms = []
            if "headache" in query_lower: symptoms.append("headache")
            if "shortness of breath" in query_lower: symptoms.append("shortness of breath")
            if "frequent urination" in query_lower: symptoms.append("frequent urination")
            if "increased thirst" in query_lower: symptoms.append("increased thirst")
            if "runny nose" in query_lower: symptoms.append("runny nose")
            if "sore throat" in query_lower: symptoms.append("sore throat")
            
            if symptoms:
                diagnosis = self.diagnostic_algorithm.diagnose(symptoms)
                tool_results["diagnostic_tool_output"] = diagnosis
        
        # Check for information retrieval intent
        if "medical information" in query_lower or "what is" in query_lower or "tell me about" in query_lower or "knowledge about" in query_lower:
            keywords = ["hypertension", "diabetes", "common cold"]
            found_info = []
            for kw in keywords:
                if kw in query_lower:
                    info = self.knowledge_base.search(kw)
                    if info:
                        found_info.append(info)
            if found_info:
                tool_results["knowledge_base_output"] = " ".join(found_info)
        
        return tool_results

    def _progressive_disclosure(self, full_response: Dict[str, Any], step_delay: float = 0.5) -> List[Dict[str, Any]]:
        """
        Reveals parts of the response progressively to enhance transparency.
        """
        disclosed_steps = []
        
        # Step 1: Core information
        step1 = {
            "step": 1,
            "content": {"medical_info": full_response["medical_info"]},
            "message": "Here is the core information regarding your query."
        }
        disclosed_steps.append(step1)
        # In a real-time system, this would be streamed or sent incrementally.
        # time.sleep(step_delay) # Simulate delay

        # Step 2: Add reasoning
        step2 = {
            "step": 2,
            "content": {
                "medical_info": full_response["medical_info"],
                "reasoning": full_response["reasoning"]
            },
            "message": "Here is the reasoning behind this information."
        }
        disclosed_steps.append(step2)
        # time.sleep(step_delay)

        # Step 3: Add confidence score and disclaimer
        step3 = {
            "step": 3,
            "content": full_response, # Full response
            "message": f"The system's confidence in this response is {full_response['confidence_score']:.0%}. Always consult a healthcare professional for definitive diagnosis and treatment."
        }
        disclosed_steps.append(step3)
        
        return disclosed_steps

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a user's medical query, integrates external tools, generates LLM response,
        and applies progressive disclosure.
        """
        tool_outputs = self._call_external_tools(query)
        
        # Construct prompt for LLM, incorporating tool outputs to enhance context
        llm_prompt = f"User query: '{query}'\n"
        if tool_outputs:
            llm_prompt += "\nExternal tool results and observations:\n"
            for tool_name, result in tool_outputs.items():
                llm_prompt += f"- {tool_name}: {result}\n"
        
        llm_response = self.llm.generate_response(llm_prompt)
        
        full_response = {
            "medical_info": llm_response.get("medical_info", "No specific medical information available."),
            "reasoning": llm_response.get("reasoning", "Reasoning not provided."),
            "confidence_score": llm_response.get("confidence_score", 0.0)
        }
        
        progressive_output = self._progressive_disclosure(full_response)
        
        return {
            "query": query,
            "external_tool_outputs_used": tool_outputs,
            "final_ai_response": full_response,
            "progressive_disclosure_steps": progressive_output
        }
