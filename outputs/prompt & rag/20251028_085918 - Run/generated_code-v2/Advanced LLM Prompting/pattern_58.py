from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

class MedicalCaseInput(BaseModel):
    patient_history: str
    symptoms: str
    lab_results: str = ""

class MetacognitiveResponse(BaseModel):
    step: str
    ai_response: str
    user_feedback_prompt: str = ""

class RecommendationOutput(BaseModel):
    final_recommendation: str
    ai_confidence: float
    user_confidence_prompt: str

app = FastAPI()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.7)

async def metacognitive_prompting_chain(case: MedicalCaseInput) -> RecommendationOutput:
    context = f"Patient History: {case.patient_history}\nSymptoms: {case.symptoms}\nLab Results: {case.lab_results}"
    
    # Step 1: Clarifying the Question
    clarify_prompt_template = PromptTemplate(
        input_variables=["context"],
        template="Given the following medical case: {context}\n\nBefore providing a preliminary judgment, what clarifying questions do you have for the clinician to ensure a comprehensive understanding? Please list them.\nClarifying Questions:"
    )
    clarify_chain = LLMChain(llm=llm, prompt=clarify_prompt_template)
    clarifying_questions = (await clarify_chain.ainvoke({"context": context}))["text"]
    
    # Simulate clinician providing more details based on clarifying questions
    simulated_clarified_info = f"{context}\nClinician provided further details based on clarifying questions: Patient is 45 years old, no known allergies, recent travel history to Southeast Asia. Symptoms started 3 days ago. \n"

    # Step 2: Preliminary Judgment
    preliminary_prompt_template = PromptTemplate(
        input_variables=["context"],
        template="Given the refined medical case: {context}\n\nProvide a preliminary assessment or a few differential diagnoses, explicitly stating the reasoning behind each. Also, mention any key factors that lead to these diagnoses.\nPreliminary Judgment:"
    )
    preliminary_chain = LLMChain(llm=llm, prompt=preliminary_prompt_template)
    preliminary_judgment = (await preliminary_chain.ainvoke({"context": simulated_clarified_info}))["text"]

    # Simulate clinician evaluation of preliminary judgment
    simulated_clinician_evaluation = f"{preliminary_judgment}\n\nClinician feedback: The differential diagnoses seem reasonable, but I would like to consider a viral infection given the travel history. Are there specific tests to rule this out quickly?"

    # Step 3: Evaluation of Response (AI incorporates feedback)
    evaluation_prompt_template = PromptTemplate(
        input_variables=["feedback"],
        template="Considering the clinician's feedback on your preliminary judgment: {feedback}\n\nHow would you adjust or refine your initial assessment? Present counter-arguments or alternative perspectives if applicable, and suggest further steps based on the feedback.\nRefined Assessment:"
    )
    evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt_template)
    refined_assessment = (await evaluation_chain.ainvoke({"feedback": simulated_clinician_evaluation}))["text"]

    # Simulate clinician decision confirmation
    simulated_confirmation = f"{refined_assessment}\n\nClinician confirmation: Yes, the refined plan considering viral infection and suggested tests aligns well with my observations. I agree to proceed with this plan."

    # Step 4: Decision Confirmation (AI presents final recommendation)
    decision_prompt_template = PromptTemplate(
        input_variables=["confirmed_assessment"],
        template="Based on the clinician's agreement and the refined assessment: {confirmed_assessment}\n\nWhat is the final recommended course of action, including diagnosis and treatment plan?\nFinal Recommendation:"
    )
    decision_chain = LLMChain(llm=llm, prompt=decision_prompt_template)
    final_recommendation = (await decision_chain.ainvoke({"confirmed_assessment": simulated_confirmation}))["text"]

    # Step 5: Confidence Assessment
    confidence_prompt_template = PromptTemplate(
        input_variables=["final_recommendation"],
        template="Considering the entire metacognitive process and the final recommendation: {final_recommendation}\n\nOn a scale of 0 to 1, how confident are you in this final recommendation? Provide a numerical value and explain the basis for your confidence. Also, ask the clinician for their confidence level. (e.g., AI Confidence: 0.85. Basis: Strong evidence, consistent feedback. Clinician, how confident are you in proceeding with this plan?)"
    )
    confidence_chain = LLMChain(llm=llm, prompt=confidence_prompt_template)
    confidence_response_text = (await confidence_chain.ainvoke({"final_recommendation": final_recommendation}))["text"]

    # Extract AI confidence (simplified extraction, actual parsing might be more robust)
    ai_confidence_str = "0.85" # Placeholder, ideally parsed from confidence_response_text
    try:
        ai_confidence = float(ai_confidence_str)
    except ValueError:
        ai_confidence = 0.75 # Default if parsing fails
    
    user_confidence_prompt = "Clinician, how confident are you in proceeding with this plan?"

    return RecommendationOutput(
        final_recommendation=final_recommendation,
        ai_confidence=ai_confidence,
        user_confidence_prompt=user_confidence_prompt
    )

@app.post("/diagnose", response_model=RecommendationOutput)
async def diagnose_medical_case(case: MedicalCaseInput):
    result = await metacognitive_prompting_chain(case)
    return result