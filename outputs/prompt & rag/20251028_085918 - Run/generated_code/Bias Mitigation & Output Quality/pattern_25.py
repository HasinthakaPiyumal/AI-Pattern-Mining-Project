import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import random
import statistics

class MockLLMService:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def generate(self, prompt: str):
        if "diagnosis" in prompt.lower():
            if "headache" in prompt.lower() and "fever" in prompt.lower():
                return "Potential flu or common cold. Consult a doctor for definitive diagnosis."
            elif "chest pain" in prompt.lower():
                return "Urgent medical attention is advised for chest pain. Could be cardiac related."
            else:
                return "Generic diagnostic advice: Symptoms suggest a mild condition, but professional medical opinion is crucial."
        elif "dietary recommendation" in prompt.lower():
            if "vegetarian" in prompt.lower():
                return "For a vegetarian diet, focus on lentils, beans, tofu, and a variety of fruits and vegetables."
            elif "mediterranean" in prompt.lower():
                return "A Mediterranean diet emphasizes whole grains, olive oil, fish, and fresh produce."
            else:
                return "General dietary advice: balanced meals, plenty of water, and limited processed foods."
        elif "cultural context: japanese" in prompt.lower():
            return "Culturally adapted advice for Japan: Consider green tea for relaxation and traditional Japanese dietary principles."
        elif "cultural context: indian" in prompt.lower():
            return "Culturally adapted advice for India: Ayurvedic principles or traditional home remedies might be relevant. Consult local practitioners."
        elif "evidence for" in prompt.lower() and "against" in prompt.lower():
            return (
                "Evidence for a claim: Studies show positive outcomes in X% of cases. "
                "Evidence against a claim: Some research indicates potential side effects or no significant improvement in Y% of cases."
            )
        else:
            return f"Mock LLM response for: '{prompt[:50]}...'"

llm_service = MockLLMService()

class AssistantRequest(BaseModel):
    query: str
    cultural_context: str = "general"
    enable_dense: bool = True
    enable_debate: bool = False

class AssistantResponse(BaseModel):
    response: str
    debug_info: dict = {}

def _apply_cultural_awareness(prompt: str, cultural_context: str) -> str:
    if cultural_context.lower() == "japanese":
        return f"Considering Japanese cultural nuances and common health practices, {prompt}"
    elif cultural_context.lower() == "indian":
        return f"Considering Indian cultural nuances and traditional health practices (e.g., Ayurveda), {prompt}"
    else:
        return prompt

def _generate_dense_prompts(query: str, cultural_context: str, num_variations: int = 3) -> list[str]:
    base_prompt = f"Provide health information or diagnostic support for: {query}"
    prompts = [_apply_cultural_awareness(base_prompt, cultural_context)]
    
    for i in range(num_variations - 1):
        variation = random.choice([
            f"Please explain {query} thoroughly.",
            f"What are the implications of {query}?",
            f"Can you offer advice on {query} from a medical perspective?",
            f"Elaborate on {query}."
        ])
        prompts.append(_apply_cultural_awareness(variation, cultural_context))
    return prompts

def _aggregate_responses(responses: list[str]) -> str:
    if not responses:
        return "No clear response from ensembling."
    return " ".join(list(set(responses)))

def _get_debate_style_evidence(query: str) -> dict:
    prompt_for = f"Provide evidence supporting the effectiveness of treatment/approach for {query}."
    prompt_against = f"Provide evidence against or limitations of treatment/approach for {query}."
    
    response_for = llm_service.generate(prompt_for)
    response_against = llm_service.generate(prompt_against)
    
    return {
        "for": response_for,
        "against": response_against
    }

def _generate_synthetic_patient_data(num_samples: int = 100):
    genders = ["Male", "Female", "Non-binary"]
    ethnicities = ["Caucasian", "African American", "Asian", "Hispanic", "Other"]
    conditions = ["Hypertension", "Diabetes", "Asthma", "Allergies", "None"]
    
    synthetic_data = []
    for _ in range(num_samples):
        synthetic_data.append({
            "age": random.randint(18, 90),
            "gender": random.choice(genders),
            "ethnicity": random.choice(ethnicities),
            "condition": random.choice(conditions)
        })
    return synthetic_data

def _select_balanced_demonstrations(data: list[dict], query_type: str = "diagnosis"):
    balanced_subset = []
    genders = ["Male", "Female", "Non-binary"] # Re-define for scope
    ethnicities = ["Caucasian", "African American", "Asian", "Hispanic", "Other"] # Re-define for scope

    gender_count = {g: 0 for g in genders}
    ethnicity_count = {e: 0 for e in ethnicities}
    
    for patient in data:
        if gender_count[patient["gender"]] < 5 and ethnicity_count[patient["ethnicity"]] < 5:
            balanced_subset.append(patient)
            gender_count[patient["gender"]] += 1
            ethnicity_count[patient["ethnicity"]] += 1
        if len(balanced_subset) >= 30:
            break
    
    return balanced_subset[:min(len(data), 20)]

synthetic_patient_data = _generate_synthetic_patient_data(200)

def _apply_guardrails(text: str) -> (bool, str):
    disallowed_keywords = ["magic cure", "guaranteed fix", "unproven remedy", "harmful advice"]
    for keyword in disallowed_keywords:
        if keyword in text.lower():
            return False, "Output contains potentially harmful or unproven medical advice."
    
    if "culturally insensitive" in text.lower():
        return False, "Output flagged for cultural insensitivity."
            
    return True, text

app = FastAPI(
    title="Personalized Healthcare Assistant AI",
    description="An AI platform providing accurate, fair, and culturally sensitive health information and preliminary diagnostic support."
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Healthcare Assistant is running."}

@app.post("/assist", response_model=AssistantResponse)
async def get_healthcare_assistance(request: AssistantRequest):
    debug_info = {}
    final_response = ""

    base_prompt_with_cultural_context = _apply_cultural_awareness(request.query, request.cultural_context)
    debug_info["base_prompt_cultural"] = base_prompt_with_cultural_context

    if request.enable_dense:
        dense_prompts = _generate_dense_prompts(request.query, request.cultural_context)
        debug_info["dense_prompts"] = dense_prompts
        
        dense_responses = []
        for prompt in dense_prompts:
            dense_responses.append(llm_service.generate(prompt))
        
        final_response = _aggregate_responses(dense_responses)
        debug_info["dense_raw_responses"] = dense_responses
        debug_info["dense_aggregated_response"] = final_response
    else:
        final_response = llm_service.generate(base_prompt_with_cultural_context)
        debug_info["single_llm_response"] = final_response

    if request.enable_debate:
        debate_evidence = _get_debate_style_evidence(request.query)
        final_response += "\n\n--- Debate-Style Evidence ---\n"
        final_response += f"For: {debate_evidence['for']}\n"
        final_response += f"Against: {debate_evidence['against']}\n"
        debug_info["debate_evidence"] = debate_evidence

    balanced_demos_info = _select_balanced_demonstrations(synthetic_patient_data, "diagnosis")
    debug_info["balanced_demonstrations_concept"] = f"A balanced set of {len(balanced_demos_info)} demonstrations would be used for training this type of query."

    is_safe, guarded_response = _apply_guardrails(final_response)
    if not is_safe:
        final_response = f"Warning: {guarded_response} Original response suppressed or modified."
        debug_info["guardrails_status"] = "failed"
        debug_info["guardrails_message"] = guarded_response
    else:
        final_response = guarded_response
        debug_info["guardrails_status"] = "passed"

    return AssistantResponse(response=final_response, debug_info=debug_info)