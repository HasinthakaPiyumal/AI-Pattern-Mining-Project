from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
from typing import List, Dict, Any

class LLMClient:
    def generate(self, prompt: str) -> str:
        if "generate synthetic data" in prompt.lower():
            return f"Simulated synthetic data based on prompt: '{prompt}'. Example: {{'product': 'Smartwatch X', 'feature': 'Heart Rate Monitor', 'sentiment': 'Positive'}} (Generated for: {prompt})"
        elif "pro argument for" in prompt.lower():
            return f"Pro-argument for '{prompt.replace('What is a pro argument for the claim: ', '').replace('?', '')}': This claim is supported because of simulated evidence A."
        elif "con argument against" in prompt.lower():
            return f"Con-argument against '{prompt.replace('What is a con argument against the claim: ', '').replace('?', '')}': This claim is refuted because of simulated evidence B."
        else:
            return f"LLM Response to: '{prompt}'"

class ChatbotService:
    def __init__(self):
        self.llm_client = LLMClient()
        self.exemplars = {
            "shipping": [
                {"input": "Where is my order?", "output": "Please provide your order number to track your shipment."},
                {"input": "When will my package arrive?", "output": "Shipping usually takes 3-5 business days. Can I get your order ID?"},
                {"input": "Is my item shipped yet?", "output": "I can check that for you. What is your order reference?"},
                {"input": "My delivery is late.", "output": "I apologize for the delay. Could you please share your order number?"}
            ],
            "returns": [
                {"input": "How do I return an item?", "output": "You can initiate a return through your account's order history within 30 days."},
                {"input": "Can I get a refund?", "output": "Refunds are processed once the returned item is received and inspected."},
                {"input": "What's your return policy?", "output": "Our return policy allows returns within 30 days of purchase for a full refund."}
            ],
            "product_info": [
                {"input": "Tell me about product X.", "output": "Product X is a high-performance gadget with features A, B, and C."},
                {"input": "Is product Y in stock?", "output": "Product Y is currently in stock. Would you like to add it to your cart?"}
            ]
        }
        self.cultural_contexts = {
            "US": {"greeting": "Hello", "farewell": "Thank you", "tone": "direct"},
            "JP": {"greeting": "Kon'nichiwa", "farewell": "Arigato", "tone": "polite"},
            "DE": {"greeting": "Guten Tag", "farewell": "Danke", "tone": "formal"}
        }

    def _demonstration_ensembling(self, query: str, intent_key: str, num_demonstrations: int = 2) -> List[str]:
        available_exemplars = self.exemplars.get(intent_key, [])
        if not available_exemplars:
            return [f"Query: {query}\nResponse:"]

        prompts = []
        for _ in range(3):
            selected_exemplars = random.sample(available_exemplars, min(num_demonstrations, len(available_exemplars)))
            demonstrations_str = "\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in selected_exemplars])
            prompts.append(f"{demonstrations_str}\nInput: {query}\nOutput:")
        return prompts

    def _select_balanced_demonstrations(self, all_exemplars: List[Dict[str, str]], count: int) -> List[Dict[str, str]]:
        if len(all_exemplars) <= count:
            return all_exemplars
        return random.sample(all_exemplars, count)

    def _apply_cultural_awareness(self, prompt: str, cultural_context: str = "US") -> str:
        context_data = self.cultural_contexts.get(cultural_context, self.cultural_contexts["US"])
        greeting = context_data["greeting"]
        farewell = context_data["farewell"]
        tone_instruction = ""
        if context_data["tone"] == "polite":
            tone_instruction = " Respond very politely and respectfully."
        elif context_data["tone"] == "formal":
            tone_instruction = " Use a formal tone."

        return f"{greeting}!{tone_instruction} {prompt} {farewell}."

    def _mitigate_bias(self, response: str) -> str:
        response = response.replace("he said", "they said").replace("she said", "they said")
        return response

    def generate_synthetic_data(self, base_prompt: str, attributes_to_vary: Dict[str, List[str]]) -> List[str]:
        synthetic_data_outputs = []
        variations = [{}]
        
        for attr, values in attributes_to_vary.items():
            new_variations = []
            for prev_var in variations:
                for val in values:
                    new_var = prev_var.copy()
                    new_var[attr] = val
                    new_variations.append(new_var)
            variations = new_variations

        for variation in variations:
            current_prompt = base_prompt
            for attr, val in variation.items():
                current_prompt = current_prompt.replace(f"{{{attr}}}", str(val))
            
            final_prompt = f"Generate a detailed response based on: '{current_prompt}'. Ensure diversity in output."
            synthetic_data_outputs.append(self.llm_client.generate(final_prompt))
        return synthetic_data_outputs

    def aggregate_debate_evidence(self, claim: str) -> Dict[str, str]:
        pro_prompt = f"What is a pro argument for the claim: '{claim}'?"
        con_prompt = f"What is a con argument against the claim: '{claim}'?"

        pro_argument = self.llm_client.generate(pro_prompt)
        con_argument = self.llm_client.generate(con_prompt)

        return {
            "claim": claim,
            "pro_argument": pro_argument,
            "con_argument": con_argument,
            "balanced_assessment": f"A balanced view on '{claim}' considers: {pro_argument} and {con_argument}."
        }

    def handle_query(self, query: str, cultural_context: str = "US", intent_key: str = "shipping") -> str:
        balanced_demonstrations = self._select_balanced_demonstrations(self.exemplars.get(intent_key, []), 2)
        
        ensembled_prompts = self._demonstration_ensembling(query, intent_key)
        
        culturally_aware_prompts = [self._apply_cultural_awareness(p, cultural_context) for p in ensembled_prompts]
        
        final_prompt_to_llm = culturally_aware_prompts[0] if culturally_aware_prompts else self._apply_cultural_awareness(f"Answer the following question: {query}", cultural_context)

        raw_llm_response = self.llm_client.generate(final_prompt_to_llm)
        
        mitigated_response = self._mitigate_bias(raw_llm_response)
        
        return mitigated_response

app = FastAPI()
chatbot_service = ChatbotService()

class ChatRequest(BaseModel):
    query: str
    cultural_context: str = "US"
    intent_key: str = "shipping"

class ChatResponse(BaseModel):
    response: str

class SyntheticDataRequest(BaseModel):
    base_prompt: str
    attributes_to_vary: Dict[str, List[str]]

class SyntheticDataResponse(BaseModel):
    generated_data: List[str]

class EvidenceAggregationRequest(BaseModel):
    claim: str

class EvidenceAggregationResponse(BaseModel):
    claim: str
    pro_argument: str
    con_argument: str
    balanced_assessment: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    response = chatbot_service.handle_query(request.query, request.cultural_context, request.intent_key)
    return ChatResponse(response=response)

@app.post("/generate_synthetic_data", response_model=SyntheticDataResponse)
async def generate_synthetic_data_endpoint(request: SyntheticDataRequest):
    generated_data = chatbot_service.generate_synthetic_data(request.base_prompt, request.attributes_to_vary)
    return SyntheticDataResponse(generated_data=generated_data)

@app.post("/aggregate_evidence", response_model=EvidenceAggregationResponse)
async def aggregate_evidence_endpoint(request: EvidenceAggregationRequest):
    evidence = chatbot_service.aggregate_debate_evidence(request.claim)
    return EvidenceAggregationResponse(**evidence)
