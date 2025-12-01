from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import random
import uvicorn

# Placeholder for a Language Model service
class LLMService:
    def generate_descriptions(self, product_details: str, customer_segment: str, num_samples: int) -> List[str]:
        base_description = f"A {product_details} perfect for {customer_segment}s."
        return [f"{base_description} This is candidate {i+1}." for i in range(num_samples)]

# Placeholder for a Reward Model service
class RewardModelService:
    def score_description(self, description: str, product_details: str, customer_segment: str) -> float:
        return random.uniform(0.1, 0.9) # Simulate a score between 0.1 and 0.9

class RejectionSamplingOrchestrator:
    def __init__(self, llm_service: LLMService, reward_model_service: RewardModelService):
        self.llm_service = llm_service
        self.reward_model_service = reward_model_service

    def get_best_description(self, product_details: str, customer_segment: str, num_samples: int = 5) -> str:
        candidate_descriptions = self.llm_service.generate_descriptions(product_details, customer_segment, num_samples)
        
        best_description = ""
        max_score = -1.0

        for description in candidate_descriptions:
            score = self.reward_model_service.score_description(description, product_details, customer_segment)
            if score > max_score:
                max_score = score
                best_description = description
        
        return best_description

app = FastAPI()

llm_service = LLMService()
reward_model_service = RewardModelService()
orchestrator = RejectionSamplingOrchestrator(llm_service, reward_model_service)

class ProductDescriptionRequest(BaseModel):
    product_details: str
    customer_segment: str
    num_samples: int = 5

class ProductDescriptionResponse(BaseModel):
    generated_description: str

@app.post("/generate-description", response_model=ProductDescriptionResponse)
def generate_product_description(request: ProductDescriptionRequest):
    best_description = orchestrator.get_best_description(
        request.product_details,
        request.customer_segment,
        request.num_samples
    )
    return ProductDescriptionResponse(generated_description=best_description)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
