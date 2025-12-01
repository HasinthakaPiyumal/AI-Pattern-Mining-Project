import random
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Bias Analyzer Module ---
class BiasAnalyzer:
    def analyze_content_for_bias(self, content: str) -> Dict:
        sentiment = random.choice(["positive", "neutral", "negative"])
        political_stance = random.choice(["left-leaning", "neutral", "right-leaning"])
        topic = random.choice(["politics", "sports", "tech", "entertainment", "finance"])
        return {
            "sentiment": sentiment,
            "political_stance": political_stance,
            "topic": topic
        }

# --- Example Demonstration Pool ---
DEMONSTRATION_POOL = [
    {
        "id": 1,
        "text": "The new economic policy is a disaster for the poor, only benefiting the wealthy elite. It will surely lead to increased inequality.",
        "summary": "New economic policy criticized for increasing inequality and harming the poor.",
        "bias_attributes": {"sentiment": "negative", "political_stance": "left-leaning", "topic": "finance"}
    },
    {
        "id": 2,
        "text": "Our nation's recent tax cuts have spurred unprecedented growth, creating jobs and prosperity for all citizens. A truly remarkable achievement!",
        "summary": "Tax cuts lauded for stimulating economic growth and job creation.",
        "bias_attributes": {"sentiment": "positive", "political_stance": "right-leaning", "topic": "finance"}
    },
    {
        "id": 3,
        "text": "Local community organizes a successful food drive, collecting over 500 cans for needy families. A heartwarming display of solidarity.",
        "summary": "Community food drive collects over 500 cans, showing strong solidarity.",
        "bias_attributes": {"sentiment": "positive", "political_stance": "neutral", "topic": "community"}
    },
    {
        "id": 4,
        "text": "Controversial decision made by the city council regarding new zoning laws sparks debate among residents. Both sides present valid points.",
        "summary": "City council zoning decision causes resident debate, with valid arguments from both sides.",
        "bias_attributes": {"sentiment": "neutral", "political_stance": "neutral", "topic": "politics"}
    },
    {
        "id": 5,
        "text": "Tech giant releases groundbreaking AI model capable of generating highly realistic images from text prompts. Experts are amazed by its capabilities.",
        "summary": "New AI model from tech giant generates realistic images from text, impressing experts.",
        "bias_attributes": {"sentiment": "positive", "political_stance": "neutral", "topic": "tech"}
    },
    {
        "id": 6,
        "text": "Report highlights severe environmental damage caused by recent industrial spill, raising concerns about long-term ecological impact.",
        "summary": "Industrial spill causes severe environmental damage, raising long-term ecological concerns.",
        "bias_attributes": {"sentiment": "negative", "political_stance": "left-leaning", "topic": "environment"}
    },
    {
        "id": 7,
        "text": "Government announces new initiatives to boost small businesses, including grants and mentorship programs. A step in the right direction for the economy.",
        "summary": "Government launches new initiatives, including grants and mentorship, to support small businesses.",
        "bias_attributes": {"sentiment": "positive", "political_stance": "right-leaning", "topic": "finance"}
    },
    {
        "id": 8,
        "text": "Recent study suggests a significant link between excessive screen time and sleep disturbances in adolescents. Further research is needed.",
        "summary": "Study suggests link between screen time and adolescent sleep issues, requiring more research.",
        "bias_attributes": {"sentiment": "neutral", "political_stance": "neutral", "topic": "health"}
    },
]

# --- 2. Demonstration Selector Module ---
class DemonstrationSelector:
    def __init__(self, demonstration_pool: List[Dict]):
        self.demonstration_pool = demonstration_pool

    def select_balanced_demonstrations(self, target_content_biases: Dict, num_demonstrations: int = 4) -> List[Dict]:
        selected_demonstrations = []
        available_pool = list(self.demonstration_pool)

        bias_keys = list(target_content_biases.keys())
        if not bias_keys:
            return random.sample(available_pool, min(num_demonstrations, len(available_pool)))

        attempts_per_demo = 5

        while len(selected_demonstrations) < num_demonstrations and available_pool:
            found_demonstration = False
            for _ in range(attempts_per_demo):
                if not available_pool:
                    break
                
                candidate = random.choice(available_pool)
                is_balanced_candidate = True

                for selected_demo in selected_demonstrations:
                    for bias_key in bias_keys:
                        if bias_key in candidate.get('bias_attributes', {}) and \
                           bias_key in selected_demo.get('bias_attributes', {}) and \
                           candidate['bias_attributes'][bias_key] == selected_demo['bias_attributes'][bias_key]:
                            is_balanced_candidate = False
                            break
                    if not is_balanced_candidate:
                        break
                
                if is_balanced_candidate:
                    selected_demonstrations.append(candidate)
                    available_pool.remove(candidate)
                    found_demonstration = True
                    break
            
            if not found_demonstration and available_pool:
                candidate = random.choice(available_pool)
                selected_demonstrations.append(candidate)
                available_pool.remove(candidate)
                
        return selected_demonstrations

# --- 3. LLM Summarizer Module ---
class LLMSummarizer:
    def __init__(self, llm_model_name: str = "SimulatedLLM"):
        self.llm_model_name = llm_model_name

    def _construct_few_shot_prompt(self, content: str, demonstrations: List[Dict]) -> str:
        prompt_parts = ["Summarize the following articles/posts in a neutral and concise manner."]

        for i, demo in enumerate(demonstrations):
            prompt_parts.append(f"\n\n### Example {i+1} Input:\n{demo['text']}")
            prompt_parts.append(f"### Example {i+1} Summary:\n{demo['summary']}")
        
        prompt_parts.append(f"\n\n### Input to Summarize:\n{content}")
        prompt_parts.append(f"### Summary:")

        return "\n".join(prompt_parts)

    def summarize(self, content: str, demonstrations: List[Dict]) -> str:
        prompt = self._construct_few_shot_prompt(content, demonstrations)

        simulated_summary = f"[SIMULATED LLM SUMMARY]: Summary of the input content based on provided examples. Length: {len(content.split()) // 5} words."
        if demonstrations:
            first_demo_biases = demonstrations[0].get('bias_attributes', {})
            if first_demo_biases:
                simulated_summary += f" Detected target biases: {[f'{k}: {v}' for k, v in first_demo_biases.items()]}."

        return simulated_summary

# --- 4. API/Interface Layer (FastAPI) ---
app = FastAPI(title="Bias-Mitigated Content Summarization API")

# Initialize core components
bias_analyzer = BiasAnalyzer()
demonstration_selector = DemonstrationSelector(DEMONSTRATION_POOL)
llm_summarizer = LLMSummarizer()

class SummarizeRequest(BaseModel):
    content: str

class SummarizeResponse(BaseModel):
    summary: str
    detected_biases: Dict
    selected_demonstration_ids: List[int]

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_content(request: SummarizeRequest):
    content = request.content

    target_content_biases = bias_analyzer.analyze_content_for_bias(content)
    balanced_demonstrations = demonstration_selector.select_balanced_demonstrations(
        target_content_biases, num_demonstrations=3
    )
    generated_summary = llm_summarizer.summarize(content, balanced_demonstrations)

    selected_ids = [demo['id'] for demo in balanced_demonstrations]

    return SummarizeResponse(
        summary=generated_summary,
        detected_biases=target_content_biases,
        selected_demonstration_ids=selected_ids
    )

# To run this API:
# 1. Save the code as `main.py`
# 2. Install dependencies: `pip install fastapi uvicorn pydantic`
# 3. Run from your terminal: `uvicorn main:app --reload`
# 4. Access the API documentation at http://127.0.0.1:8000/docs

