import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import random
import asyncio

# --- 0. Mock External Services (for demonstration) ---

class MockLLM:
    """A mock LLM to simulate responses."""
    def __init__(self, name: str = "MockLLM"):
        self.name = name

    def generate(self, prompt: str) -> str:
        """Simulates LLM generation."""
        print(f"[{self.name}] Generating for prompt:\n---START PROMPT---\n{prompt}\n---END PROMPT---")
        # Simulate different responses for ensembling
        if "ENSEMBLE_VARIANT" in prompt:
            variant_id = prompt.split("ENSEMBLE_VARIANT_")[1].split(" ")[0]
            if "troubleshooting" in prompt.lower():
                return f"Response {variant_id} for troubleshooting: Check power cable. LLM {self.name} suggests it might be a loose connection."
            elif "product details" in prompt.lower():
                return f"Response {variant_id} for product details: The 'Xtreme Gadget' features a 12-hour battery life. LLM {self.name} confirmed."
            else:
                return f"Mocked LLM response {variant_id} for: {prompt[:50]}..."
        elif "DEBATE_FOR" in prompt:
            return f"Argument FOR: Our product is highly durable due to its aerospace-grade aluminum casing. This ensures longevity and resistance to impact. (Source: Product Spec Sheet)"
        elif "DEBATE_AGAINST" in prompt:
            return f"Argument AGAINST: While durable, the aluminum casing can add significant weight compared to plastic alternatives, which some users might find less convenient for portability. (Source: User Review Forum)"
        elif "CULTURAL_SENSITIVITY: formal" in prompt:
            return f"Dear Valued Customer, We hope this message finds you well. Regarding your inquiry, kindly be informed..."
        elif "CULTURAL_SENSITIVITY: informal" in prompt:
            return f"Hey there! Thanks for reaching out. About your question, just so you know..."
        elif "TONE: friendly" in prompt:
            return f"Hi! Happy to help you with that. It sounds like a common issue, and we can definitely sort it out for you with a smile!"
        elif "TONE: formal" in prompt:
            return f"Greetings. We acknowledge receipt of your inquiry and are prepared to provide assistance. Please allow us to address your concern with the utmost professionalism."
        elif "VERBOSITY: concise" in prompt:
            return "Quick fix: Reboot device."
        elif "VERBOSITY: detailed" in prompt:
            return "To resolve the issue, please perform a full system reboot by holding the power button for 10 seconds, then waiting 30 seconds before powering it back on. This refreshes system processes."
        else:
            if "troubleshooting" in prompt.lower():
                return "Please try restarting your device to resolve the issue."
            elif "product details" in prompt.lower():
                return "The product features a high-resolution display and a long-lasting battery."
            elif "refund policy" in prompt.lower():
                return "Our refund policy allows returns within 30 days of purchase with a valid receipt."
            else:
                return f"Mocked LLM general response for: {prompt[:50]}..."

class MockVectorDB:
    """A mock vector database to simulate exemplar and knowledge retrieval."""
    def __init__(self):
        self.exemplars = [
            {"id": "ex1", "text": "How to reset password? -> Go to settings, security, then reset password.", "metadata": {"demographic": "general", "product": "all", "sentiment": "neutral", "locale": "en-US"}},
            {"id": "ex2", "text": "My device won't turn on -> Check power supply, then try a hard reset.", "metadata": {"demographic": "tech-savvy", "product": "gadget_x", "sentiment": "frustrated", "locale": "en-US"}},
            {"id": "ex3", "text": "Quiero devolver mi compra -> Consulte nuestra política de devoluciones en el sitio web.", "metadata": {"demographic": "general", "product": "all", "sentiment": "neutral", "locale": "es-ES"}},
            {"id": "ex4", "text": "I want to return my purchase -> Please refer to our return policy on the website.", "metadata": {"demographic": "general", "product": "all", "sentiment": "neutral", "locale": "en-US"}},
            {"id": "ex5", "text": "How do I upgrade my subscription? -> Navigate to account settings and select 'Upgrade Plan'.", "metadata": {"demographic": "general", "product": "service_y", "sentiment": "curious", "locale": "en-US"}},
            {"id": "ex6", "text": "My gadget is overheating. -> Ensure adequate ventilation and close background apps.", "metadata": {"demographic": "tech-savvy", "product": "gadget_x", "sentiment": "concerned", "locale": "en-AU"}},
            {"id": "ex7", "text": "My gadget is overheating. -> Check for excessive background processes.", "metadata": {"demographic": "tech-savvy", "product": "gadget_x", "sentiment": "concerned", "locale": "en-US"}}, # Similar but distinct for DENSE
            {"id": "ex8", "text": "How do I change my profile picture? -> Go to 'My Profile' and click 'Edit Picture'.", "metadata": {"demographic": "novice", "product": "social_app_z", "sentiment": "neutral", "locale": "en-GB"}},
            {"id": "ex9", "text": "What are the benefits of the premium plan? -> Priority support and exclusive features.", "metadata": {"demographic": "business", "product": "service_y", "sentiment": "inquisitive", "locale": "en-US"}},
            {"id": "ex10", "text": "What are the drawbacks of the premium plan? -> Higher cost for features not all users may need.", "metadata": {"demographic": "business", "product": "service_y", "sentiment": "inquisitive", "locale": "en-US"}}
        ]
        self.knowledge_base = [
            {"id": "kb1", "text": "Product A is made of durable aluminum.", "metadata": {"topic": "Product A durability", "stance": "for"}},
            {"id": "kb2", "text": "Product A's aluminum casing increases its weight, potentially affecting portability.", "metadata": {"topic": "Product A durability", "stance": "against"}},
            {"id": "kb3", "text": "Our return policy states a 30-day window for unopened items.", "metadata": {"topic": "Return Policy", "stance": "neutral"}},
            {"id": "kb4", "text": "For hygiene reasons, opened items like headphones cannot be returned.", "metadata": {"topic": "Return Policy", "stance": "against"}}
        ]

    def retrieve_exemplars(self, query: str, num_exemplars: int = 3, filters: Optional[Dict] = None) -> List[Dict]:
        """Simulates retrieval of relevant exemplars."""
        # Simple keyword matching for demonstration
        found_exemplars = []
        for ex in self.exemplars:
            if filters:
                if not all(ex["metadata"].get(k) == v for k, v in filters.items()):
                    continue
            if query.lower() in ex["text"].lower() or any(term in ex["text"].lower() for term in query.lower().split()):
                found_exemplars.append(ex)
            if len(found_exemplars) >= num_exemplars:
                break
        return found_exemplars

    def retrieve_knowledge(self, query: str, topic: str = None, stance: str = None, num_docs: int = 2) -> List[Dict]:
        """Simulates retrieval of knowledge base documents."""
        found_docs = []
        for doc in self.knowledge_base:
            match = False
            if topic and doc["metadata"].get("topic") == topic:
                match = True
            if stance and doc["metadata"].get("stance") == stance:
                match = True
            if query.lower() in doc["text"].lower() or any(term in doc["text"].lower() for term in query.lower().split()):
                match = True

            if match:
                found_docs.append(doc)
            if len(found_docs) >= num_docs:
                break
        return found_docs


# --- I. Core Components (simplified) ---

# Mock the LLM and VectorDB instances globally for this single-file example
mock_llm = MockLLM("MainLLM")
mock_llm_ensemble_1 = MockLLM("EnsembleLLM_1")
mock_llm_ensemble_2 = MockLLM("EnsembleLLM_2")
mock_vector_db = MockVectorDB()

# --- II. AI Design Pattern Modules ---

class DemonstrationEnsemblingModule:
    def __init__(self, llms: List[MockLLM]):
        self.llms = llms

    async def generate_and_aggregate(self, base_prompt: str, num_variants: int = 3) -> str:
        """Generates responses from multiple LLMs with slight prompt variations and aggregates."""
        responses = []
        tasks = []
        for i, llm in enumerate(self.llms):
            # Introduce slight variation for DENSE
            variant_prompt = f"ENSEMBLE_VARIANT_{i+1} {base_prompt}"
            # In a real scenario, this would involve distinct exemplar subsets for each prompt
            # For this mock, we just use a variant tag.
            tasks.append(asyncio.create_task(self._call_llm(llm, variant_prompt)))
        
        raw_responses = await asyncio.gather(*tasks)
        
        # Simple aggregation: e.g., vote on keywords or combine
        # For this mock, we'll concatenate and deduplicate.
        
        # In a real DENSE implementation, you might parse structured outputs
        # and perform majority voting, confidence-weighted averaging, etc.
        aggregated_phrases = []
        for res in raw_responses:
            phrases = [p.strip() for p in res.replace("LLM Mocked LLM response", "").replace("LLM MainLLM", "").replace("LLM EnsembleLLM_1", "").replace("LLM EnsembleLLM_2", "").split(":") if p.strip()]
            aggregated_phrases.extend(phrases)
        
        # Remove duplicates and order
        final_response_parts = list(dict.fromkeys(aggregated_phrases))
        return " ".join(final_response_parts)

    async def _call_llm(self, llm: MockLLM, prompt: str) -> str:
        """Helper to call LLM, can be extended for async calls in real app."""
        # Simulate async call
        await asyncio.sleep(0.1)
        return llm.generate(prompt)

class BalancedDemonstrationsSelector:
    def __init__(self, vector_db: MockVectorDB):
        self.vector_db = vector_db

    def get_balanced_exemplars(self, query: str, user_demographics: Dict, num_exemplars: int = 3) -> List[str]:
        """Retrieves exemplars balanced by user demographics/product type."""
        # For demonstration, let's prioritize locale and then product type
        filters = {"locale": user_demographics.get("locale", "en-US")}
        exemplars = self.vector_db.retrieve_exemplars(query, num_exemplars, filters=filters)

        if len(exemplars) < num_exemplars:
            # If not enough, try to get more, maybe balancing by sentiment or product
            # This is a simplified balancing; real-world would involve more complex sampling.
            remaining_needed = num_exemplars - len(exemplars)
            additional_exemplars = self.vector_db.retrieve_exemplars(query, remaining_needed)
            exemplars.extend([ex for ex in additional_exemplars if ex not in exemplars]) # Avoid exact duplicates

        return [ex["text"] for ex in exemplars[:num_exemplars]]

class CulturalAwarenessModule:
    def adjust_prompt_for_culture(self, prompt: str, user_locale: str) -> str:
        """Adjusts prompt for cultural sensitivity (e.g., formality, common phrases)."""
        if user_locale.startswith("es"): # Spanish-speaking regions
            return f"CULTURAL_SENSITIVITY: formal (Spanish) {prompt}"
        elif user_locale.startswith("fr"): # French-speaking regions
            return f"CULTURAL_SENSITIVITY: formal (French) {prompt}"
        elif user_locale == "en-GB": # British English
            return f"CULTURAL_SENSITIVITY: polite (British) {prompt}"
        elif user_locale == "en-US":
            return f"CULTURAL_SENSITIVITY: neutral (American) {prompt}"
        else: # Default to a general formal tone for unknown locales
            return f"CULTURAL_SENSITIVITY: formal {prompt}"

class AttrPromptModule:
    def inject_attribute_controls(self, prompt: str, attributes: Dict) -> str:
        """Injects instructions for varying attributes like tone, verbosity."""
        modified_prompt = prompt
        if attributes.get("tone"):
            modified_prompt = f"TONE: {attributes['tone']} {modified_prompt}"
        if attributes.get("verbosity"):
            modified_prompt = f"VERBOSITY: {attributes['verbosity']} {modified_prompt}"
        if attributes.get("formality"):
            modified_prompt = f"FORMALITY: {attributes['formality']} {modified_prompt}"
        return modified_prompt

class BiasMitigationModule:
    def __init__(self):
        # In a real system, this would involve loaded bias detection models/rules
        self.biased_keywords = ["problematic_term_1", "gendered_language", "stereotypical_phrase"]

    def check_and_mitigate_response(self, response: str) -> str:
        """Checks response for biases and attempts mitigation."""
        for keyword in self.biased_keywords:
            if keyword in response.lower():
                print(f"BIAS ALERT: Detected potential bias keyword: '{keyword}' in response.")
                # Simple mitigation: replace or flag. Real mitigation would be more sophisticated.
                response = response.replace(keyword, "[REDACTED_BIAS_TERM]")
                response = "Note: This response has been reviewed for potential bias. " + response
        # Add more sophisticated checks here (e.g., sentiment analysis for unfair negative tone)
        return response

class DebateEvidenceAggregator:
    def __init__(self, vector_db: MockVectorDB):
        self.vector_db = vector_db

    async def generate_balanced_explanation(self, query: str, topic: str, llm: MockLLM) -> str:
        """Retrieves evidence for and against a claim and prompts LLM to synthesize."""
        evidence_for = self.vector_db.retrieve_knowledge(query, topic=topic, stance="for", num_docs=1)
        evidence_against = self.vector_db.retrieve_knowledge(query, topic=topic, stance="against", num_docs=1)

        for_text = evidence_for[0]["text"] if evidence_for else "No strong evidence found for this claim."
        against_text = evidence_against[0]["text"] if evidence_against else "No strong evidence found against this claim."

        debate_prompt = (
            f"Here is a claim related to '{topic}': \"{query}\"\n\n"
            f"Evidence supporting the claim: \"{for_text}\"\n"
            f"Evidence opposing the claim: \"{against_text}\"\n\n"
            "Synthesize a balanced explanation, presenting both sides fairly and concluding with a neutral summary. "
            "DEBATE_FOR DEBATE_AGAINST" # Tag for mock LLM
        )
        # Simulate async call
        await asyncio.sleep(0.1)
        return llm.generate(debate_prompt)

# --- Prompt Engineering & Orchestration Layer ---

class CustomerSupportOrchestrator:
    def __init__(self, llm: MockLLM, vector_db: MockVectorDB):
        self.llm = llm
        self.vector_db = vector_db
        self.dense_module = DemonstrationEnsemblingModule([llm, mock_llm_ensemble_1, mock_llm_ensemble_2])
        self.balanced_selector = BalancedDemonstrationsSelector(vector_db)
        self.cultural_awareness = CulturalAwarenessModule()
        self.attr_prompt = AttrPromptModule()
        self.bias_mitigator = BiasMitigationModule()
        self.debate_aggregator = DebateEvidenceAggregator(vector_db)

    async def process_customer_query(self, query: str, user_context: Dict) -> str:
        """
        Orchestrates the processing of a customer query, applying various AI design patterns.
        """
        print(f"\n--- Processing Query: '{query}' with context: {user_context} ---")

        # 1. Select Balanced Demonstrations (if few-shot prompting is used)
        # For this example, we'll retrieve some general exemplars that might be balanced.
        balanced_exemplars_texts = self.balanced_selector.get_balanced_exemplars(
            query, user_context.get("demographics", {}), num_exemplars=2
        )
        exemplar_prompt_part = "\n".join([f"Example: {ex}" for ex in balanced_exemplars_texts])
        base_prompt = f"{exemplar_prompt_part}\nCustomer Query: {query}\nProvide a helpful response."

        # 2. Cultural Awareness (adjust prompt based on locale)
        user_locale = user_context.get("locale", "en-US")
        culturally_aware_prompt = self.cultural_awareness.adjust_prompt_for_culture(base_prompt, user_locale)

        # 3. AttrPrompt (inject desired attributes like tone, verbosity)
        response_attributes = user_context.get("response_attributes", {})
        attr_controlled_prompt = self.attr_prompt.inject_attribute_controls(culturally_aware_prompt, response_attributes)

        final_prompt = attr_controlled_prompt
        response = ""

        # 4. Debate-Style Evidence Aggregation (if query suggests a debate/comparison)
        if "compare" in query.lower() or "pros and cons" in query.lower():
            # Extract topic for debate from query, simplification for mock
            topic_match = "Product A durability" if "product a" in query.lower() else ("Return Policy" if "return policy" in query.lower() else None)
            if topic_match:
                print(f"Applying Debate-Style Evidence Aggregation for topic: {topic_match}")
                response = await self.debate_aggregator.generate_balanced_explanation(query, topic_match, self.llm)
            else:
                response = await self.llm.generate(final_prompt)
        # 5. Demonstration Ensembling (for accuracy in general queries)
        elif user_context.get("use_dense", False):
            print("Applying Demonstration Ensembling (DENSE)")
            response = await self.dense_module.generate_and_aggregate(final_prompt, num_variants=3)
        else:
            # Standard LLM call
            response = await self.llm.generate(final_prompt)

        # 6. Bias-Aware Design & Mitigation (post-processing the LLM output)
        mitigated_response = self.bias_mitigator.check_and_mitigate_response(response)

        print(f"--- Final Response for '{query}': ---")
        return mitigated_response


# --- III. API & User Interface Layer (FastAPI Backend) ---

app = FastAPI(
    title="Advanced AI-Powered Customer Support Assistant",
    description="An AI assistant integrating various design patterns for enhanced LLM performance, bias mitigation, and output quality."
)

# Initialize the orchestrator
orchestrator = CustomerSupportOrchestrator(mock_llm, mock_vector_db)

class CustomerQuery(BaseModel):
    query: str
    user_id: str
    user_context: Dict[str, Any] = {}
    # Example user_context:
    # {
    #   "locale": "en-US",
    #   "demographics": {"age_group": "25-34", "product_ownership": ["gadget_x"]},
    #   "response_attributes": {"tone": "friendly", "verbosity": "concise"},
    #   "use_dense": False # Flag to activate DENSE for specific queries
    # }

class AssistantResponse(BaseModel):
    response: str
    processed_by_patterns: List[str] = [] # To indicate which patterns were used

@app.post("/support/query", response_model=AssistantResponse)
async def handle_customer_query(customer_query: CustomerQuery):
    """
    Handles an incoming customer support query, applying AI design patterns.
    """
    print(f"Received query from user {customer_query.user_id}: {customer_query.query}")

    # For demonstration, we'll manually indicate which patterns are "used" based on flags/keywords
    patterns_applied = []
    if customer_query.user_context.get("use_dense"):
        patterns_applied.append("Demonstration Ensembling (DENSE)")
    if "compare" in customer_query.query.lower() or "pros and cons" in customer_query.query.lower() or "return policy" in customer_query.query.lower():
        patterns_applied.append("Debate-Style Evidence Aggregation")
    
    # These are always applied implicitly through prompt construction/post-processing
    patterns_applied.extend([
        "Selecting Balanced Demonstrations",
        "Cultural Awareness",
        "AttrPrompt",
        "Bias-Aware Design & Mitigation"
    ])
    
    response_text = await orchestrator.process_customer_query(
        customer_query.query, customer_query.user_context
    )

    return AssistantResponse(response=response_text, processed_by_patterns=list(set(patterns_applied))) # Use set to remove duplicates

@app.get("/")
async def root():
    return {"message": "Advanced AI-Powered Customer Support Assistant API is running!"}

# --- Main Execution ---
if __name__ == "__main__":
    # To run this FastAPI app, save it as e.g., `main.py` and run:
    # uvicorn main:app --reload --port 8000
    # For a simple execution without --reload, use:
    uvicorn.run(app, host="0.0.0.0", port=8000)
