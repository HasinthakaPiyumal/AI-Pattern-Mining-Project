import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import random
from typing import List, Dict, Any, Optional

app = FastAPI(
    title="Culturally Aware & Bias-Mitigated Customer Support LLM",
    description="LLM service for global e-commerce, focusing on cultural sensitivity, bias mitigation, and robust responses."
)

# Placeholder for a loaded LLM pipeline
# In a real-world scenario, you would load a more powerful model like Llama 2, Mistral, etc.
# For demonstration, we use a simple text generation pipeline.
# Make sure to install transformers and a backend like PyTorch or TensorFlow.
# Example: pip install transformers torch

try:
    llm_pipeline = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    print(f"Warning: Could not load default text-generation pipeline (distilgpt2). "
          f"Please ensure 'transformers' and a backend (e.g., 'torch') are installed: {e}")
    llm_pipeline = None # Set to None if loading fails


class QueryRequest(BaseModel):
    query: str
    customer_id: Optional[str] = None
    culture_context: Optional[str] = "neutral"  # e.g., "Japanese", "German", "Brazilian"


class LLMCustomerSupport:
    def __init__(self, llm_pipeline_instance):
        self.llm = llm_pipeline_instance
        if not self.llm:
            raise RuntimeError("LLM pipeline not initialized. Cannot process requests.")

        # Pre-defined exemplars for demonstration. In a real system, these would be dynamic.
        self.exemplars = {
            "product_info": [
                {"input": "What is the warranty for product X?", "output": "The standard warranty for product X is one year from the date of purchase.", "culture": "neutral"},
                {"input": "Can you tell me more about the features of the new smartphone?", "output": "The new smartphone boasts a high-resolution camera, long-lasting battery, and a vibrant OLED display.", "culture": "neutral"},
                {"input": "Does this item ship to Germany?", "output": "Yes, we offer shipping to Germany for this item. Standard international shipping rates apply.", "culture": "German"},
                {"input": "Do you have this product in stock in Tokyo?", "output": "We have limited stock of this product in our Tokyo warehouse. Please check the product page for real-time availability.", "culture": "Japanese"}
            ],
            "return_policy": [
                {"input": "What is your return policy?", "output": "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition.", "culture": "neutral"},
                {"input": "I want to return a faulty item, how do I proceed?", "output": "Please visit our returns page and follow the instructions to initiate a return for a faulty item.", "culture": "neutral"}
            ],
            "shipping_inquiry": [
                {"input": "How long will shipping take to France?", "output": "Standard shipping to France typically takes 5-7 business days.", "culture": "French"},
                {"input": "When will my order arrive? (Order ID: 12345)", "output": "Your order with ID 12345 is expected to arrive by [Date]. You can track it here: [Tracking Link].", "culture": "neutral"}
            ]
        }

    def _select_balanced_demonstrations(self, query_category: str, num_demos: int = 2, culture_context: str = "neutral") -> List[Dict[str, str]]:
        """Selects balanced demonstrations based on query category and cultural context."""
        available_demos = self.exemplars.get(query_category, [])
        
        # Prioritize culturally relevant examples
        cultural_demos = [d for d in available_demos if d.get("culture") == culture_context and d.get("culture") != "neutral"]
        neutral_demos = [d for d in available_demos if d.get("culture") == "neutral"]
        other_cultural_demos = [d for d in available_demos if d.get("culture") != culture_context and d.get("culture") != "neutral"]

        selected_demos = []
        # Try to include at least one cultural demo if available and relevant
        if cultural_demos and num_demos > 0:
            selected_demos.append(random.choice(cultural_demos))
            num_demos -= 1
        
        # Fill the rest with neutral or other cultural demos, ensuring balance
        remaining_demos_pool = neutral_demos + other_cultural_demos
        random.shuffle(remaining_demos_pool)
        selected_demos.extend(remaining_demos_pool[:num_demos])

        return selected_demos

    def _construct_prompt(
        self, 
        query: str,
        culture_context: str,
        demonstrations: List[Dict[str, str]],
        include_debate: bool = False
    ) -> str:
        """Constructs a prompt with cultural awareness, demonstrations, and optional debate style."""
        prompt_parts = []
        
        # Cultural Awareness Injection
        if culture_context and culture_context != "neutral":
            prompt_parts.append(f"You are a helpful customer support agent for a global e-commerce platform. Please provide culturally sensitive responses appropriate for a customer in {culture_context}.")
        else:
            prompt_parts.append("You are a helpful customer support agent for a global e-commerce platform. Provide clear and concise responses.")

        # Few-Shot Demonstrations
        if demonstrations:
            prompt_parts.append("\nHere are some examples of good customer support interactions:")
            for demo in demonstrations:
                prompt_parts.append(f"Customer: {demo['input']}\nAgent: {demo['output']}")
            prompt_parts.append("\nNow, answer the customer's query based on the examples and your knowledge:")

        # Debate-Style Evidence Aggregation
        if include_debate:
            prompt_parts.append("\nWhen responding to the following query, present arguments both for and against potential solutions or claims, providing a balanced perspective. Conclude with a recommendation based on the balanced view.")
        
        prompt_parts.append(f"Customer: {query}\nAgent:")
        
        return "\n".join(prompt_parts)

    def _apply_dense(self, query: str, culture_context: str, query_category: str, num_ensembles: int = 3) -> str:
        """Applies Demonstration Ensembling (DENSE) by aggregating outputs from multiple prompts."""
        responses = []
        for i in range(num_ensembles):
            # Vary demonstrations for each ensemble member
            demos = self._select_balanced_demonstrations(query_category, num_demos=2, culture_context=culture_context)
            prompt = self._construct_prompt(query, culture_context, demos)
            
            # Simulate LLM call
            # In a real scenario, this would involve proper stopping conditions and response parsing.
            if self.llm:
                try:
                    output = self.llm(prompt, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']
                    # Extract only the agent's response part
                    agent_response = output.split("Agent:")[-1].strip()
                    responses.append(agent_response)
                except Exception as e:
                    print(f"Error during LLM inference for DENSE ensemble {i}: {e}")
                    responses.append(f"Error processing query in ensemble {i}.")
            else:
                responses.append(f"LLM not available. Simulated response for ensemble {i}.")

        # Simple aggregation: return the most common response (for classification-like tasks)
        # or a summary of responses (for generative tasks).
        # For this demonstration, we'll join them, indicating different perspectives.
        if len(set(responses)) == 1: # All responses are identical
            return responses[0]
        else:
            return 