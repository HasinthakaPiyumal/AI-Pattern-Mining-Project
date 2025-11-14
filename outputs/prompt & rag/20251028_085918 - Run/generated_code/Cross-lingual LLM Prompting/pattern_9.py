import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
import torch

# Define FastAPI app
app = FastAPI(
    title="Multilingual Customer Support Chatbot",
    description="Chatbot leveraging InCLT Crosslingual Transfer Prompting to assist customers in multiple languages."
)

# Pydantic models for request and response
class ChatRequest(BaseModel):
    user_query: str
    user_language: str # e.g., "en", "es", "fr", "de"

class ChatResponse(BaseModel):
    response: str

# 1. Initialize Multilingual LLM
# Using facebook/mbart-large-50 for its strong multilingual capabilities
MODEL_NAME = "facebook/mbart-large-50"

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

llm = None
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    text_generation_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=200,
        truncation=True,
        do_sample=True,
        top_k=50,
        temperature=0.7
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Please ensure you have `torch` installed and a compatible GPU, or sufficient RAM for CPU inference.")

# 2. In-Context Learning (InCLT) Examples
INCLT_EXAMPLES = [
    {
        "source_query": "Qual è il mio numero d'ordine?", # Italian
        "target_concept": "Order number inquiry", # English concept
        "target_response": "Il tuo numero d'ordine è 12345. Posso aiutarti con qualcos'altro?" # Italian response
    },
    {
        "source_query": "¿Cuál es la política de devoluciones?", # Spanish
        "target_concept": "Return policy question", # English concept
        "target_response": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días de la compra." # Spanish response
    },
    {
        "source_query": "How do I track my package?", # English
        "target_concept": "Package tracking", # English concept
        "target_response": "You can track your package by entering your tracking number on our website." # English response
    },
    {
        "source_query": "Comment réinitialiser mon mot de passe?", # French
        "target_concept": "Password reset", # English concept
        "target_response": "Vous pouvez réinitialiser votre mot de passe en cliquant sur 'Mot de passe oublié' sur la page de connexion." # French response
    }
]

# 3. Prompt Management (Langchain PromptTemplate)
SYSTEM_INSTRUCTIONS = """
You are a helpful and friendly multilingual customer support assistant.
Your goal is to answer customer queries accurately and politely.
Leverage the provided examples to understand cross-lingual contexts and respond in the customer's language.
If you cannot find relevant information, politely state that you cannot assist with that specific query.
"""

EXAMPLE_TEMPLATE = """
Customer Query (Source Lang): {source_query}
Bot's Internal Understanding (Concept): {target_concept}
Bot Response (Target Lang - usually same as source or based on instruction): {target_response}
"""

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["system_instructions", "in_context_examples", "user_language", "user_query"],
    template="""{system_instructions}

Here are some examples of how to understand and respond to customer queries across different languages:
{in_context_examples}

---
Customer Query ({user_language}): {user_query}
Bot Response ({user_language}):"""
)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if llm is None:
        return ChatResponse(response="Chatbot service is not available. LLM failed to initialize.")

    # Format in-context examples for the prompt
    formatted_examples = ""
    for example in INCLT_EXAMPLES:
        formatted_examples += EXAMPLE_TEMPLATE.format(
            source_query=example["source_query"],
            target_concept=example["target_concept"],
            target_response=example["target_response"]
        )
        formatted_examples += "\n" # Add a newline after each example

    # Construct the full prompt
    full_prompt = PROMPT_TEMPLATE.format(
        system_instructions=SYSTEM_INSTRUCTIONS,
        in_context_examples=formatted_examples.strip(), # Remove trailing newline
        user_language=request.user_language,
        user_query=request.user_query
    )

    print(f"--- Full Prompt Sent to LLM ---\n{full_prompt}\n------------------------------") # For debugging

    try:
        raw_llm_response = llm.invoke(full_prompt)

        # Basic cleaning to extract only the bot's response
        cleaned_response = raw_llm_response
        response_prefix = f"Bot Response ({request.user_language}):"

        # Attempt to split after the expected prefix
        if response_prefix in cleaned_response:
            cleaned_response = cleaned_response.split(response_prefix, 1)[1].strip()
        else:
            # Fallback if prefix not found, try to remove the prompt part heuristically
            # This is a common issue with generic text2text models when prompted directly.
            # More robust parsing or fine-tuning is needed for production.
            cleaned_response = cleaned_response.replace(full_prompt, "").strip()

            # Further heuristic cleaning for MBart which might echo parts of the input
            if request.user_query in cleaned_response:
                cleaned_response = cleaned_response.split(request.user_query, 1)[-1].strip()

            # If still problematic, try to find the last meaningful part after an instruction
            lines = cleaned_response.split('\n')
            filtered_lines = []
            capture_next = False
            for line in reversed(lines):
                if line.strip().startswith("Bot Response"):
                    filtered_lines.insert(0, line.split(':', 1)[-1].strip())
                    capture_next = True
                    break
                elif capture_next:
                    filtered_lines.insert(0, line.strip())
            if filtered_lines:
                cleaned_response = " ".join(filtered_lines).strip()
            else:
                # Final crude fallback if nothing else works
                cleaned_response = raw_llm_response.split(':')[-1].strip()

        # Handle cases where the response might be empty or too short after cleaning
        if not cleaned_response or len(cleaned_response) < 5:
            cleaned_response = "I am sorry, I couldn't generate a relevant response. Please try again or rephrase your query."

        return ChatResponse(response=cleaned_response.strip())

    except Exception as e:
        print(f"Error during LLM inference: {e}")
        return ChatResponse(response="An error occurred while processing your request. Please try again later.")
