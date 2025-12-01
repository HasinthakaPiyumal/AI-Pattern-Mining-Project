import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

llm_pipeline = None
try:
    llm_pipeline = pipeline("text2text-generation", model="google/mt5-base", tokenizer="google/mt5-base")
except Exception as e:
    print(f"Error loading LLM: {e}. Please ensure you have sufficient memory or consider a smaller model or running on GPU.")
    print("Chatbot functionality will be limited until the LLM can be loaded.")

CROSS_LINGUAL_EXAMPLES = [
    {
        "input": "User query: I need to return a product. The product description says 'defekt'. What does it mean in English? (Query in Spanish: Necesito devolver un producto. La descripción del producto dice 'defekt'. ¿Qué significa en inglés?)",
        "output": "Response in English: 'Defekt' means 'defective' in English. Can you tell me more about the issue with your product? (Respuesta en inglés: 'Defekt' significa 'defectuoso' en inglés. ¿Puedes darme más detalles sobre el problema con tu producto?)"
    },
    {
        "input": "Customer question: My order status is 'en cours de livraison'. When will it arrive? (Question in German: Mein Bestellstatus ist 'en cours de livraison'. Wann kommt es an?)",
        "output": "Response in German: 'En cours de livraison' means 'in delivery' in French. It should arrive soon. Would you like me to check the exact estimated delivery date? (Antwort auf Deutsch: 'En cours de livraison' bedeutet 'in Lieferung' auf Französisch. Es sollte bald ankommen. Möchten Sie, dass ich das genaue voraussichtliche Lieferdatum überprüfe?)"
    },
    {
        "input": "I have a question about the 'Garantiebedingungen'. Can you explain them in English? (Query in Japanese: 'Garantiebedingungen'について質問があります。英語で説明してもらえますか？)",
        "output": "Response in English: 'Garantiebedingungen' means 'warranty conditions' in German. Please tell me which specific conditions you'd like to understand. (回答は英語: 「Garantiebedingungen」はドイツ語で「保証条件」を意味します。具体的な条件についてどの点をご理解されたいか教えてください。)"
    }
]

def build_icl_prompt(user_query: str, target_language: str = "en") -> str:
    system_instruction = (
        "You are a helpful multilingual customer support assistant for a global e-commerce platform. "
        "Your goal is to assist customers with their queries, even when their questions involve "
        "information or concepts across different languages. Leverage your cross-lingual understanding "
        "to provide accurate and contextually relevant responses. "
        f"The user expects the response primarily in {target_language}."
    )

    example_strings = []
    for example in CROSS_LINGUAL_EXAMPLES:
        example_strings.append(f"{example['input']}\n{example['output']}")

    icl_examples_section = "\n\n".join(example_strings)

    prompt_template = PromptTemplate(
        template="{system_instruction}\n\n{icl_examples_section}\n\nUser query: {user_query}\nResponse:",
        input_variables=["system_instruction", "icl_examples_section", "user_query"]
    )

    final_prompt = prompt_template.format(
        system_instruction=system_instruction,
        icl_examples_section=icl_examples_section,
        user_query=user_query
    )
    return final_prompt

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    target_language: str = "en"

class ChatResponse(BaseModel):
    response: str
    detected_language: str
    prompt_used: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if llm_pipeline is None:
        return ChatResponse(response="Chatbot service is temporarily unavailable due to an issue loading the language model. Please try again later.", detected_language="en", prompt_used="LLM loading failed.")

    user_query = request.query
    target_language = request.target_language

    detected_lang = "unknown"
    try:
        detected_lang = detect(user_query)
    except Exception as e:
        pass

    final_prompt = build_icl_prompt(user_query, target_language)

    try:
        llm_output = llm_pipeline(
            final_prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.01,
            top_p=0.9,
            num_return_sequences=1
        )
        generated_text = llm_output[0]['generated_text'].strip()

        response_marker = "Response:"
        if response_marker in generated_text:
            actual_response = generated_text.split(response_marker, 1)[1].strip()
        else:
            user_query_marker = f"User query: {user_query}"
            if user_query_marker in generated_text:
                actual_response = generated_text.split(user_query_marker, 1)[1].replace("Response:", "").strip()
            else:
                actual_response = generated_text

    except Exception as e:
        actual_response = f"An error occurred during LLM generation: {e}"

    return ChatResponse(
        response=actual_response,
        detected_language=detected_lang,
        prompt_used=final_prompt
    )