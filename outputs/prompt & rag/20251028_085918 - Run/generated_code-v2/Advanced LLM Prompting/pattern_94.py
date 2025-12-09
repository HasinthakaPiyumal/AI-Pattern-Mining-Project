import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import language_tool_python

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
translator_pipeline = pipeline("translation", model=model, tokenizer=tokenizer)

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if source_lang == "en" and target_lang == "fr":
        result = translator_pipeline(text)
        return result[0]["translation_text"]
    else:
        return f"Error: Translation from {source_lang} to {target_lang} not supported by this demo model."

lang_tool = language_tool_python.LanguageTool('fr')

def check_grammar_style(text: str) -> str:
    matches = lang_tool.check(text)
    suggestions = []
    for match in matches:
        suggestions.append(f"Error: {match.message} (Context: '{match.context}', Suggestions: {', '.join(match.replacements)})")
    if not suggestions:
        return "No grammar or style suggestions."
    return "\n".join(suggestions)

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "fr"

class RefineRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    draft_translation: str

class RefinementResponse(BaseModel):
    suggestions: str

@app.post("/translate", response_model=TranslationResponse)
async def api_translate(request: TranslateRequest):
    try:
        translated_text = translate_text(request.text, request.source_lang, request.target_lang)
        return TranslationResponse(draft_translation=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refine", response_model=RefinementResponse)
async def api_refine(request: RefineRequest):
    try:
        suggestions_text = check_grammar_style(request.text)
        return RefinementResponse(suggestions=suggestions_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

BACKEND_URL = "http://localhost:8000"

def get_initial_draft(customer_query: str, target_lang: str):
    try:
        response = requests.post(
            f"{BACKEND_URL}/translate",
            json={"text": customer_query, "source_lang": "en", "target_lang": target_lang}
        )
        response.raise_for_status()
        draft = response.json().get("draft_translation", "Error: No translation received.")
    except requests.exceptions.ConnectionError:
        draft = "Error: FastAPI backend not running or unreachable. Please start it first."
    except Exception as e:
        draft = f"An error occurred: {e}"
    return draft, draft

def get_refinement_suggestions(edited_translation: str):
    try:
        response = requests.post(
            f"{BACKEND_URL}/refine",
            json={"text": edited_translation}
        )
        response.raise_for_status()
        suggestions = response.json().get("suggestions", "Error: No suggestions received.")
    except requests.exceptions.ConnectionError:
        suggestions = "Error: FastAPI backend not running or unreachable. Please start it first."
    except Exception as e:
        suggestions = f"An error occurred: {e}"
    return suggestions

with gr.Blocks() as demo:
    gr.Markdown("# Multilingual Customer Support Assistant (Iterative Prompting Demo)")
    gr.Markdown("Enter a customer query, get an AI draft translation, refine it, and get grammar/style suggestions.")

    with gr.Row():
        query_input = gr.Textbox(label="Customer Query (English)", placeholder="e.g., My internet is not working.")
        target_lang_dropdown = gr.Dropdown(
            ["fr"],
            label="Target Language (for AI translation)",
            value="fr",
            interactive=True
        )
        initial_translate_btn = gr.Button("1. Get Draft Translation")

    with gr.Row():
        draft_output = gr.Textbox(label="AI Draft Translation", interactive=False)

    with gr.Row():
        human_edited_translation = gr.Textbox(label="2. Human Edited Translation (Refine Here)", placeholder="Edit the AI draft translation for accuracy and fluency.")
        refine_btn = gr.Button("3. Check Grammar/Style for Refinement")

    with gr.Row():
        suggestions_output = gr.Textbox(label="Grammar/Style Suggestions", interactive=False)

    initial_translate_btn.click(
        get_initial_draft,
        inputs=[query_input, target_lang_dropdown],
        outputs=[draft_output, human_edited_translation]
    )

    refine_btn.click(
        get_refinement_suggestions,
        inputs=[human_edited_translation],
        outputs=suggestions_output
    )

if __name__ == "__main__":
    print("--- Multilingual Customer Support Assistant Setup ---")
    print("To run this application:")
    print("1. Save this code as 'app.py'.")
    print("2. Open a new terminal and start the FastAPI backend:")
    print("   `uvicorn app:app --reload`")
    print("3. Once the FastAPI server is running, run this script in another terminal to launch the Gradio frontend:")
    print("   `python app.py`")
    print("\nLaunching Gradio interface (ensure FastAPI backend is running!)...")
    demo.launch()