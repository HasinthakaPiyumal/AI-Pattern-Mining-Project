
import gradio as gr
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import requests
import json
from threading import Thread

app = FastAPI()

session_data = {}

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

try:
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    translator_pipeline = pipeline("translation_en_to_es", model=model, tokenizer=tokenizer)
except Exception:
    class DummyModel:
        def __init__(self):
            pass
        def __call__(self, text, max_length=512):
            return [{"translation_text": f"DUMMY_TRANSLATION: {text}"}]
        def generate(self, input_ids, max_length=512, num_beams=5, early_stopping=True, no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95, temperature=0.7):
            return [0] # Dummy output for tokenizer.decode

    translator_pipeline = DummyModel()
    tokenizer = None
    model = None

class InitialTranslateRequest(BaseModel):
    text: str
    session_id: str

class FinalTranslateRequest(BaseModel):
    session_id: str
    clarifications: dict

@app.post("/initial_translate")
async def initial_translate(request: InitialTranslateRequest):
    original_text = request.text
    session_id = request.session_id

    initial_translation_output = translator_pipeline(original_text, max_length=512)
    initial_translation = initial_translation_output[0]["translation_text"]

    prompt_for_questions = (
        f"Identify ambiguous phrases in the following English text and ask concise clarifying questions for a human expert. "
        f"Format each as: [AMBIGUITY]: [QUESTION]\n\n"
        f"English Text: {original_text}"
    )
    
    if model and tokenizer:
        input_ids = tokenizer.encode(prompt_for_questions, return_tensors="pt")
        generated_ids = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        questions_raw_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    else:
        questions_raw_output = "[AMBIGUITY]: What is the exact meaning of 'flu-like symptoms'?\n[AMBIGUITY]: Clarify 'cardiac arrest' in this context."

    questions_list = []
    for line in questions_raw_output.split("\n"):
        if "[AMBIGUITY]:" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                question_part = parts[1].strip()
                if question_part.startswith("What is") or question_part.startswith("Clarify"): # Simple heuristic to filter questions
                    questions_list.append(question_part)

    session_data[session_id] = {
        "original_text": original_text,
        "initial_translation": initial_translation,
        "questions": questions_list,
    }

    return {"initial_translation": initial_translation, "questions": questions_list}


@app.post("/final_translate")
async def final_translate(request: FinalTranslateRequest):
    session_id = request.session_id
    clarifications = request.clarifications

    if session_id not in session_data:
        return {"error": "Session not found or expired."}

    data = session_data[session_id]
    original_text = data["original_text"]
    initial_translation = data["initial_translation"]

    clarifications_str = "\n".join([f"- {q}: {a}" for q, a in clarifications.items()])

    final_translation_prompt = (
        f"Translate the following English medical text to Spanish, incorporating the provided clarifications to resolve ambiguities. "
        f"Original English Text: {original_text}\n"
        f"Initial Spanish Translation (for context): {initial_translation}\n"
        f"Clarifications from Expert:\n{clarifications_str}\n\n"
        f"Final, accurate Spanish Translation:"
    )

    if model and tokenizer:
        input_ids = tokenizer.encode(final_translation_prompt, return_tensors="pt")
        generated_ids = model.generate(input_ids, max_length=1024, num_beams=5, early_stopping=True)
        final_translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    else:
        final_translation = f"DUMMY_FINAL_TRANSLATION with clarifications: {clarifications_str}"

    if session_id in session_data:
        del session_data[session_id]

    return {"final_translation": final_translation}


def get_initial_translation_and_questions_gradio(text):
    session_id = str(uuid.uuid4())
    try:
        response = requests.post(
            "http://localhost:8000/initial_translate",
            json={"text": text, "session_id": session_id}
        ).json()
        initial_translation = response.get("initial_translation", "")
        questions = response.get("questions", [])

        q_labels_updates = []
        a_inputs_updates = []
        
        for i in range(3):
            if i < len(questions):
                q_labels_updates.append(gr.Textbox.update(value=questions[i], visible=True, label=f"Question {i+1}"))
                a_inputs_updates.append(gr.Textbox.update(value="", visible=True, label=f"Answer {i+1}"))
            else:
                q_labels_updates.append(gr.Textbox.update(value="", visible=False))
                a_inputs_updates.append(gr.Textbox.update(value="", visible=False))

        return (initial_translation, json.dumps(questions, indent=2), session_id, *q_labels_updates, *a_inputs_updates)
    except requests.exceptions.ConnectionError:
        return ("Error: FastAPI backend not running. Please start the backend.", "[]", "", *[gr.Textbox.update(visible=False)]*6)
    except Exception as e:
        return (f"An error occurred: {e}", "[]", "", *[gr.Textbox.update(visible=False)]*6)


def get_final_translation_gradio(session_id, q1_text, a1_text, q2_text, a2_text, q3_text, a3_text):
    if not session_id:
        return "Error: No active session. Please start from Step 1."

    clarifications = {}
    if q1_text and a1_text:
        clarifications[q1_text] = a1_text
    if q2_text and a2_text:
        clarifications[q2_text] = a2_text
    if q3_text and a3_text:
        clarifications[q3_text] = a3_text
    
    try:
        response = requests.post(
            "http://localhost:8000/final_translate",
            json={"session_id": session_id, "clarifications": clarifications}
        ).json()
        return response.get("final_translation", "Error: Could not get final translation.")
    except requests.exceptions.ConnectionError:
        return "Error: FastAPI backend not running. Please start the backend."
    except Exception as e:
        return f"An error occurred: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## Medical Document Interactive Translation System (ICP)")
    gr.Markdown("This system uses GenAI to translate medical documents. If ambiguities are detected, it will ask for human clarification to refine the final translation.")

    session_id_state = gr.State(value="")

    with gr.Tab("Step 1: Upload Document & Get Initial Translation"):
        text_input = gr.Textbox(label="Enter Medical Document Text (English)", lines=10, placeholder="e.g., The patient presented with a severe case of 'flu-like symptoms' and 'cardiac arrest'.")
        translate_btn = gr.Button("1. Get Initial Translation & Clarification Questions")
        initial_translation_output = gr.Textbox(label="Initial Translation (Spanish)", interactive=False, lines=5)
        question_json_output = gr.JSON(label="Raw Clarification Questions (for debugging)", interactive=False)

    with gr.Tab("Step 2: Human Clarification"):
        gr.Markdown("Provide answers to the generated clarification questions below:")
        
        q1_label = gr.Textbox(label="Question 1", interactive=False, visible=False)
        a1_input = gr.Textbox(label="Answer 1", visible=False)
        q2_label = gr.Textbox(label="Question 2", interactive=False, visible=False)
        a2_input = gr.Textbox(label="Answer 2", visible=False)
        q3_label = gr.Textbox(label="Question 3", interactive=False, visible=False)
        a3_input = gr.Textbox(label="Answer 3", visible=False)
        
        submit_clarifications_btn = gr.Button("2. Get Final Translation with Clarifications")
        final_translation_output = gr.Textbox(label="Final Translation (Spanish)", interactive=False, lines=5)

    translate_btn.click(
        get_initial_translation_and_questions_gradio,
        inputs=[text_input],
        outputs=[
            initial_translation_output,
            question_json_output,
            session_id_state,
            q1_label, a1_input,
            q2_label, a2_input,
            q3_label, a3_input
        ]
    )

    submit_clarifications_btn.click(
        get_final_translation_gradio,
        inputs=[
            session_id_state,
            q1_label, a1_input,
            q2_label, a2_input,
            q3_label, a3_input
        ],
        outputs=[final_translation_output]
    )

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

fastapi_thread = Thread(target=run_fastapi)
fastapi_thread.daemon = True
fastapi_thread.start()

print("Starting Gradio interface... please wait for FastAPI backend to be ready.")
demo.launch(server_name="0.0.0.0", server_port=7860)
