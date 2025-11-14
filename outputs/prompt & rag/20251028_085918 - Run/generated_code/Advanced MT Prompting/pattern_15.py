
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langdetect import detect, DetectorFactory
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import gradio as gr
import threading
import time

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

app = FastAPI()

# --- 1. Global Components & Models Initialization ---

# Mock external MT system (e.g., Google Cloud Translation, DeepL)
class MockTranslationAPI:
    def translate(self, text: str, target_language: str, source_language: str = "auto") -> str:
        # In a real scenario, this would call an external API.
        # For demonstration, we'll just simulate a translation to English if not already English.
        if target_language == "en" and source_language != "en":
            print(f"[Mock MT] Translating '{text}' from {source_language} to {target_language}")
            # Very basic mock: just append a tag indicating translation
            return f"[TRANSLATED_FROM_{source_language.upper()}] {text}"
        return text

mock_mt_api = MockTranslationAPI()

# Multilingual LLM for contextual understanding and generation
# Using a smaller model for demonstration; replace with larger models like NLLB, mBART, Llama 2 for production
# For text generation, we'll use a simple pipeline with a general-purpose text generation model.
print("Loading tokenizer and model for Multilingual LLM (e.g., Google/flan-t5-small). This might take a moment...")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
llm_pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer)
print("Multilingual LLM loaded.")

# Sentence Transformer for embeddings and semantic search
print("Loading Sentence Transformer (e.g., paraphrase-multilingual-MiniLM-L12-v2). This might take a moment...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Sentence Transformer loaded.")

# Domain Glossary/Knowledge Base (simplified as a dictionary for demonstration)
# In a real app, this would be a vector DB like ChromaDB with more extensive content.
DOMAIN_KNOWLEDGE_BASE = {
    "shipping_policy": {
        "keywords": ["shipping", "delivery", "track order", "where is my package"],
        "answer_en": "Our standard shipping takes 3-5 business days. You can track your order using the link provided in your confirmation email.",
        "answer_es": "Nuestro envío estándar tarda de 3 a 5 días hábiles. Puede rastrear su pedido utilizando el enlace proporcionado en su correo electrónico de confirmación."
    },
    "return_policy": {
        "keywords": ["return", "refund", "exchange", "damaged item"],
        "answer_en": "We offer free returns within 30 days of purchase. Items must be in original condition. Please visit our returns page for more details.",
        "answer_es": "Ofrecemos devoluciones gratuitas dentro de los 30 días posteriores a la compra. Los artículos deben estar en su estado original. Visite nuestra página de devoluciones para obtener más detalles."
    },
    "payment_methods": {
        "keywords": ["payment", "credit card", "paypal", "methods"],
        "answer_en": "We accept Visa, MasterCard, American Express, PayPal, and Google Pay.",
        "answer_es": "Aceptamos Visa, MasterCard, American Express, PayPal y Google Pay."
    }
}

# In-memory storage for conversation history and HIL drafts
conversation_history = {}
current_hil_drafts = {}

# --- 2. Pydantic Models for FastAPI ---

class CustomerQuery(BaseModel):
    session_id: str
    text: str

class HILFeedback(BaseModel):
    session_id: str
    agent_id: str
    approved: bool
    edited_text: str = None

class AIResponse(BaseModel):
    session_id: str
    original_query: str
    detected_language: str
    translated_query_en: str
    draft_response: str
    requires_human_review: bool
    final_response: str = None # Only populated after HIL approval

# --- 3. Core AI Logic Functions ---

def get_relevant_knowledge(query_embeddings):
    """Simulates semantic search on the knowledge base."""
    best_match = None
    max_similarity = -1
    
    query_embedding = embedding_model.encode(query_embeddings, convert_to_tensor=True)

    for key, item in DOMAIN_KNOWLEDGE_BASE.items():
        for keyword in item["keywords"]:
            keyword_embedding = embedding_model.encode(keyword, convert_to_tensor=True)
            similarity = float(torch.nn.functional.cosine_similarity(query_embedding, keyword_embedding, dim=0))
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = item
    
    # Set a threshold for relevance
    if max_similarity > 0.6: # Adjust threshold as needed
        return best_match
    return None

def generate_llm_response(prompt: str) -> str:
    """Generates a response using the loaded LLM pipeline."""
    # A more sophisticated LangChain agent would handle this with prompts and few-shot examples
    print(f"[LLM] Generating response for prompt: {prompt[:100]}...")
    try:
        # Ensure the input is a string for the pipeline
        result = llm_pipeline(str(prompt), max_new_tokens=150, num_return_sequences=1)
        return result[0]['generated_text']
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "I'm sorry, I'm having trouble generating a response right now. Please try again."

def process_customer_query_internal(session_id: str, text: str) -> AIResponse:
    """Internal logic for processing a customer query through the AI pipeline."""
    detected_lang = "en"
    try:
        detected_lang = detect(text)
    except:
        print("Could not detect language, defaulting to English.")
        detected_lang = "en"

    translated_text_en = text
    if detected_lang != "en":
        translated_text_en = mock_mt_api.translate(text, target_language="en", source_language=detected_lang)
    
    # Store in history
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({"role": "user", "text": text, "lang": detected_lang})

    # Query Decomposition & Intent Recognition (simplified)
    # In a full LangChain setup, this would be an agent with tools.
    intent_prompt = f"Analyze the following customer query and identify the main intent. Query: '{translated_text_en}'\nIntent: "
    intent = generate_llm_response(intent_prompt)
    print(f"[Intent] Detected intent: {intent}")

    # Context Augmentation: Knowledge Base Retrieval
    relevant_knowledge = get_relevant_knowledge(translated_text_en)
    knowledge_context = ""
    if relevant_knowledge:
        knowledge_context = f"Using the following knowledge: {relevant_knowledge['answer_en']}.\n"
        print(f"[Knowledge] Retrieved relevant knowledge: {relevant_knowledge['answer_en'][:50]}...")

    # Response Planning & Generation
    # This is a simplified chain. LangChain would handle more complex multi-step reasoning.
    response_prompt = (
        f"You are a helpful customer support agent. Based on the user's query and the provided context, "
        f"generate a concise and helpful response. If the query is complex or ambiguous, indicate that human review might be needed.\n"
        f"User Query (translated to English): '{translated_text_en}'\n"
        f"{knowledge_context}"
        f"Previous conversation (if any): {conversation_history.get(session_id, [])[-2:]}\n"
        f"Your response: "
    )

    draft_response_en = generate_llm_response(response_prompt)

    # Determine if human review is needed (simplified)
    requires_human_review = "human review" in draft_response_en.lower() or "ambiguous" in draft_response_en.lower() or len(text) > 200 # Example condition

    # Store the draft for HIL
    if requires_human_review:
        current_hil_drafts[session_id] = {
            "original_query": text,
            "detected_language": detected_lang,
            "translated_query_en": translated_text_en,
            "draft_response_en": draft_response_en
        }
    
    # If not requiring human review, translate back to original language (mock for now)
    final_response = draft_response_en
    if not requires_human_review and detected_lang != "en":
        # In a real system, this would translate draft_response_en back to detected_lang
        final_response = f"[BACK_TRANSLATED_TO_{detected_lang.upper()}] {draft_response_en}"

    conversation_history[session_id].append({"role": "agent_draft", "text": draft_response_en, "lang": "en"})

    return AIResponse(
        session_id=session_id,
        original_query=text,
        detected_language=detected_lang,
        translated_query_en=translated_text_en,
        draft_response=draft_response_en,
        requires_human_review=requires_human_review,
        final_response=final_response if not requires_human_review else None
    )

# --- 4. FastAPI Endpoints ---

@app.post("/query", response_model=AIResponse)
async def customer_query(query: CustomerQuery):
    """Receives a customer query and processes it through the AI pipeline."""
    print(f"[API] Received query for session {query.session_id}: {query.text}")
    response = process_customer_query_internal(query.session_id, query.text)
    return response

@app.post("/feedback")
async def human_feedback(feedback: HILFeedback):
    """Receives human feedback for draft responses."""
    session_id = feedback.session_id
    if session_id not in current_hil_drafts:
        raise HTTPException(status_code=404, detail="No pending draft for this session ID.")
    
    draft_data = current_hil_drafts.pop(session_id) # Remove after feedback

    final_text = feedback.edited_text if feedback.edited_text else draft_data["draft_response_en"]

    if feedback.approved:
        print(f"[HIL] Session {session_id} approved by agent {feedback.agent_id}. Final text: {final_text[:50]}...")
        # In a real system, translate final_text back to original_lang if needed
        original_lang = draft_data["detected_language"]
        if original_lang != "en":
            final_text = f"[BACK_TRANSLATED_TO_{original_lang.upper()}] {final_text}"
            
        # Update conversation history with the final human-approved response
        conversation_history[session_id].append({"role": "agent_final", "text": final_text, "lang": original_lang})

        return {"message": "Feedback processed. Response finalized.", "final_response": final_text}
    else:
        print(f"[HIL] Session {session_id} rejected by agent {feedback.agent_id}. Edited: {final_text[:50]}...")
        # If rejected or edited, we might re-prompt the LLM or just use the human's edit.
        # For this demo, we'll consider the edited_text as the new final_response if provided.
        # If not approved and no edit, it implies further work is needed or the query cannot be answered.
        final_text = "I'm sorry, I need more information or a human agent will assist you shortly." if not feedback.edited_text else final_text
        
        original_lang = draft_data["detected_language"]
        if original_lang != "en":
            final_text = f"[BACK_TRANSLATED_TO_{original_lang.upper()}] {final_text}"

        conversation_history[session_id].append({"role": "agent_rejected/edited", "text": final_text, "lang": original_lang})
        return {"message": "Feedback processed. Further action might be required.", "final_response": final_text}

# --- 5. Gradio Interface for Human-in-the-Loop ---

def get_pending_drafts():
    """Gradio function to retrieve and display pending drafts."""
    if not current_hil_drafts:
        return "No pending drafts for review.", "", "", "", "", False
    
    session_id = next(iter(current_hil_drafts)) # Get the first pending session
    draft_data = current_hil_drafts[session_id]
    
    return (
        f"Session ID: {session_id}",
        draft_data["original_query"],
        draft_data["detected_language"],
        draft_data["translated_query_en"],
        draft_data["draft_response_en"],
        True # Enable submission buttons
    )

def submit_hil_review(session_info: str, original_query: str, detected_language: str, translated_query: str, drafted_response: str, agent_id: str, feedback_action: str, edited_response: str):
    """Gradio function to submit human review feedback."""
    if not session_info or not agent_id:
        return "Please load a draft and enter your Agent ID to submit feedback."
    
    session_id = session_info.split(": ")[1] # Extract session ID from the display string
    
    approved = (feedback_action == "Approve")
    text_to_send = edited_response if edited_response else drafted_response

    # Simulate calling the FastAPI /feedback endpoint
    # In a real Gradio app, you might use requests.post to your own FastAPI endpoint
    # For this single-file demo, we'll directly call the internal logic
    
    # Create a mock HILFeedback object
    mock_feedback = HILFeedback(
        session_id=session_id,
        agent_id=agent_id,
        approved=approved,
        edited_text=edited_response if approved and edited_response else None # Only send edited if approved or explicitly edited
    )

    try:
        # Directly call the internal feedback processing logic
        response = human_feedback(mock_feedback)
        current_hil_drafts.pop(session_id, None) # Remove after processing
        return f"Feedback submitted successfully: {response.get('message', '')} Final Response: {response.get('final_response', '')}"
    except HTTPException as e:
        return f"Error submitting feedback: {e.detail}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

with gr.Blocks() as hil_interface:
    gr.Markdown("# Human-in-the-Loop Review for Customer Support AI")
    with gr.Row():
        agent_id_input = gr.Textbox(label="Agent ID", placeholder="Enter your agent ID", scale=1)
        load_draft_btn = gr.Button("Load Next Draft", scale=0)
    
    session_info_output = gr.Textbox(label="Session Information", interactive=False)
    original_query_output = gr.Textbox(label="Original Customer Query", interactive=False)
    detected_lang_output = gr.Textbox(label="Detected Language", interactive=False)
    translated_query_output = gr.Textbox(label="Translated Query (English)", interactive=False)
    drafted_response_output = gr.Textbox(label="AI Drafted Response (English)", interactive=False)

    edited_response_input = gr.Textbox(label="Edit Response (Optional)", placeholder="Make any necessary edits here...")

    with gr.Row():
        approve_btn = gr.Button("Approve & Finalize", variant="primary", enabled=False)
        reject_btn = gr.Button("Reject / Need More Info", variant="secondary", enabled=False)
    
    feedback_status_output = gr.Textbox(label="Feedback Status", interactive=False)

    load_draft_btn.click(
        get_pending_drafts,
        inputs=[],
        outputs=[
            session_info_output,
            original_query_output,
            detected_lang_output,
            translated_query_output,
            drafted_response_output,
            approve_btn, # Enable button if draft loaded
            reject_btn # Enable button if draft loaded
        ]
    )

    approve_btn.click(
        submit_hil_review,
        inputs=[
            session_info_output,
            original_query_output,
            detected_lang_output,
            translated_query_output,
            drafted_response_output,
            agent_id_input,
            gr.State("Approve"),
            edited_response_input
        ],
        outputs=feedback_status_output
    )
    
    reject_btn.click(
        submit_hil_review,
        inputs=[
            session_info_output,
            original_query_output,
            detected_lang_output,
            translated_query_output,
            drafted_response_output,
            agent_id_input,
            gr.State("Reject"),
            edited_response_input
        ],
        outputs=feedback_status_output
    )

# Mount Gradio app
app = gr.mount_gradio_app(app, hil_interface, path="/hil_review")

# --- 6. Main Execution Block --- 

if __name__ == "__main__":
    # To run both FastAPI and Gradio, we start FastAPI with uvicorn and Gradio mounted.
    # For a fully separate Gradio app, you'd run hil_interface.launch() in a separate process.
    # Here, Gradio is part of the FastAPI application structure.
    print("\n--- Multilingual AI Customer Support Agent --- ")
    print("FastAPI server running on http://127.0.0.1:8000")
    print("Gradio Human-in-the-Loop UI available at http://127.00.1:8000/hil_review")
    print("\nTo test, make a POST request to http://127.0.0.1:8000/query with a JSON body like:")
    print("{\"session_id\": \"user123\", \"text\": \"¿Cuál es su política de envío?\"}")
    print("or {\"session_id\": \"user456\", \"text\": \"How do I return a damaged item? This is urgent and complex.\"}")
    print("Queries flagged for human review will appear in the Gradio interface.")

    # Run FastAPI using uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

