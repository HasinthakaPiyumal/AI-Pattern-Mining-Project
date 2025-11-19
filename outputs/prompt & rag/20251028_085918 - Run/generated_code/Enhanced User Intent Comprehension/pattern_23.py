import gradio as gr
from PIL import Image
import os
import io

# Ensure transformers is installed or provide a note
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    # If transformers is not installed, provide a dummy implementation
    print("Transformers library not found. LLM functionality will be simulated.")
    class DummyTokenizer:
        def encode(self, text, return_tensors=None, padding=True, truncation=True): return [1, 2]
        def decode(self, tokens): return "simulated text"
        def pad(self, *args, **kwargs): return {"input_ids": [[1,2]], "attention_mask": [[1,1]]}
        @property
        def eos_token_id(self): return 0
        @property
        def pad_token(self): return "<|endoftext|>"
        @pad_token.setter
        def pad_token(self, value): pass

    class DummyModel:
        def __call__(self, *args, **kwargs): return [{"generated_text": "Simulated LLM response for: " + args[0]}]

    def AutoTokenizer_from_pretrained(*args, **kwargs): return DummyTokenizer()
    def AutoModelForCausalLM_from_pretrained(*args, **kwargs): return DummyModel()
    def pipeline(*args, **kwargs):
        def dummy_pipeline(text):
            return [{"generated_text": f"Simulated LLM response for: {text}"}]
        return dummy_pipeline


# --- 1. Simulated External Tools ---

# Simulate Speech-to-Text
def simulate_speech_to_text(audio_file) -> str:
    """
    Simulates speech-to-text conversion. In a real app, this would use a library
    like SpeechRecognition or a cloud API (e.g., Google Cloud Speech-to-Text).
    For this demo, we'll just return a placeholder text based on presence of audio.
    """
    if audio_file is None:
        return ""
    # In a real scenario, you'd process the audio file content here.
    return "User said: 'My internet is not working, and I need a refund.'" # Example output for any audio input

# Simulate Image Analysis
def simulate_image_analysis(image_path: str) -> str:
    """
    Simulates image analysis. In a real app, this would use a Vision Transformer
    or a cloud Vision API to describe the image or detect objects/issues.
    For this demo, we'll return a placeholder based on file presence.
    """
    if image_path is None:
        return ""
    try:
        img = Image.open(image_path)
        # Simulate simple analysis based on image properties or just a generic message
        if img.width > 800 and img.height > 600:
            return "Image analysis: User provided a high-resolution screenshot. Appears to show an error message 'Error 404 - Page Not Found'."
        else:
            return "Image analysis: User provided an image, possibly a product photo or a smaller screenshot."
    except Exception as e:
        return f"Image processing failed: {e}"


# Simulate Machine Translation
def simulate_translate(text: str, dest_lang: str = 'en') -> str:
    """
    Simulates machine translation. For simplicity, this demo only 'translates'
    if a specific keyword is present, otherwise assumes English.
    In a real app, an actual translation library or API would be used.
    """
    if not text:
        return ""
    # Simple simulation: if text contains "hola", assume it's Spanish and "translate"
    if "hola" in text.lower() and dest_lang == 'en':
        print(f"Simulating translation from Spanish to English for: '{text}'")
        return text.lower().replace("hola", "hello") + " (translated from Spanish)"
    if "bonjour" in text.lower() and dest_lang == 'en':
        print(f"Simulating translation from French to English for: '{text}'")
        return text.lower().replace("bonjour", "hello") + " (translated from French)"
    return text # Assume it's already in the target language or no specific keyword was found

# --- 2. LLM Core for Intent Recognition & Dialogue Management ---

# Load a smaller, general-purpose LLM
# For a production system, this would be a fine-tuned, more capable model.
# Using a small model for demonstration purposes to avoid large downloads/memory.
model_name = "facebook/opt-125m" # A small, pre-trained LLM
llm_tokenizer = None
llm_model = None
llm_pipeline = None

def load_llm_components():
    global llm_tokenizer, llm_model, llm_pipeline
    if llm_tokenizer is None and AutoTokenizer != DummyTokenizer: # Check if transformers is available
        print(f"Loading LLM tokenizer: {model_name}")
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_tokenizer.pad_token = llm_tokenizer.eos_token # For generation
        llm_tokenizer.padding_side = "left" # For generation

    if llm_model is None and AutoModelForCausalLM != DummyModel: # Check if transformers is available
        print(f"Loading LLM model: {model_name}")
        llm_model = AutoModelForCausalLM.from_pretrained(model_name)

    if llm_pipeline is None:
        print(f"Creating LLM pipeline...")
        # Use the dummy pipeline if transformers is not available
        if pipeline != type: # Check if pipeline is the actual function or the dummy one
            llm_pipeline = pipeline(
                "text-generation",
                model=llm_model,
                tokenizer=llm_tokenizer,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                eos_token_id=llm_tokenizer.eos_token_id
            )
        else: # Fallback to dummy pipeline
            llm_pipeline = pipeline("dummy")
    return llm_pipeline

def process_llm_query(full_context: str) -> str:
    """
    Processes the combined multimodal input using the LLM for intent recognition
    and response generation, including ambiguity clarification.
    """
    llm = load_llm_components()

    # Simple prompt engineering for intent recognition and dialogue management
    prompt = (
        f"You are a helpful and polite customer support agent. Your goal is to understand "
        f"the user's issue and provide a clear solution or ask for clarification. "
        f"Based on the following user input, identify the core issue and provide a concise response. "
        f"If the information is ambiguous, ask a clarifying question. "
        f"User input:\n---\n{full_context}\n---\n\nAgent response:"
    )

    print(f"\nSending to LLM:\n{prompt}\n")

    try:
        generated_text = llm(prompt)[0]['generated_text']

        # Post-process to extract only the agent's response, if the model continues the prompt
        # This part requires careful handling depending on the LLM's behavior
        if prompt in generated_text:
            response = generated_text.replace(prompt, "").strip()
            # Try to cut at the first newline or sentence end for conciseness
            first_sentence_end = response.find('.')
            if first_sentence_end != -1 and first_sentence_end < 100:
                response = response[:first_sentence_end+1]
            else:
                response = ' '.join(response.split()[:50]) # Limit words
            return response + "..." if len(response) > 100 else response
        else:
            return generated_text.strip() # If model generated something different

    except Exception as e:
        print(f"LLM processing failed: {e}")
        return "I apologize, but I encountered an error trying to understand your request. Could you please rephrase?"

# --- 3. Main Multimodal Customer Support Agent Logic ---

def customer_support_agent(text_input: str, audio_file, image_file) -> str:
    """
    Integrates all components to process multimodal user inputs.
    """
    print(f"\n--- New Interaction ---")

    processed_inputs = []

    # Process text input (translate if necessary)
    if text_input:
        translated_text = simulate_translate(text_input, dest_lang='en')
        processed_inputs.append(f"Text input: {translated_text}")

    # Process audio input (STT)
    if audio_file:
        audio_text = simulate_speech_to_text(audio_file)
        if audio_text:
            # Assume STT output is already in English or translate it
            processed_inputs.append(f"Speech input: {simulate_translate(audio_text, dest_lang='en')}")

    # Process image input (Analysis)
    if image_file:
        image_analysis_text = simulate_image_analysis(image_file)
        if image_analysis_text:
            processed_inputs.append(f"Image input: {image_analysis_text}")

    # Combine all processed inputs for the LLM
    full_context = "\n".join(processed_inputs)
    if not full_context:
        return "Please provide some input (text, audio, or image) so I can assist you."

    print(f"\nCombined context for LLM:\n{full_context}")

    # Get LLM response
    llm_response = process_llm_query(full_context)

    # Simple post-processing to indicate potential clarification (can be enhanced with LLM parsing)
    if "clarify" in llm_response.lower() or "what" in llm_response.lower() or "how" in llm_response.lower():
        return llm_response + "\n\n(The agent is seeking clarification for ambiguity.)"

    return llm_response

# --- 4. Gradio Interface ---

# Load LLM components on startup for efficiency
load_llm_components()

# Define Gradio Interface
iface = gr.Interface(
    fn=customer_support_agent,
    inputs=[
        gr.Textbox(label="Text Input (e.g., 'My order is late' or 'Hola, mi producto no funciona')", placeholder="Type your query here..."),
        gr.Audio(type="filepath", label="Voice Input (upload a .wav file)", value=None),
        gr.Image(type="filepath", label="Image Input (e.g., an error screenshot)", tool="upload", value=None)
    ],
    outputs=gr.Textbox(label="Agent Response", lines=5),
    title="AI-Powered Multimodal Customer Support Agent",
    description=(
        "This agent processes text, voice, and image inputs to understand your request. "
        "It simulates speech-to-text, image analysis, and uses a language model for intent recognition and response generation. "
        "Try typing a question (even in basic Spanish/French), uploading an audio file, or an image (like a screenshot)."
        "\n\n**Note:** External tools and LLM are simulated or use small models for demonstration purposes. "
        "For actual voice/image recognition and robust translation, dedicated APIs/models would be used."
    ),
    examples=[
        ["My internet is down.", None, None],
        ["Hola, mi impresora no funciona.", None, None],
        [None, "audio_placeholder.wav", None], # You would need to provide an actual .wav file if running locally.
        ["I have an issue with my product.", None, "image_placeholder.png"], # You would need to provide an actual image file if running locally.
        ["The screen looks like this.", None, "image_placeholder_error.png"] # You would need to provide an actual image file if running locally.
    ]
)

# To run this Gradio app, save it as a .py file (e.g., multimodal_customer_agent.py)
# and execute `python multimodal_customer_agent.py`. 
# You may need to install libraries: `pip install gradio Pillow transformers`
# For the examples to work, replace 'audio_placeholder.wav', 'image_placeholder.png', 
# and 'image_placeholder_error.png' with actual files or comment out the examples.

if __name__ == "__main__":
    iface.launch()