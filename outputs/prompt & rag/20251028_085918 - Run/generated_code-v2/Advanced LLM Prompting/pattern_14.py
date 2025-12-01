from flask import Flask, request, jsonify, render_template
from transformers import MarianTokenizer, MarianMTModel
import torch

app = Flask(__name__)

# Cache for translation models
translation_models = {}

def get_translator(src_lang, tgt_lang):
    model_key = f"{src_lang}-{tgt_lang}"
    if model_key not in translation_models:
        try:
            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            translation_models[model_key] = (tokenizer, model)
            print(f"Loaded translation model: {model_name}")
        except Exception as e:
            print(f"Error loading translation model {model_name}: {e}")
            return None, None
    return translation_models[model_key]

def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text
    
    tokenizer, model = get_translator(src_lang, tgt_lang)
    if tokenizer is None or model is None:
        return f"[Translation Error: Model not found for {src_lang}-{tgt_lang}] {text}"
    
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"Error during translation from {src_lang} to {tgt_lang}: {e}")
        return f"[Translation Failed from {src_lang} to {tgt_lang}] {text}"

def simulate_generative_ai_core(english_query):
    # This is a placeholder for an actual English-centric Generative AI model.
    # In a real application, this would integrate with an LLM like GPT, Gemini, etc.
    print(f"Simulating GenAI for English query: {english_query}")
    if "hello" in english_query.lower() or "hi" in english_query.lower():
        return "Hello! How can I assist you with your e-commerce needs today?"
    elif "order status" in english_query.lower():
        return "To check your order status, please provide your order number."
    elif "return policy" in english_query.lower():
        return "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging. Please visit our returns page for more details."
    elif "shipping" in english_query.lower():
        return "Shipping times vary based on your location and the selected shipping method. Standard shipping usually takes 5-7 business days."
    else:
        return f"I received your query: '{english_query}'. How else can I help you?"

@app.route("/")
def index():
    # Frontend for user interaction
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multilingual Customer Support Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
            .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { text-align: center; color: #333; }
            .chat-box { border: 1px solid #ddd; padding: 15px; min-height: 200px; max-height: 400px; overflow-y: auto; margin-bottom: 15px; border-radius: 5px; background-color: #e9e9e9; }
            .message { margin-bottom: 10px; }
            .user-message { text-align: right; color: #007bff; }
            .bot-message { text-align: left; color: #28a745; }
            input[type="text"] { width: calc(100% - 100px); padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-right: 10px; }
            select { padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-right: 10px; }
            button { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>E-commerce Chatbot</h1>
            <div class="chat-box" id="chat-box"></div>
            <div>
                <input type="text" id="user-input" placeholder="Type your message...">
                <select id="language-select">
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="it">Italian</option>
                    <option value="pt">Portuguese</option>
                </select>
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const userInput = document.getElementById('user-input');
                const languageSelect = document.getElementById('language-select');
                const chatBox = document.getElementById('chat-box');
                const query = userInput.value;
                const lang = languageSelect.value;

                if (!query.trim()) return;

                chatBox.innerHTML += `<div class="message user-message">You (${lang}): ${query}</div>`;
                userInput.value = '';

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: query, lang: lang })
                    });
                    const data = await response.json();
                    chatBox.innerHTML += `<div class="message bot-message">Bot (${lang}): ${data.response}</div>`;
                } catch (error) {
                    console.error('Error:', error);
                    chatBox.innerHTML += `<div class="message bot-message" style="color: red;">Error: Could not get response.</div>`;
                }
                chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
            }

            // Allow sending message with Enter key
            document.getElementById('user-input').addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")
    source_lang = data.get("lang", "en") # Default to English if not provided

    if not user_query:
        return jsonify({"response": "Please provide a query."}), 400

    print(f"Received query: '{user_query}' in language: {source_lang}")

    # 1. Translate non-English input to English
    if source_lang != "en":
        english_query = translate_text(user_query, src_lang=source_lang, tgt_lang="en")
        print(f"Translated to English: {english_query}")
    else:
        english_query = user_query

    # 2. Process with English-centric Generative AI core
    english_response = simulate_generative_ai_core(english_query)
    print(f"GenAI English response: {english_response}")

    # 3. Translate English response back to original language (if needed)
    if source_lang != "en":
        final_response = translate_text(english_response, src_lang="en", tgt_lang=source_lang)
        print(f"Translated back to {source_lang}: {final_response}")
    else:
        final_response = english_response

    return jsonify({"response": final_response})


if __name__ == "__main__":
    # Initial model loading for a few common pairs to warm up
    # These models will be downloaded on first request for a given language pair
    # It's recommended to pre-download models or run this in an environment with cached models
    # get_translator("es", "en")
    # get_translator("en", "es")
    # get_translator("fr", "en")
    # get_translator("en", "fr")
    app.run(debug=True, port=5000)
