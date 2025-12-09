import speech_recognition as sr
from PIL import Image
from googletrans import Translator
from pydub import AudioSegment
import os
import pathlib

class SpeechToTextModule:
    def transcribe_audio(self, audio_path):
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition service error: {e}"

class ImageAnalysisModule:
    def analyze_image(self, image_path):
        try:
            img = Image.open(image_path)
            filename = pathlib.Path(image_path).stem.lower()
            if "defect" in filename or "broken" in filename:
                return "Image shows a product defect."
            elif "shipping" in filename or "package" in filename:
                return "Image related to shipping or packaging."
            elif "return" in filename:
                return "Image related to a product return."
            else:
                return "Image content is general, no specific issue detected from filename."
        except FileNotFoundError:
            return "Image file not found."
        except Exception as e:
            return f"Error analyzing image: {e}"

class MachineTranslationModule:
    def translate_text(self, text, dest_lang='en'):
        translator = Translator()
        try:
            translated = translator.translate(text, dest=dest_lang)
            return translated.text
        except Exception as e:
            return f"Translation error: {e}"

class SimulatedLLM:
    def generate_response(self, context):
        context_lower = context.lower()
        if "product defect" in context_lower or "broken" in context_lower:
            return "I understand you have an issue with a product defect. Please provide your order number so we can initiate a return or replacement."
        elif "shipping" in context_lower or "package" in context_lower:
            return "Regarding your shipping query, could you please share your tracking number or order ID?"
        elif "return" in context_lower:
            return "You're looking to return an item. Please confirm the item and your order details."
        elif "general, no specific issue" in context_lower:
            return "I'm sorry, I couldn't identify a specific issue from the image. Can you describe your problem in more detail?"
        elif "hello" in context_lower or "hi" in context_lower:
            return "Hello! How can I assist you today?"
        else:
            return "Thank you for reaching out. Please provide more details about your concern so I can assist you better."

class SmartCustomerSupportAssistant:
    def __init__(self):
        self.stt_module = SpeechToTextModule()
        self.image_module = ImageAnalysisModule()
        self.translation_module = MachineTranslationModule()
        self.llm = SimulatedLLM()

    def process_input(self, input_data, input_type=None, source_lang='auto'):
        processed_info = []

        if input_type == "audio" or (input_type is None and isinstance(input_data, (str, pathlib.Path)) and (str(input_data).endswith('.wav') or str(input_data).endswith('.mp3'))):
            audio_text = self.stt_module.transcribe_audio(input_data)
            processed_info.append(f"Transcribed audio: {audio_text}")
            if source_lang != 'en' and audio_text and not audio_text.startswith("Could not understand"):
                translated_audio_text = self.translation_module.translate_text(audio_text, dest_lang='en')
                if not translated_audio_text.startswith("Translation error"):
                    processed_info.append(f"Translated audio: {translated_audio_text}")
                else:
                    processed_info.append(f"Translation failed for audio: {translated_audio_text}")
            elif audio_text and not audio_text.startswith("Could not understand"):
                processed_info.append(f"Audio (English): {audio_text}")
            else:
                processed_info.append(f"Audio processing failed: {audio_text}")

        elif input_type == "image" or (input_type is None and isinstance(input_data, (str, pathlib.Path)) and (str(input_data).endswith(('.jpg', '.png', '.jpeg')))):
            image_description = self.image_module.analyze_image(input_data)
            processed_info.append(f"Image analysis: {image_description}")

        elif input_type == "text" or (input_type is None and isinstance(input_data, str)):
            if source_lang != 'en':
                translated_text = self.translation_module.translate_text(input_data, dest_lang='en')
                if not translated_text.startswith("Translation error"):
                    processed_info.append(f"Translated text: {translated_text}")
                else:
                    processed_info.append(f"Translation failed for text: {translated_text}")
            else:
                processed_info.append(f"Received text: {input_data}")

        context_for_llm = ". ".join(processed_info)
        if not context_for_llm:
            context_for_llm = "No discernible input detected."

        response = self.llm.generate_response(context_for_llm)
        return response

if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()

    # Setup for demonstration
    temp_dir = pathlib.Path("temp_multimodal_demo")
    temp_dir.mkdir(exist_ok=True)

    # 1. Simulate an audio input (e.g., a customer voicemail)
    dummy_audio_path = temp_dir / "product_defect_query.wav"
    AudioSegment.silent(duration=500).overlay(AudioSegment.from_wav(pathlib.Path(__file__).parent / "_dummy_hello.wav")).export(dummy_audio_path, format="wav")
    # To run this line, you'd need a simple _dummy_hello.wav file in the same directory
    # For this demonstration, we'll simulate a transcription for simplicity if no dummy wav is available.

    print("\n--- Test Case 1: Audio Input (Simulated) ---")
    # If _dummy_hello.wav doesn't exist, this will likely fail to transcribe, showing error handling.
    # In a real scenario, you'd provide an actual audio file.
    audio_response = assistant.process_input(dummy_audio_path, input_type="audio")
    print(f"Assistant Response (Audio): {audio_response}")
    os.remove(dummy_audio_path)

    # 2. Simulate an image input (e.g., customer sent a photo of a damaged product)
    dummy_image_path_defect = temp_dir / "product_defect_photo.jpg"
    Image.new('RGB', (100, 100), color = 'red').save(dummy_image_path_defect)

    print("\n--- Test Case 2: Image Input (Defect) ---")
    image_response_defect = assistant.process_input(dummy_image_path_defect, input_type="image")
    print(f"Assistant Response (Image - Defect): {image_response_defect}")
    os.remove(dummy_image_path_defect)

    dummy_image_path_shipping = temp_dir / "shipping_box_issue.png"
    Image.new('RGB', (100, 100), color = 'blue').save(dummy_image_path_shipping)

    print("\n--- Test Case 3: Image Input (Shipping) ---")
    image_response_shipping = assistant.process_input(dummy_image_path_shipping, input_type="image")
    print(f"Assistant Response (Image - Shipping): {image_response_shipping}")
    os.remove(dummy_image_path_shipping)

    # 3. Simulate text inputs (English and Foreign Language)
    print("\n--- Test Case 4: English Text Input ---")
    text_response_english = assistant.process_input("I want to know about my order status.", input_type="text", source_lang='en')
    print(f"Assistant Response (English Text): {text_response_english}")

    print("\n--- Test Case 5: French Text Input ---")
    text_response_french = assistant.process_input("Bonjour, mon colis n'est pas arrivé.", input_type="text", source_lang='fr')
    print(f"Assistant Response (French Text): {text_response_french}")

    print("\n--- Test Case 6: Combined Text from Audio (Simulated transcription of 'My product is broken') ---")
    # Simulating the scenario where audio was transcribed as 'My product is broken'
    simulated_audio_transcription = "My product is broken."
    combined_response = assistant.process_input(simulated_audio_transcription, input_type="text", source_lang='en')
    print(f"Assistant Response (Simulated Audio Text): {combined_response}")

    # Clean up temp directory
    temp_dir.rmdir()

    print("\nDemonstration Complete.")
