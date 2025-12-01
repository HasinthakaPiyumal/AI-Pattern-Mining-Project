import nltk
import os
import time
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class DocumentChunker:
    def chunk_text(self, text: str, max_tokens: int) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0

        for sentence in sentences:
            # Approximate token count (words)
            sentence_tokens = len(sentence.split())
            if current_chunk_tokens + sentence_tokens <= max_tokens or not current_chunk:
                current_chunk.append(sentence)
                current_chunk_tokens += sentence_tokens
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_tokens = sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

class MedicalLLMTranslator:
    def __init__(self, client: OpenAI, llm_model: str):
        self.client = client
        self.llm_model = llm_model

    def _build_few_shot_prompt(self, chunk: str, few_shot_examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": "You are a highly accurate medical translator. Translate the provided medical text precisely, maintaining all medical terminology and context."}
        ]
        for example in few_shot_examples:
            messages.append({"role": "user", "content": example["source"]})
            messages.append({"role": "assistant", "content": example["target"]})
        
        messages.append({"role": "user", "content": chunk})
        return messages

    def translate_chunk(self, chunk: str, target_language: str, few_shot_examples: List[Dict[str, str]], retries: int = 3, delay: int = 2) -> str:
        prompt_messages = self._build_few_shot_prompt(chunk, few_shot_examples)
        
        for i in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=prompt_messages + [{"role": "system", "content": f"Translate the above medical text into {target_language}."}],
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if i < retries - 1:
                    time.sleep(delay * (i + 1))
                else:
                    raise e
        return ""

class ContextualIntegrator:
    def __init__(self, client: OpenAI, llm_model: str):
        self.client = client
        self.llm_model = llm_model

    def integrate_translations(self, original_chunks: List[str], translated_chunks: List[str], target_language: str, retries: int = 3, delay: int = 2) -> str:
        integration_prompt = """Review the original medical document chunks and their individual translations. 
Your task is to integrate these translated chunks into a single, cohesive, and contextually accurate final translation in {target_language}. 
Ensure consistent medical terminology, proper grammatical flow, resolve any repetitions, and maintain the overall meaning and tone of the original document. 
Do not omit any information from the original document.

Original Chunks:
"""
        for i, original_chunk in enumerate(original_chunks):
            integration_prompt += f"Chunk {i+1}:\n{original_chunk}\n\n"
        
        integration_prompt += "Translated Chunks (Individual):\n"""
        for i, translated_chunk in enumerate(translated_chunks):
            integration_prompt += f"Translated Chunk {i+1}:\n{translated_chunk}\n\n"

        integration_prompt += """Final Integrated Translation in {target_language}:"""

        for i in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert medical editor and translator, specializing in integrating translated segments into a flawless, coherent document."},
                        {"role": "user", "content": integration_prompt.format(target_language=target_language)}
                    ],
                    temperature=0.2
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if i < retries - 1:
                    time.sleep(delay * (i + 1))
                else:
                    raise e
        return ""

class MedicalDocumentTranslatorApp:
    def __init__(self, api_key: str, llm_model: str = "gpt-4-0125-preview"):
        self.client = OpenAI(api_key=api_key)
        self.chunker = DocumentChunker()
        self.translator = MedicalLLMTranslator(self.client, llm_model)
        self.integrator = ContextualIntegrator(self.client, llm_model)
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> List[Dict[str, str]]:
        return [
            {
                "source": "The patient presented with symptoms of acute myocardial infarction, including chest pain radiating to the left arm, dyspnea, and diaphoresis.",
                "target": "El paciente presentó síntomas de infarto agudo de miocardio, incluyendo dolor torácico que se irradiaba al brazo izquierdo, disnea y diaforesis."
            },
            {
                "source": "The histological examination revealed a poorly differentiated adenocarcinoma with lymph node metastasis.",
                "target": "El examen histológico reveló un adenocarcinoma pobremente diferenciado con metástasis en los ganglios linfáticos."
            }
        ]

    def translate_document(self, document_text: str, target_language: str = "Spanish", max_chunk_tokens: int = 500) -> str:
        original_chunks = self.chunker.chunk_text(document_text, max_chunk_tokens)
        translated_chunks = []

        for i, original_chunk in enumerate(original_chunks):
            translated_chunk = self.translator.translate_chunk(original_chunk, target_language, self.few_shot_examples)
            translated_chunks.append(translated_chunk)
        
        final_translation = self.integrator.integrate_translations(original_chunks, translated_chunks, target_language)
        return final_translation

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")
    else:
        app = MedicalDocumentTranslatorApp(api_key=api_key)

        sample_medical_document = (
            "The patient, a 65-year-old male, was admitted to the emergency department with severe acute abdominal pain. "
            "Physical examination revealed tenderness in the right lower quadrant with rebound tenderness. "
            "Laboratory tests showed leukocytosis with a shift to the left. "
            "An abdominal CT scan was performed, which indicated appendicitis. "
            "Surgical consultation was obtained, and an appendectomy was scheduled. "
            "Postoperative recovery was uneventful, and the patient was discharged on the third day with oral antibiotics. "
            "Follow-up was arranged for two weeks later to check on wound healing and overall recovery. "
            "The pathology report confirmed acute suppurative appendicitis with no evidence of perforation. "
            "The patient was advised to continue a light diet for a week and avoid strenuous activities."
        )

        print("Original Medical Document:\n" + sample_medical_document + "\n")

        try:
            translated_document = app.translate_document(sample_medical_document, target_language="Spanish", max_chunk_tokens=100)
            print("Translated Medical Document (Spanish):\n" + translated_document)
        except Exception as e:
            print(f"An error occurred during translation: {e}")