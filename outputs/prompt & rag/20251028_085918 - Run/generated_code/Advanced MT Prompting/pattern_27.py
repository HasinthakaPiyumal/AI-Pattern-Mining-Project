import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

class MedicalTranslator:
    def __init__(self, 
                 mt_model_name="Helsinki-NLP/opus-mt-es-en", 
                 llm_model_name="google/gemma-2b-it",
                 exemplar_embeddings_path="medical_exemplar_embeddings.npy",
                 medical_ontology=None,
                 exemplar_texts=None):

        self.mt_pipeline = pipeline("translation", model=mt_model_name, tokenizer=mt_model_name)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.llm_pipeline = pipeline("text-generation", model=self.llm_model, tokenizer=self.llm_tokenizer, max_new_tokens=512)

        if medical_ontology is None:
            self.medical_ontology = {
                "hypertension": "High blood pressure, a condition in which the force of the blood against the artery walls is too high.",
                "diabetes": "A disease in which the body's ability to produce or respond to the hormone insulin is impaired, resulting in abnormal metabolism of carbohydrates and elevated levels of glucose in the blood and urine.",
                "myocardial infarction": "A heart attack; a sudden and sometimes fatal occurrence of coronary thrombosis, typically resulting in the death of part of the heart muscle."
            }
        else:
            self.medical_ontology = medical_ontology

        if exemplar_texts is None:
            self.exemplar_texts = [
                ("El paciente presenta hipertensión y diabetes tipo 2.", "The patient presents with hypertension and type 2 diabetes."),
                ("Se recomienda una biopsia para descartar malignidad.", "A biopsy is recommended to rule out malignancy."),
                ("Dolor torácico agudo con irradiación al brazo izquierdo.", "Acute chest pain radiating to the left arm.")
            ]
        else:
            self.exemplar_texts = exemplar_texts

        # Simulate embeddings for exemplars (in a real system, use sentence-transformers)
        self.exemplar_embeddings = self._generate_mock_embeddings(self.exemplar_texts)

    def _generate_mock_embeddings(self, texts):
        # In a real system, use a model like sentence-transformers to get actual embeddings
        # For this example, we'll create random vectors as a placeholder.
        np.random.seed(42) # for reproducibility
        return np.random.rand(len(texts), 768) # 768 is a common embedding dimension

    def _get_text_embedding(self, text):
        # Simulate getting an embedding for a given text
        # In a real system, use the same embedding model as for exemplars
        np.random.seed(hash(text) % (2**32 - 1)) # Simple hash-based seed for 'unique' mock embedding
        return np.random.rand(768)

    def _retrieve_exemplars(self, query_text, top_k=2):
        query_embedding = self._get_text_embedding(query_text)
        similarities = np.dot(self.exemplar_embeddings, query_embedding) / \
                       (np.linalg.norm(self.exemplar_embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.exemplar_texts[i] for i in top_k_indices]

    def _chunk_text(self, text, max_chunk_length=500):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_chunk_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _preprocess_input(self, text, source_lang='es'):
        if source_lang != 'en':
            # Translate non-English to English using an MT model
            translated_text = self.mt_pipeline(text, src_lang=source_lang, tgt_lang='en')[0]['translation_text']
            return translated_text
        return text

    def _augment_prompt(self, preprocessed_text, retrieved_exemplars):
        prompt_parts = [f"Translate the following medical text into highly accurate and lexically precise English. Consider medical terminology and context."]
        
        # Add ontology definitions
        for term, definition in self.medical_ontology.items():
            if term.lower() in preprocessed_text.lower():
                prompt_parts.append(f"Definition of '{term}': {definition}")
        
        # Add exemplars
        if retrieved_exemplars:
            prompt_parts.append("Here are some examples of medical translations for reference:")
            for src, tgt in retrieved_exemplars:
                prompt_parts.append(f"Source: {src}\nTarget: {tgt}")
        
        prompt_parts.append(f"Original medical text to translate: {preprocessed_text}\nAccurate English Translation:")
        return "\n\n".join(prompt_parts)

    def _translate_with_llm(self, prompt):
        # Ensure the LLM pipeline generates text that continues the prompt rather than rewriting it
        # num_return_sequences=1 ensures a single best generation
        generated_text = self.llm_pipeline(prompt, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)[0]['generated_text']
        
        # The LLM might repeat the prompt or add conversational filler. Extract the actual translation part.
        # This is a heuristic and might need fine-tuning based on LLM behavior.
        if "Accurate English Translation:" in generated_text:
            translation = generated_text.split("Accurate English Translation:", 1)[1].strip()
            return translation
        return generated_text.strip() # Fallback

    def translate_document(self, document_text, source_lang='es', max_refinement_steps=2):
        print(f"\n--- Starting translation for document (Language: {source_lang}) ---")
        
        chunks = self._chunk_text(document_text)
        translated_chunks = []

        for i, chunk in enumerate(chunks):
            print(f"\nTranslating chunk {i+1}/{len(chunks)}...")
            
            # 1. Input Pre-processing
            preprocessed_chunk = self._preprocess_input(chunk, source_lang)
            print(f"Preprocessed (to English): {preprocessed_chunk[:100]}...")

            # 2. Prompt Augmentation (RAG)
            retrieved_exemplars = self._retrieve_exemplars(preprocessed_chunk)
            augmented_prompt = self._augment_prompt(preprocessed_chunk, retrieved_exemplars)
            print(f"Augmented prompt generated (length: {len(augmented_prompt)})...")

            # 3. Core Translation with LLM & Iterative Refinement
            current_translation = ""
            for step in range(max_refinement_steps + 1):
                if step == 0:
                    print(f"Initial LLM translation attempt...")
                    llm_output = self._translate_with_llm(augmented_prompt)
                    current_translation = llm_output
                else:
                    print(f"Refinement step {step}...")
                    # Self-correction prompt
                    refinement_prompt = f"The previous translation was: '{current_translation}'. Review the original text and refine the translation for accuracy, consistency, and medical precision. Original text: '{preprocessed_chunk}'\nRefined English Translation:"
                    llm_output = self._translate_with_llm(refinement_prompt)
                    
                    # Simulate human-in-the-loop: if the LLM output is not significantly better,
                    # we might stick to the previous one or ask for specific changes.
                    # For this example, we simply update with the new LLM output.
                    if len(llm_output) > 10 and llm_output != current_translation: # Basic check for meaningful output
                        current_translation = llm_output
                    else:
                        print("Refinement did not yield significant improvement or was empty. Skipping further refinements for this chunk.")
                        break
                print(f"Translation after step {step}: {current_translation[:100]}...")
                
            translated_chunks.append(current_translation)

        final_translation = " ".join(translated_chunks)
        print("\n--- Document translation complete ---")
        print(f"Final Translation: {final_translation[:500]}...")
        return final_translation

# Example Usage:
if __name__ == "__main__":
    translator = MedicalTranslator()

    spanish_medical_text = """
    El paciente, de 65 años, fue admitido con dolor torácico agudo y disnea. 
    El electrocardiograma mostró elevación del segmento ST en derivaciones inferiores. 
    Se diagnosticó infarto agudo de miocardio. 
    Se administró terapia trombolítica y se estabilizó su condición. 
    Se recomienda seguimiento cardiológico y manejo de factores de riesgo como hipertensión y diabetes.
    """

    # Test with Spanish medical text
    translated_doc = translator.translate_document(spanish_medical_text, source_lang='es')
    print(f"\nOriginal Spanish:\n{spanish_medical_text}")
    print(f"\nTranslated English:\n{translated_doc}")

    # Test with an English medical text to show pre-processing skip
    english_medical_text = """
    The patient presented with a severe headache and photophobia. 
    A lumbar puncture was performed to check for meningitis. 
    Results are pending. Continue symptomatic treatment and monitor vital signs.
    """
    translated_doc_en = translator.translate_document(english_medical_text, source_lang='en')
    print(f"\nOriginal English:\n{english_medical_text}")
    print(f"\nTranslated English (processed):\n{translated_doc_en}")
