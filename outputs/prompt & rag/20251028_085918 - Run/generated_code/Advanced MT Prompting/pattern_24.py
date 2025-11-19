
class MMCAService:
    def __init__(self):
        self.medical_terms_db = {
            "fever": "An abnormally high body temperature, usually accompanied by shivering, headache, and in severe instances, delirium.",
            "cough": "A sudden, forceful expulsion of air from the lungs, often to clear the throat or airways of irritants or fluids.",
            "headache": "A continuous pain in the head.",
            "diabetes": "A chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces.",
            "hypertension": "A medical condition in which the blood pressure in the arteries is persistently elevated."
        }
        self.medical_exemplars_db = [
            {"condition": "flu symptoms", "english": "Symptoms of the flu often include fever, body aches, cough, and a sore throat.", "swahili": "Dalili za mafua mara nyingi hujumuisha homa, maumivu ya mwili, kikohozi, na koo kuuma.", "mandarin": "流感症状通常包括发烧、身体疼痛、咳嗽和喉咙痛。"},
            {"condition": "diabetes management", "english": "Managing diabetes involves monitoring blood sugar, taking medication, and adopting a healthy diet.", "swahili": "Kudhibiti kisukari kunahusisha kufuatilia sukari ya damu, kutumia dawa, na kufuata lishe bora.", "mandarin": "糖尿病管理包括监测血糖、服用药物和采用健康饮食。"}
        ]
        self.language_map = {
            "swahili": "Swahili",
            "mandarin": "Mandarin",
            "hindi": "Hindi",
            "english": "English"
        }
        self.translation_rules = {
            ("swahili", "english"): {"homa": "fever", "kikohozi": "cough", "maumivu ya kichwa": "headache", "nilisikia": "I felt", "kila mahali": "all over"},
            ("english", "swahili"): {"fever": "homa", "cough": "kikohozi", "headache": "maumivu ya kichwa", "I felt": "nilihisi", "all over": "kila mahali"},
            ("mandarin", "english"): {"发烧": "fever", "咳嗽": "cough", "头痛": "headache"},
            ("english", "mandarin"): {"fever": "发烧", "cough": "咳嗽", "headache": "头痛"}
        }

    def _mock_stt(self, audio_input, lang):
        print(f"[STT] Transcribing audio for {lang}...")
        # Simulate STT by returning a predefined text based on lang
        if lang == "swahili":
            return "Nilisikia homa na kikohozi. Maumivu ya kichwa kila mahali."
        elif lang == "mandarin":
            return "我感觉发烧和咳嗽。头痛无处不在。"
        return audio_input # Assume English if not specified

    def _mock_translate(self, text, source_lang, target_lang, domain="medical"):
        print(f"[MT] Translating from {source_lang} to {target_lang}...")
        # Simple rule-based translation simulation
        translated_text = text
        rules = self.translation_rules.get((source_lang, target_lang), {})
        for original, translation in rules.items():
            translated_text = translated_text.replace(original, translation)
        return translated_text

    def _mock_ner(self, text):
        print("[NER] Extracting medical terms...")
        found_terms = []
        for term in self.medical_terms_db.keys():
            if term in text.lower():
                found_terms.append(term)
        return found_terms

    def _mock_medical_dictionary_lookup(self, term):
        print(f"[Dictionary] Looking up '{term}'...")
        return self.medical_terms_db.get(term, f"Definition for {term} not found.")

    def _mock_retrieve_exemplars(self, query_text, target_lang):
        print("[Exemplar Retrieval] Searching for relevant exemplars...")
        relevant_exemplars = []
        for exemplar in self.medical_exemplars_db:
            if any(keyword in query_text.lower() for keyword in exemplar["condition"].split()):
                relevant_exemplars.append(exemplar[target_lang])
        return relevant_exemplars

    def _construct_prompt(self, pivot_text, medical_definitions, exemplars):
        print("[Prompt Augmentation] Constructing augmented prompt...")
        prompt_parts = [
            "Based on the following medical context, provide a precise and medically accurate translation.",
            f"Patient statement (pivot language): {pivot_text}"
        ]
        if medical_definitions:
            prompt_parts.append("Medical Term Definitions:")
            for term, definition in medical_definitions.items():
                prompt_parts.append(f"- {term}: {definition}")
        if exemplars:
            prompt_parts.append("Relevant Medical Exemplars:")
            for ex in exemplars:
                prompt_parts.append(f"- {ex}")
        prompt_parts.append("Please translate the patient's statement considering all provided context.")
        return "\n".join(prompt_parts)

    def _segment_text(self, text):
        print("[Task Decomposition] Segmenting text...")
        # Basic sentence segmentation
        return [s.strip() for s in text.split('.') if s.strip()]

    def _analyze_semantic_chunk(self, chunk):
        print(f"[Semantic Analysis] Analyzing chunk: '{chunk}'...")
        # Simulate semantic analysis by identifying key terms
        entities = self._mock_ner(chunk)
        return {"chunk": chunk, "entities": entities}

    def _mock_llm_translate(self, augmented_prompt, chunks_analysis, target_lang, patient_lang):
        print("[GenAI Translation] Processing with LLM (mocked)...")
        translated_chunks = []
        clarification_needed = False
        clarification_question = ""

        full_pivot_text = augmented_prompt.split('Patient statement (pivot language): ')[1].split('\n')[0].strip()

        # Simple mock LLM logic: if 'all over' is present, it's ambiguous
        if "all over" in full_pivot_text.lower() or "无处不在" in full_pivot_text or "kila mahali" in full_pivot_text:
            clarification_needed = True
            clarification_question = self._generate_clarification_question_mock(full_pivot_text, patient_lang)
            return "", True, clarification_question # Return early if clarification is needed

        for analysis in chunks_analysis:
            chunk = analysis["chunk"]
            # Simulate contextual translation for each chunk
            # This is a very basic mock, a real LLM would use the prompt for nuance
            translated_chunk = self._mock_translate(chunk, "english", target_lang)
            translated_chunks.append(translated_chunk)
        
        return ". ".join(translated_chunks), clarification_needed, clarification_question

    def _detect_ambiguity(self, text):
        print("[Refinement] Detecting ambiguity...")
        # Simple keyword-based ambiguity detection
        if "all over" in text.lower() or "unclear" in text.lower():
            return True
        return False
    
    def _generate_clarification_question_mock(self, ambiguous_text, patient_lang):
        print("[Refinement] Generating clarification question...")
        if "all over" in ambiguous_text.lower() or "kila mahali" in ambiguous_text.lower():
            if patient_lang == "swahili":
                return "Unaweza kuelezea maumivu kwa undani zaidi? Je, ni makali, butu, au yanawaka? Unayasikia wapi hasa?"
            elif patient_lang == "mandarin":
                return "你能更具体地描述疼痛吗？是尖锐的、钝的还是灼热的？你具体在哪里感到疼痛？"
            return "Can you describe the pain more specifically? Is it sharp, dull, burning? Where exactly do you feel it?"
        return "Could you please provide more details?"

    def _perform_consistency_check(self, translated_text, target_lang):
        print("[Refinement] Performing consistency checks...")
        # Simulate checking if translated medical terms are consistent
        # This is highly simplified
        for term_en, term_target in self.translation_rules.get(("english", target_lang), {}).items():
            if term_target in translated_text and term_en not in translated_text:
                print(f"  - Consistency check: Found '{term_target}' in translation. Assumed consistent.")
                return True
        print(f"  - Consistency check: No specific medical terms found or check not comprehensive.")
        return True

    def _calculate_semantic_similarity(self, original_text, translated_text):
        print("[Refinement] Calculating semantic similarity...")
        # In a real scenario, this would use sentence embeddings and cosine similarity.
        # Here, we'll just give a high score if texts are somewhat similar in length/content.
        if len(original_text) * 0.7 < len(translated_text) < len(original_text) * 1.3:
            return 0.85 # High confidence
        return 0.60 # Moderate confidence

    def _mock_tts(self, text, lang):
        print(f"[TTS] Synthesizing speech for {lang}: '{text[:30]}...' ")
        return f"[Audio output in {lang} for: '{text}']"

    def process_consultation(self, audio_input, patient_lang, doctor_lang, is_clarification_response=False):
        print("\n--- Starting MMCA Consultation Process ---")
        print(f"Patient Language: {patient_lang.upper()}, Doctor Language: {doctor_lang.upper()}")

        # 1. Input Pre-processing
        transcribed_text = self._mock_stt(audio_input, patient_lang)
        print(f"Transcribed ({patient_lang}): {transcribed_text}")

        # If this is a response to a clarification, we might skip initial MT if pivot is already known
        pivot_language = "english"
        if patient_lang != pivot_language:
            pivot_text = self._mock_translate(transcribed_text, patient_lang, pivot_language)
        else:
            pivot_text = transcribed_text
        print(f"Pivot Text ({pivot_language}): {pivot_text}")

        # 2. Prompt Augmentation
        medical_entities = self._mock_ner(pivot_text)
        medical_definitions = {entity: self._mock_medical_dictionary_lookup(entity) for entity in medical_entities}
        exemplars = self._mock_retrieve_exemplars(pivot_text, pivot_language)
        
        augmented_prompt = self._construct_prompt(pivot_text, medical_definitions, exemplars)
        print(f"Augmented Prompt:\n{augmented_prompt[:200]}...")

        # 3. Task Decomposition and Planning
        segmented_pivot_text = self._segment_text(pivot_text)
        chunks_analysis = [self._analyze_semantic_chunk(chunk) for chunk in segmented_pivot_text]
        print(f"Analyzed Chunks: {chunks_analysis}")

        # 4. Generative AI (LLM) for Contextual Translation
        llm_translated_text_to_doctor_lang, clarification_needed, clarification_q = self._mock_llm_translate(augmented_prompt, chunks_analysis, doctor_lang, patient_lang)
        
        if clarification_needed:
            print(f"--- CLARIFICATION NEEDED ---")
            print(f"Clarification Question ({patient_lang}): {clarification_q}")
            tts_clarification = self._mock_tts(clarification_q, patient_lang)
            print(f"Patient will hear: {tts_clarification}")
            return {
                "status": "clarification_needed",
                "clarification_question": clarification_q,
                "tts_output": tts_clarification
            }

        print(f"LLM Translated (to Doctor's Lang): {llm_translated_text_to_doctor_lang}")
        
        # If LLM translates to a pivot language first (e.g., English), then translate to doctor_lang
        # For this mock, we assume _mock_llm_translate can target doctor_lang directly.
        final_translation_for_doctor = llm_translated_text_to_doctor_lang

        # 5. Iterative Refinement and Feedback (on LLM's direct translation)
        ambiguous = self._detect_ambiguity(pivot_text) # Check ambiguity of original input
        consistency_ok = self._perform_consistency_check(final_translation_for_doctor, doctor_lang)
        semantic_similarity_score = self._calculate_semantic_similarity(pivot_text, final_translation_for_doctor)
        
        print(f"Ambiguity in original input detected: {ambiguous}")
        print(f"Consistency Check OK: {consistency_ok}")
        print(f"Semantic Similarity Score: {semantic_similarity_score}")

        # If refinement detects issues, could trigger re-translation or human review (mocked)
        if not consistency_ok or semantic_similarity_score < 0.7:
            print("[Refinement] Issues detected, flagging for review.")
            final_translation_for_doctor = "[REVIEW REQUIRED] " + final_translation_for_doctor

        # 6. Output Post-processing
        # Assuming final_translation_for_doctor is already in the correct language
        tts_output_doctor = self._mock_tts(final_translation_for_doctor, doctor_lang)

        # Also prepare translation for patient (e.g., doctor's response back to patient)
        # This part assumes a doctor's response is also generated and needs translation
        # For now, let's just translate the patient's original statement into the doctor's language for simplicity
        # or assume the LLM output is the final response for the doctor.
        # If the doctor then responds, that would be a new cycle.

        print("--- MMCA Consultation Process Completed ---")
        return {
            "status": "success",
            "original_patient_input_text": transcribed_text,
            "pivot_processed_text": pivot_text,
            "final_translation_for_doctor": final_translation_for_doctor,
            "tts_output_for_doctor": tts_output_doctor,
            "consistency_check_ok": consistency_ok,
            "semantic_similarity_score": semantic_similarity_score
        }


# Example Usage:
if __name__ == "__main__":
    mmca = MMCAService()

    # Scenario 1: Patient input, normal flow
    print("\n=======================================")
    print("Scenario 1: Normal Patient Input (Swahili to English Doctor)")
    print("=======================================")
    result1 = mmca.process_consultation("audio_data_swahili_1", "swahili", "english")
    print("Result 1 Status:", result1["status"])
    if result1["status"] == "success":
        print("Final Translation for Doctor:", result1["final_translation_for_doctor"])
        print("TTS Output for Doctor:", result1["tts_output_for_doctor"])
    
    # Scenario 2: Patient input, leading to clarification
    print("\n=======================================")
    print("Scenario 2: Ambiguous Patient Input (Mandarin to English Doctor)")
    print("=======================================")
    # We'll simulate an ambiguous input directly via the mock_stt for mandarin
    mmca.translation_rules[("mandarin", "english")] = {"我感觉发烧和咳嗽。头痛无处不在。": "I felt fever and cough. Headache all over."}
    result2 = mmca.process_consultation("audio_data_mandarin_ambiguous", "mandarin", "english")
    print("Result 2 Status:", result2["status"])
    if result2["status"] == "clarification_needed":
        print("Clarification Question:", result2["clarification_question"])
        print("TTS Output for Clarification:", result2["tts_output"])
        
        # Simulate patient's response to clarification
        print("\n--- Simulating Patient's Clarification Response ---")
        mmca.translation_rules[("mandarin", "english")] = {"是搏动性疼痛，集中在前额。": "It is a throbbing pain, focused on the forehead."}
        result2_response = mmca.process_consultation("audio_data_mandarin_clarified", "mandarin", "english", is_clarification_response=True)
        print("Result 2 Response Status:", result2_response["status"])
        if result2_response["status"] == "success":
            print("Final Translation for Doctor (after clarification):", result2_response["final_translation_for_doctor"])
            print("TTS Output for Doctor (after clarification):", result2_response["tts_output_for_doctor"])

    # Scenario 3: English Patient, English Doctor (direct flow)
    print("\n=======================================")
    print("Scenario 3: English Patient to English Doctor")
    print("=======================================")
    result3 = mmca.process_consultation("I have a terrible headache and feel very tired.", "english", "english")
    print("Result 3 Status:", result3["status"])
    if result3["status"] == "success":
        print("Final Translation for Doctor:", result3["final_translation_for_doctor"])
        print("TTS Output for Doctor:", result3["tts_output_for_doctor"])
