import re
import time
from typing import List, Dict, Tuple

# --- Mock External Services ---

class MockMachineTranslationService:
    """Simulates an external machine translation API."""
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        # Simulate translation delay and simple translation logic
        time.sleep(0.1)
        if "non-english" in source_lang.lower() and target_lang.lower() == "english":
            print(f"[MT Service] Translating from {source_lang} to English...")
            return f"[English Translation from {source_lang}] {text.upper()}" # Simple mock
        print(f"[MT Service] Translating from {source_lang} to {target_lang}...")
        return f"[Translated {target_lang}] {text}"

class MockMedicalOntologyService:
    """Simulates querying a medical dictionary/ontology (e.g., UMLS, SNOMED CT)."""
    _medical_terms = {
        "cardiac arrest": "Sudden cessation of cardiac function, causing the absence of circulation.",
        "myocardial infarction": "Heart attack; irreversible necrosis of heart muscle secondary to prolonged ischemia.",
        "hypertension": "A medical condition in which the blood pressure in the arteries is persistently elevated.",
        "metastasize": "To spread to other sites in the body by metastasis."
    }

    def get_definition(self, term: str) -> str:
        print(f"[Ontology Service] Looking up definition for '{term}'...")
        return self._medical_terms.get(term.lower(), f"Definition not found for '{term}'.")

class MockExemplarRetrievalService:
    """Simulates retrieving relevant translated exemplars from a database."""
    _exemplars = {
        "cardiac arrest treatment": {
            "source_text": "紧急心脏骤停治疗", 
            "english_translation": "Emergency treatment for cardiac arrest.",
            "context": "This exemplar shows the translation of critical care procedures."
        },
        "blood pressure regulation": {
            "source_text": "血压调节机制", 
            "english_translation": "Blood pressure regulation mechanism.",
            "context": "Relevant for understanding physiological processes."
        }
    }

    def retrieve_exemplars(self, query: str, num_exemplars: int = 1) -> List[Dict[str, str]]:
        print(f"[Exemplar Service] Retrieving {num_exemplars} exemplars for '{query}'...")
        # Simple keyword-based retrieval for demonstration
        found_exemplars = []
        for key, value in self._exemplars.items():
            if query.lower() in key.lower() or query.lower() in value['source_text'].lower() or query.lower() in value['english_translation'].lower():
                found_exemplars.append(value)
                if len(found_exemplars) >= num_exemplars:
                    break
        return found_exemplars

# --- Core Modules ---

class TextChunker:
    """Splits a document into manageable chunks."""
    def chunk_text(self, text: str, max_words_per_chunk: int = 150) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_words_per_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        print(f"[Chunker] Split text into {len(chunks)} chunks.")
        return chunks

class PromptAugmentor:
    """Augments translation prompts with contextual information and exemplars."""
    def __init__(self, ontology_service: MockMedicalOntologyService, exemplar_service: MockExemplarRetrievalService):
        self.ontology_service = ontology_service
        self.exemplar_service = exemplar_service

    def augment_prompt(self, 
                       original_text_chunk: str, 
                       source_lang: str, 
                       target_lang: str) -> str:
        
        augmented_parts = []
        
        # 1. Medical Terminology Lookup
        # A more sophisticated version would use NLP to extract key terms
        potential_terms = ["cardiac arrest", "myocardial infarction", "hypertension", "metastasize"]
        for term in potential_terms:
            if term in original_text_chunk.lower():
                definition = self.ontology_service.get_definition(term)
                if definition:
                    augmented_parts.append(f"Definition of '{term}': {definition}.")
        
        # 2. Exemplar Retrieval
        # A more sophisticated version would use embeddings for semantic search
        exemplars = self.exemplar_service.retrieve_exemplars(original_text_chunk, num_exemplars=1)
        for ex in exemplars:
            augmented_parts.append(
                f"Consider this related translation example: "
                f"Source: '{ex['source_text']}', English Translation: '{ex['english_translation']}'. "
                f"Context: {ex['context']}"
            )
            
        if augmented_parts:
            augmentation_str = "\n\n" + "\n".join(augmented_parts)
            print(f"[Augmentor] Added {len(augmented_parts)} augmentations.")
        else:
            augmentation_str = ""
            print("[Augmentor] No augmentations added.")

        prompt = (
            f"Translate the following medical text from {source_lang} to {target_lang}. "
            f"Ensure accuracy and use appropriate medical terminology. "
            f"{augmentation_str}\n\nTEXT TO TRANSLATE: {original_text_chunk}"
        )
        return prompt

class TranslationQualityEvaluator:
    """Simulates evaluating the quality of a medical translation."""
    def evaluate(self, original_source: str, translated_text: str, expected_keywords: List[str]) -> Tuple[float, List[str]]:
        # Simple mock evaluation: check for presence of keywords and consistency
        score = 0.0
        feedback = []
        
        translated_lower = translated_text.lower()
        source_lower = original_source.lower()
        
        present_keywords = [kw for kw in expected_keywords if kw.lower() in translated_lower]
        missing_keywords = [kw for kw in expected_keywords if kw.lower() not in translated_lower]
        
        if present_keywords:
            score += (len(present_keywords) / len(expected_keywords)) * 0.7 # 70% for keyword presence
            feedback.append(f"Keywords present: {', '.join(present_keywords)}.")
        if missing_keywords:
            feedback.append(f"Warning: Missing expected medical keywords: {', '.join(missing_keywords)}.")
            
        # Simulate some consistency check (e.g., if a term was translated differently in different chunks)
        if "cardiac arrest" in source_lower and "heart attack" in translated_lower and "cardiac arrest" not in translated_lower:
             score -= 0.1
             feedback.append("Potential inconsistency: 'cardiac arrest' translated as 'heart attack'. Consider consistent terminology.")
             
        score = max(0.0, min(1.0, score + 0.3)) # Add some baseline score and clamp

        print(f"[Evaluator] Translation score: {score:.2f}, Feedback: {'; '.join(feedback) if feedback else 'None'}")
        return score, feedback

# --- Main Application Logic ---

class GlobalMedicalResearchTranslator:
    """Orchestrates the multi-strategy medical translation process."""
    def __init__(self):
        self.mt_service = MockMachineTranslationService()
        self.ontology_service = MockMedicalOntologyService()
        self.exemplar_service = MockExemplarRetrievalService()
        self.text_chunker = TextChunker()
        self.prompt_augmentor = PromptAugmentor(self.ontology_service, self.exemplar_service)
        self.evaluator = TranslationQualityEvaluator()

    def translate_document(self, 
                           document_text: str, 
                           source_lang: str, 
                           target_lang: str = "English",
                           max_refinement_iterations: int = 2,
                           expected_keywords_for_eval: List[str] = []) -> str:
        
        print(f"\n--- Starting Translation for a document from {source_lang} to {target_lang} ---")
        
        final_translated_chunks = []
        
        # 1. Input Pre-processing: Translate non-English to high-resource language if needed
        processed_source_text = document_text
        initial_target_lang = target_lang
        if source_lang.lower() != target_lang.lower() and source_lang.lower() != "english":
            print(f"[Main] Pre-processing: Translating source from {source_lang} to English for initial understanding.")
            processed_source_text = self.mt_service.translate(document_text, source_lang, "English")
            # For simplicity, further steps will work with this 'pre-processed' English text
            # A real system might use this to guide a direct translation to target_lang
            # For this demo, we'll assume the goal is ultimately English, or we use this as an intermediary.
            # Let's adjust: if target_lang is English, we use processed_source_text directly.
            # If target_lang is not English, a further step would translate processed_source_text to target_lang
            # For this example, let's assume the goal is English after initial pre-processing.
            # If the original target was not English, this pre-processing step is purely for augmenting the prompt
            # and an additional translation from English to the *actual* target would be needed.
            # Let's keep it simple for now and assume the 'high-resource' target is also the final target (English).
            if target_lang.lower() != "english":
                 print("[WARNING] This demo simplifies multi-stage translation. If source is non-English and target is non-English, it first goes to English then to target.")
                 # In a real scenario, this processed_source_text (English) would be used to augment the prompt for the *final* target_lang translation.
                 # For simplicity of this mock, we'll proceed as if the pre-processed English is the main input for augmentation.
            source_lang = "English" # Update source_lang for subsequent steps after pre-processing
            document_text = processed_source_text # Use the pre-processed text for chunking and augmentation
            
        # 2. Task Decomposition and Planning (Chunking)
        text_chunks = self.text_chunker.chunk_text(document_text)
        
        # Process each chunk
        for i, chunk in enumerate(text_chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(text_chunks)} ---")
            current_chunk_translation = ""
            best_chunk_translation = ""
            highest_score = -1.0
            
            for iteration in range(max_refinement_iterations):
                print(f"[Chunk {i+1}] Refinement Iteration {iteration+1}")
                
                # 3. Prompt Augmentation
                augmented_prompt = self.prompt_augmentor.augment_prompt(chunk, source_lang, target_lang)
                
                # Perform translation with augmented prompt
                # In a real system, this would call a powerful LLM / GenAI
                print("[Main] Calling GenAI for translation (mocked by MT Service with augmented prompt).")
                # Simulating GenAI by simply taking the augmented prompt as the input to a basic MT service
                # A real GenAI would understand the prompt and generate translation.
                current_chunk_translation = self.mt_service.translate(augmented_prompt, source_lang, target_lang)
                
                # 4. Iterative Refinement (Automated Feedback)
                score, feedback = self.evaluator.evaluate(chunk, current_chunk_translation, expected_keywords_for_eval)
                
                if score > highest_score:
                    highest_score = score
                    best_chunk_translation = current_chunk_translation
                    
                if score >= 0.95: # Arbitrary threshold for satisfactory quality
                    print(f"[Chunk {i+1}] Satisfactory quality achieved in iteration {iteration+1}. Stopping refinement.")
                    break
                elif iteration < max_refinement_iterations - 1:
                    print(f"[Chunk {i+1}] Refining translation based on feedback: {'; '.join(feedback)}")
                    # In a real system, feedback would be used to modify the next prompt
                    # For this mock, we just let it run through iterations with same augmentations
            
            final_translated_chunks.append(best_chunk_translation if best_chunk_translation else current_chunk_translation)
            
        full_translated_document = "\n\n".join(final_translated_chunks)
        print("\n--- Translation Complete ---")
        print(f"Full Translated Document (excerpt):\n{full_translated_document[:500]}...")
        return full_translated_document

# --- Example Usage ---
if __name__ == "__main__":
    translator_app = GlobalMedicalResearchTranslator()

    # Example 1: Non-English to English translation with medical terms
    print("\n========== Example 1: Chinese Medical Text to English ==========")
    chinese_medical_text = (
        "高血压是一种常见的慢性疾病，如果不进行有效治疗，可能导致心脏病和中风。 "
        "心肌梗死是急性冠脉综合征的严重形式，需要紧急医疗干预。 "
        "癌细胞有时会转移到身体的其他部位。" # Hypertension, Myocardial Infarction, Metastasize
    )
    # Expected keywords for evaluation (in English, as target is English)
    expected_keywords_1 = ["hypertension", "chronic disease", "heart disease", "stroke", 
                             "myocardial infarction", "acute coronary syndrome", "emergency medical intervention",
                             "cancer cells", "metastasize"]
    
    translated_doc_1 = translator_app.translate_document(
        document_text=chinese_medical_text,
        source_lang="Chinese",
        target_lang="English",
        expected_keywords_for_eval=expected_keywords_1
    )
    print("\nFinal Translated Document (Example 1):\n", translated_doc_1)

    # Example 2: English medical text (demonstrates augmentation and refinement)
    print("\n========== Example 2: English Medical Text (Augmentation Demo) ==========")
    english_medical_text = (
        "The patient presented with symptoms indicative of cardiac arrest. "
        "Immediate intervention was required to prevent irreversible damage. "
        "Understanding myocardial infarction mechanisms is crucial for prevention."
    )
    expected_keywords_2 = ["cardiac arrest", "immediate intervention", "irreversible damage",
                             "myocardial infarction", "mechanisms", "prevention"]
                             
    translated_doc_2 = translator_app.translate_document(
        document_text=english_medical_text,
        source_lang="English",
        target_lang="English", # Translate English to English to show augmentation/refinement on complex text
        expected_keywords_for_eval=expected_keywords_2
    )
    print("\nFinal Translated Document (Example 2):\n", translated_doc_2)

    # Example 3: Shorter text with one iteration to show baseline
    print("\n========== Example 3: Shorter Text (Single Iteration) ==========")
    short_text = "Cardiac arrest needs immediate medical attention."
    expected_keywords_3 = ["cardiac arrest", "immediate medical attention"]
    translated_doc_3 = translator_app.translate_document(
        document_text=short_text,
        source_lang="English",
        target_lang="English",
        max_refinement_iterations=1,
        expected_keywords_for_eval=expected_keywords_3
    )
    print("\nFinal Translated Document (Example 3):\n", translated_doc_3)
