import torch
import numpy as np
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# --- 4. Dummy Medical Knowledge Base (for Simulation) ---
dummy_medical_knowledge_base = [
    "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain.",
    "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy.",
    "Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated.",
    "The human heart has four chambers: two atria and two ventricles. It pumps blood throughout the body.",
    "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing."
]

class SyntheticQAGenerator:
    def __init__(self, qa_model_name="deepset/roberta-base-squad2", qg_model_name="t5-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name, device=0 if self.device == "cuda" else -1)
        
        self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name).to(self.device)

    def generate_question_from_answer_and_passage(self, answer, passage):
        # For simplicity, this uses T5 for text generation. More advanced QG models might exist.
        # This assumes a prompt format like 'generate question: <answer> | <passage>'
        text = f"generate question: {answer} | {passage}"
        input_ids = self.qg_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        outputs = self.qg_model.generate(input_ids, max_new_tokens=64, num_beams=4, early_stopping=True)
        question = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question

    def generate_qa_from_passage(self, passage):
        qa_pairs = []
        # Use QA pipeline to find potential answers
        # For simplicity, we'll try to extract one answer based on a generic question or heuristic
        # In a real scenario, this would involve more sophisticated answer extraction.
        
        # Let's simulate by asking a general question to get an answer, then generate a specific question
        try:
            qa_result = self.qa_pipeline(question="What is mentioned in the text?", context=passage, top_k=1)
            if qa_result and qa_result['score'] > 0.1: # Basic confidence threshold
                answer = qa_result['answer']
                generated_question = self.generate_question_from_answer_and_passage(answer, passage)
                qa_pairs.append({'passage': passage, 'question': generated_question, 'answer': answer})
        except Exception as e:
            pass # Handle cases where QA pipeline might fail for short or irrelevant passages
        
        return qa_pairs

class ConsistencyChecker:
    def __init__(self, q_regeneration_model_name="t5-small", sentence_transformer_model_name="all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.qr_tokenizer = AutoTokenizer.from_pretrained(q_regeneration_model_name)
        self.qr_model = AutoModelForSeq2SeqLM.from_pretrained(q_regeneration_model_name).to(self.device)
        self.sentence_model = SentenceTransformer(sentence_transformer_model_name, device=self.device)

    def regenerate_question_from_answer(self, answer):
        # Use a text generation model to regenerate a question from an answer
        text = f"generate question: {answer}"
        input_ids = self.qr_tokenizer(text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(self.device)
        outputs = self.qr_model.generate(input_ids, max_new_tokens=64, num_beams=4, early_stopping=True)
        regenerated_question = self.qr_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return regenerated_question

    def get_question_similarity(self, question1, question2):
        embeddings = self.sentence_model.encode([question1, question2], convert_to_tensor=True, device=self.device)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return similarity

    def verify_answer_in_passage(self, question, answer, passage, knowledge_base=None):
        # Simplified approach: Check if key terms from the answer are in the passage
        # and if the question's essence is addressed.
        answer_terms = set(word.lower() for word in answer.replace('.', '').replace(',', '').split() if len(word) > 2)
        passage_terms = set(word.lower() for word in passage.replace('.', '').replace(',', '').split() if len(word) > 2)
        
        # Basic check: all significant answer terms are in the passage
        answer_terms_in_passage = all(term in passage_terms for term in answer_terms)

        # If a knowledge base is provided, could perform a more advanced check
        if knowledge_base:
            # For demo, just check if the answer is directly present in any KB entry
            # In a real system, this would be semantic search and QA over KB
            answer_confirmed_by_kb = any(answer.lower() in kb_entry.lower() for kb_entry in knowledge_base)
            return answer_terms_in_passage and answer_confirmed_by_kb

        return answer_terms_in_passage

class FilteringOrchestrator:
    def __init__(self, qa_generator, consistency_checker, similarity_threshold=0.7):
        self.qa_generator = qa_generator
        self.consistency_checker = consistency_checker
        self.similarity_threshold = similarity_threshold

    def process_medical_passages(self, passages, knowledge_base=None):
        high_quality_qa_pairs = []
        for passage in passages:
            initial_qa_pairs = self.qa_generator.generate_qa_from_passage(passage)

            for qa_pair in initial_qa_pairs:
                original_question = qa_pair['question']
                generated_answer = qa_pair['answer']
                original_passage = qa_pair['passage']

                # 1. Question Regeneration and Similarity Check
                regenerated_question = self.consistency_checker.regenerate_question_from_answer(generated_answer)
                question_similarity = self.consistency_checker.get_question_similarity(original_question, regenerated_question)

                # 2. Answer Verification
                answer_verified = self.consistency_checker.verify_answer_in_passage(
                    original_question, generated_answer, original_passage, knowledge_base
                )

                # Apply filtering rules
                if question_similarity >= self.similarity_threshold and answer_verified:
                    high_quality_qa_pairs.append({
                        'passage': original_passage,
                        'question': original_question,
                        'answer': generated_answer,
                        'question_similarity': question_similarity,
                        'answer_verified': answer_verified
                    })
        return high_quality_qa_pairs

if __name__ == "__main__":
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize components
    qa_generator = SyntheticQAGenerator(device=device)
    consistency_checker = ConsistencyChecker(device=device)
    orchestrator = FilteringOrchestrator(qa_generator, consistency_checker, similarity_threshold=0.6)

    # Example Medical Passages
    medical_passages = [
        "Insulin is a hormone that regulates blood sugar levels. It is produced by the pancreas.",
        "Fever is a temporary increase in your body temperature, often due to an illness. A fever can be a symptom of many different conditions, from minor to serious.",
        "The liver is a vital organ that processes nutrients, filters blood, and creates bile to help digest fats.",
        "Migraine is a severe headache often accompanied by symptoms such as throbbing pain, sensitivity to light and sound, nausea, and vomiting."
    ]

    # Process passages and filter QA pairs
    print("\n--- Generating and Filtering QA Pairs ---")
    filtered_qa_data = orchestrator.process_medical_passages(medical_passages, knowledge_base=dummy_medical_knowledge_base)

    print("\n--- High-Quality Filtered QA Data ---")
    if filtered_qa_data:
        for i, qa in enumerate(filtered_qa_data):
            print(f"QA Pair {i+1}:")
            print(f"  Passage: {qa['passage'][:100]}...")
            print(f"  Question: {qa['question']}")
            print(f"  Answer: {qa['answer']}")
            print(f"  Question Similarity (Original vs. Regenerated): {qa['question_similarity']:.2f}")
            print(f"  Answer Verified: {qa['answer_verified']}")
            print("---------------------")
    else:
        print("No high-quality QA pairs found after filtering.")

    print("\n--- Demonstration Complete ---")