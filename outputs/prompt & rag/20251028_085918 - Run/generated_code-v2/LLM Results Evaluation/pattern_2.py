import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# Suppress warnings from transformers
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class DataLoader:
    def load_medical_passages(self):
        return [
            "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). The body either doesn't produce enough insulin, or it resists the effects of insulin. Symptoms include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
            "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Symptoms are often not noticeable, which is why it's called the 'silent killer'. Severe hypertension can cause headaches, shortness of breath, or nosebleeds.",
            "Alzheimer's disease is a progressive neurological disorder that causes the brain to shrink and brain cells to die. It is the most common cause of dementia, a continuous decline in thinking, behavioral and social skills that disrupts a person's ability to function independently. Early signs may include forgetting recent events or conversations. As the disease progresses, memory impairments worsen and other symptoms emerge."
        ]

class QAGenerator:
    def __init__(self, model_name="google/flan-t5-small", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model.to(self.device)

    def generate_qa_pair(self, passage):
        # Use a prompt for question generation given a context
        question_prompt = f"Given the medical passage, generate a question about it: {passage}"
        question_inputs = self.tokenizer(question_prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        question_outputs = self.model.generate(question_inputs.input_ids, max_new_tokens=64, num_beams=5, early_stopping=True)
        question = self.tokenizer.decode(question_outputs[0], skip_special_tokens=True)

        # Use a prompt for answer generation given a context and question
        answer_prompt = f"Answer the following question based on the medical passage. Question: {question} Context: {passage}"
        answer_inputs = self.tokenizer(answer_prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        answer_outputs = self.model.generate(answer_inputs.input_ids, max_new_tokens=128, num_beams=5, early_stopping=True)
        answer = self.tokenizer.decode(answer_outputs[0], skip_special_tokens=True)

        return question, answer

class ConsistencyFilter:
    def __init__(self, qg_model_name="google/flan-t5-small", embed_model_name="all-MiniLM-L6-v2", device="cpu"):
        self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name)
        self.embedder = SentenceTransformer(embed_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.qg_model.to(self.device)
        self.embedder.to(self.device)

    def check_consistency(self, original_passage, generated_question, generated_answer, similarity_threshold=0.7):
        # Attempt to regenerate the question from the answer and original passage
        reverse_prompt = f"Given the medical passage and the answer, generate a question: Answer: {generated_answer} Context: {original_passage}"
        reverse_inputs = self.qg_tokenizer(reverse_prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        reverse_outputs = self.qg_model.generate(reverse_inputs.input_ids, max_new_tokens=64, num_beams=5, early_stopping=True)
        re_generated_question = self.qg_tokenizer.decode(reverse_outputs[0], skip_special_tokens=True)

        # Compute semantic similarity
        embeddings = self.embedder.encode([generated_question, re_generated_question], convert_to_tensor=True)
        cosine_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        return cosine_similarity >= similarity_threshold, cosine_similarity

def main():
    data_loader = DataLoader()
    qa_generator = QAGenerator(device="cuda")
    consistency_filter = ConsistencyFilter(device="cuda")

    medical_passages = data_loader.load_medical_passages()
    filtered_qa_pairs = []

    print("Starting QA pair generation and consistency filtering...")

    for i, passage in enumerate(medical_passages):
        print(f"\nProcessing passage {i+1}/{len(medical_passages)}")
        try:
            generated_question, generated_answer = qa_generator.generate_qa_pair(passage)
            print(f"  Generated Q: {generated_question}")
            print(f"  Generated A: {generated_answer}")

            is_consistent, similarity = consistency_filter.check_consistency(
                passage, generated_question, generated_answer
            )

            if is_consistent:
                print(f"  Consistency Check: PASSED (Similarity: {similarity:.2f})")
                filtered_qa_pairs.append({
                    "passage": passage,
                    "question": generated_question,
                    "answer": generated_answer,
                    "similarity": similarity
                })
            else:
                print(f"  Consistency Check: FAILED (Similarity: {similarity:.2f})")

        except Exception as e:
            print(f"  An error occurred processing passage {i+1}: {e}")

    print("\n--- Filtered QA Pairs ---")
    if not filtered_qa_pairs:
        print("No consistent QA pairs were generated.")
    for j, qa_pair in enumerate(filtered_qa_pairs):
        print(f"\nQA Pair {j+1}:")
        print(f"  Passage: {qa_pair['passage'][:100]}...")
        print(f"  Question: {qa_pair['question']}")
        print(f"  Answer: {qa_pair['answer']}")
        print(f"  Consistency Similarity: {qa_pair['similarity']:.2f}")

if __name__ == "__main__":
    main()
