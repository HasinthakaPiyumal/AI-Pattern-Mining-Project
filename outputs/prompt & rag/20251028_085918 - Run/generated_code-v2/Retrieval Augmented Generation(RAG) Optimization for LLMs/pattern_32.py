import os
import json
from transformers import pipeline

class MedicalQADatasetGenerator:
    def __init__(self, model_name="t5-small"):
        self.qa_pipeline = pipeline("text2text-generation", model=model_name)

    def ingest_texts(self, input_dir):
        passages = []
        if not os.path.exists(input_dir):
            print(f"Input directory '{input_dir}' not found. Please create it and add .txt files.")
            return []

        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    passages.append(f.read())
        return passages

    def generate_question(self, context):
        prompt = f"generate question: {context}"
        result = self.qa_pipeline(prompt, max_new_tokens=50, do_sample=False)
        return result[0]["generated_text"].strip() if result else ""

    def generate_answer(self, question, context):
        prompt = f"answer: {question} context: {context}"
        result = self.qa_pipeline(prompt, max_new_tokens=100, do_sample=False)
        return result[0]["generated_text"].strip() if result else ""

    def calculate_jaccard_similarity(self, text1, text2):
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / union

    def check_consistency(self, original_answer, generated_question, original_context, similarity_threshold=0.5):
        re_generated_answer = self.generate_answer(generated_question, original_context)
        similarity = self.calculate_jaccard_similarity(original_answer, re_generated_answer)
        return similarity >= similarity_threshold

    def save_qa_pairs(self, qa_pairs, output_filepath):
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=4)

    def main(self, input_dir="input_medical_docs", output_filepath="output_qa_dataset.json", num_qa_per_passage=1, similarity_threshold=0.5):
        print(f"Ingesting texts from {input_dir}...")
        passages = self.ingest_texts(input_dir)
        print(f"Found {len(passages)} passages.")

        if not passages:
            print("No passages found to process. Exiting.")
            return

        synthetic_qa_dataset = []
        for i, context in enumerate(passages):
            print(f"Processing passage {i+1}/{len(passages)}...")
            for _ in range(num_qa_per_passage):
                question = self.generate_question(context)
                answer = self.generate_answer(question, context)

                if question and answer:
                    if self.check_consistency(answer, question, context, similarity_threshold):
                        synthetic_qa_dataset.append({
                            "context": context,
                            "question": question,
                            "answer": answer
                        })
                        print(f"  Generated consistent QA pair. Total: {len(synthetic_qa_dataset)}")
                    else:
                        print("  QA pair failed consistency check.")
                else:
                    print("  Skipped generating empty question or answer.")

        self.save_qa_pairs(synthetic_qa_dataset, output_filepath)
        print(f"Generated {len(synthetic_qa_dataset)} synthetic QA pairs saved to {output_filepath}")

if __name__ == "__main__":
    generator = MedicalQADatasetGenerator()
    generator.main(
        input_dir="input_medical_docs",
        output_filepath="output_qa_dataset.json",
        num_qa_per_passage=2,
        similarity_threshold=0.6
    )
