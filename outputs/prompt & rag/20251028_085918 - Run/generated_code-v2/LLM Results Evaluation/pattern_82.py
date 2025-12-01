import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class MockLLMGenerator:
    def __init__(self, model_name="MockLLM"):
        self.model_name = model_name

    def generate_qa_pair(self, passage):
        if "feature X" in passage.lower():
            question = "How do I activate Feature X?"
            answer = "To activate Feature X, navigate to settings and toggle the Feature X switch."
        elif "troubleshooting" in passage.lower():
            question = "What are common troubleshooting steps?"
            answer = "Common troubleshooting steps include restarting the device and checking network connections."
        else:
            question = f"What is the main point about {passage.split('.')[0]}?"
            answer = f"The main point is related to {passage.split('.')[0]}."
        return question, answer

    def regenerate_question_from_answer(self, answer):
        if "activate Feature X" in answer:
            return "How to turn on Feature X?"
        elif "troubleshooting steps" in answer:
            return "What actions should I take to fix issues?"
        else:
            return f"Tell me more about: {answer.split('.')[0]}?"

def calculate_similarity(text1, text2, embedding_model):
    embeddings = embedding_model.encode([text1, text2])
    return cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]

def round_trip_consistency_check(original_question, generated_answer, llm_generator, embedding_model, similarity_threshold):
    re_generated_question = llm_generator.regenerate_question_from_answer(generated_answer)
    similarity = calculate_similarity(original_question, re_generated_question, embedding_model)
    return similarity >= similarity_threshold, similarity

def main():
    similarity_threshold = 0.75
    output_filename = "filtered_faqs.json"

    llm_generator = MockLLMGenerator()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    raw_passages = [
        "This device comes with Feature X, which can be activated in the settings menu. Simply navigate to 'Settings' > 'Features' and toggle 'Feature X' to 'On'.",
        "If you encounter any issues, perform basic troubleshooting steps: 1. Restart the device. 2. Check your internet connection. 3. Update the firmware.",
        "Our new privacy policy ensures user data is encrypted at rest and in transit. For more details, please refer to section 3.1.",
        "The battery life of the new model is approximately 10 hours under continuous usage conditions. Charging takes about 2 hours."
    ]

    accepted_faqs = []

    for i, passage in enumerate(raw_passages):
        print(f"Processing passage {i+1}/{len(raw_passages)}...")
        original_question, generated_answer = llm_generator.generate_qa_pair(passage)

        is_consistent, similarity_score = round_trip_consistency_check(
            original_question,
            generated_answer,
            llm_generator,
            embedding_model,
            similarity_threshold
        )

        if is_consistent:
            accepted_faqs.append({
                "passage": passage,
                "question": original_question,
                "answer": generated_answer,
                "consistency_score": similarity_score
            })
            print(f"  Accepted: Q='{original_question}', A='{generated_answer}' (Similarity: {similarity_score:.2f})")
        else:
            print(f"  Rejected: Q='{original_question}', A='{generated_answer}' (Similarity: {similarity_score:.2f} < {similarity_threshold})")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(accepted_faqs, f, indent=4, ensure_ascii=False)

    print(f"\nFiltered {len(accepted_faqs)} FAQs saved to {output_filename}")
    print("Example accepted FAQ:")
    if accepted_faqs:
        print(json.dumps(accepted_faqs[0], indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()