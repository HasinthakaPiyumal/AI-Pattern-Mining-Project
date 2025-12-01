import random

def ingest_and_preprocess_text(source_material):
    passages = [p.strip() for p in source_material.split(". ") if p.strip()]
    return passages

def generate_qa_forward_pass(passage):
    if "heart" in passage.lower():
        question = "What is the primary function of the heart?"
        answer = "The heart pumps blood throughout the body."
    elif "lungs" in passage.lower():
        question = "What is the main role of the lungs in the body?"
        answer = "The lungs facilitate gas exchange, taking in oxygen and expelling carbon dioxide."
    else:
        question = f"What is described in '{passage[:50]}...'?"
        answer = f"The passage describes: {passage[:50]}..."
    return {"original_passage": passage, "question": question, "answer": answer}

def regenerate_question_from_answer(answer_text):
    if "pumps blood" in answer_text.lower():
        return "What organ circulates blood?"
    elif "gas exchange" in answer_text.lower():
        return "Which organs are responsible for breathing?"
    else:
        return f"Can you rephrase the question for: '{answer_text[:50]}...'?"

def verify_answer_from_question(question_text, original_passages):
    retrieved_passage = ""
    for p in original_passages:
        if question_text.lower().replace("what is","").replace("what organ"," ").strip() in p.lower():
            retrieved_passage = p
            break

    if retrieved_passage:
        if "heart" in retrieved_passage.lower() and "pumps blood" in question_text.lower():
            return "The heart, a muscular organ, effectively pumps blood throughout the circulatory system."
        elif "lungs" in retrieved_passage.lower() and "breathing" in question_text.lower():
            return "The lungs perform vital gas exchange, supplying oxygen and removing carbon dioxide."
        else:
            return f"Based on '{retrieved_passage[:50]}...', the answer is: ..."
    return "No relevant passage found to verify the answer."

def _calculate_mock_similarity(text1, text2):
    if text1 == text2:
        return 1.0
    elif any(word in text1.lower() for word in text2.lower().split()) or any(word in text2.lower() for word in text1.lower().split()):
        return random.uniform(0.7, 0.95)
    else:
        return random.uniform(0.3, 0.6)

def evaluate_consistency(q_gen, q_prime, a_gen, a_double_prime, similarity_threshold):
    q_similarity = _calculate_mock_similarity(q_gen, q_prime)
    a_similarity = _calculate_mock_similarity(a_gen, a_double_prime)

    print(f"  - Q_gen: {q_gen}")
    print(f"  - Q_prime: {q_prime}")
    print(f"  - Q Similarity: {q_similarity:.2f}")
    print(f"  - A_gen: {a_gen}")
    print(f"  - A_double_prime: {a_double_prime}")
    print(f"  - A Similarity: {a_similarity:.2f}")

    return q_similarity >= similarity_threshold and a_similarity >= similarity_threshold

def compile_study_guide(filtered_qa_pairs):
    study_guide_content = []
    for i, qa in enumerate(filtered_qa_pairs):
        study_guide_content.append(f"-- QA Pair {i+1} --")
        study_guide_content.append(f"Passage: {qa['original_passage']}")
        study_guide_content.append(f"Question: {qa['question']}")
        study_guide_content.append(f"Answer: {qa['answer']}")
        study_guide_content.append("")
    return "\n".join(study_guide_content)


if __name__ == "__main__":
    medical_literature = (
        "The heart is a muscular organ in most animals, which pumps blood through the blood vessels of the circulatory system. "
        "Blood provides the body with oxygen and nutrients, as well as assisting in the removal of metabolic wastes. "
        "In humans, the heart is roughly the size of a clenched fist and is located between the lungs, in the middle compartment of the chest. "
        "The lungs are a pair of spongy, air-filled organs located on either side of the chest (thorax). "
        "They are the main organs of the respiratory system and are responsible for the process of gas exchange (respiration). "
        "The diaphragm is a sheet of internal skeletal muscle that extends across the bottom of the thoracic cavity. "
        "The liver is a large, fleshy organ that sits on the right side of the belly. It's essential for digesting food and ridding your body of toxic substances. "
        "Kidneys are a pair of bean-shaped organs on either side of your spine, below your ribs and behind your belly. Each kidney is about 4 or 5 inches long, about the size of a fist. "
        "Their main job is to filter your blood, removing waste and excess water to make urine."
    )

    print("Starting Study Guide Generation...")

    # 1. Data Ingestion & Preprocessing
    passages = ingest_and_preprocess_text(medical_literature)
    print(f"Ingested {len(passages)} passages.\n")

    filtered_qa_pairs = []
    consistency_threshold = 0.75 # Example threshold

    # Prepare for verification (simplified database of original passages)
    original_passage_database = passages

    for i, passage in enumerate(passages):
        print(f"Processing Passage {i+1}: '{passage[:70]}...'\n")

        # 2. Synthetic QA Generation (Forward Pass)
        qa_pair = generate_qa_forward_pass(passage)
        q_gen = qa_pair["question"]
        a_gen = qa_pair["answer"]
        print(f"Generated QA: Q='{q_gen}', A='{a_gen}'")

        # 3.1. Answer-to-Question Regeneration (Reverse Pass)
        q_prime = regenerate_question_from_answer(a_gen)
        print(f"Regenerated Q from A: Q'='{q_prime}'")

        # 3.2. Question-to-Answer/Passage Verification
        a_double_prime = verify_answer_from_question(q_gen, original_passage_database)
        print(f"Verified A from Q and original passage: A''='{a_double_prime}'")

        # 3.3. Consistency Evaluation
        is_consistent = evaluate_consistency(q_gen, q_prime, a_gen, a_double_prime, consistency_threshold)

        if is_consistent:
            print(f"--> QA Pair is CONSISTENT. Adding to study guide.\n")
            filtered_qa_pairs.append(qa_pair)
        else:
            print(f"--> QA Pair is INCONSISTENT. Discarding.\n")

    # 4. Study Guide Compilation
    compiled_guide = compile_study_guide(filtered_qa_pairs)

    print("\n--- Final Compiled Study Guide ---")
    if filtered_qa_pairs:
        print(compiled_guide)
    else:
        print("No consistent QA pairs were generated.")
