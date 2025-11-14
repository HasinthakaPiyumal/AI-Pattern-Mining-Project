from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset
from tqdm import tqdm

def generate_qa_pairs_from_text(text_chunks, model_name="t5-small", num_qa_per_chunk=1):
    """
    Generates synthetic QA pairs from a list of text chunks using a pre-trained T5 model.

    Args:
        text_chunks (list): A list of text strings (medical documents or sections).
        model_name (str): The name of the pre-trained T5 model to use.
        num_qa_per_chunk (int): Number of QA pairs to attempt to generate per chunk.

    Returns:
        datasets.Dataset: A Dataset object containing generated 'question' and 'answer' pairs.
    """
    print(f"Loading T5 model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    qa_data = []

    for i, chunk in enumerate(tqdm(text_chunks, desc="Generating QA pairs")):
        for _ in range(num_qa_per_chunk):
            try:
                # Task: Generate a question from the context
                input_text_question = f"generate question: {chunk}"
                inputs_question = tokenizer(input_text_question, return_tensors="pt", max_length=512, truncation=True)
                outputs_question = model.generate(**inputs_question, num_beams=4, max_length=64, early_stopping=True)
                question = tokenizer.decode(outputs_question[0], skip_special_tokens=True)

                # Task: Generate an answer from the context and generated question
                input_text_answer = f"answer_from_context: {question} context: {chunk}"
                inputs_answer = tokenizer(input_text_answer, return_tensors="pt", max_length=512, truncation=True)
                outputs_answer = model.generate(**inputs_answer, num_beams=4, max_length=128, early_stopping=True)
                answer = tokenizer.decode(outputs_answer[0], skip_special_tokens=True)

                # Simple round-trip consistency check (can be more sophisticated)
                # Here, we just check if the answer is not empty and seems somewhat relevant
                if question and answer and len(answer.split()) > 3:
                    qa_data.append({"question": question, "answer": answer, "context": chunk})

            except Exception as e:
                print(f"Error generating QA for chunk {i}: {e}")
                continue

    return Dataset.from_list(qa_data)

if __name__ == "__main__":
    # Example medical text chunks (in a real scenario, these would come from medical journals, EHRs, etc.)
    medical_texts = [
        "Insulin resistance is a key feature of type 2 diabetes, where cells in your muscles, fat, and liver don't respond well to insulin and can't easily take up glucose from your blood.",
        "Hypertension, or high blood pressure, significantly increases the risk of heart disease and stroke. Lifestyle changes like diet, exercise, and reducing sodium intake are crucial for management.",
        "Metformin is a common medication used to treat type 2 diabetes. It works by decreasing glucose production by the liver and improving your body's sensitivity to insulin.",
        "Congestive heart failure occurs when the heart muscle doesn't pump blood as well as it should. Symptoms often include shortness of breath, fatigue, and swelling in the legs.",
        "Regular monitoring of blood glucose levels is essential for effective diabetes management, helping patients and doctors make informed decisions about diet, exercise, and medication adjustments."
    ]

    # Generate synthetic QA data
    synthetic_qa_dataset = generate_qa_pairs_from_text(medical_texts, num_qa_per_chunk=2)

    print(f"\nGenerated {len(synthetic_qa_dataset)} synthetic QA pairs:")
    for i, example in enumerate(synthetic_qa_dataset.select(range(min(5, len(synthetic_qa_dataset))))):
        print(f"-- QA Pair {i+1} --")
        print(f"Context: {example['context'][:100]}...")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answer']}\n")

    # Save the generated dataset (e.g., to a JSON file)
    # synthetic_qa_dataset.to_json("synthetic_medical_qa.json")
    # print("Synthetic QA data saved to synthetic_medical_qa.json")
