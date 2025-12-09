from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

def generate_synthetic_qa(product_description, model, tokenizer, num_exemplars=3):
    synthetic_qa_pairs = []
    prompt_template = (
        f"Based on the following product description, generate {num_exemplars} unique customer questions and their accurate answers. "
        "Each question should be followed by its answer. Format strictly as 'Q: [Question]\nA: [Answer]'.\n\n"
        f"Product Description: {product_description}\n\n"
    )

    input_ids = tokenizer.encode(prompt_template, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_new_tokens=200 * num_exemplars,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    generated_qa_section = generated_text[len(prompt_template):].strip()

    qa_matches = re.findall(r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)', generated_qa_section, re.DOTALL)

    for q, a in qa_matches:
        synthetic_qa_pairs.append({"query": q.strip(), "answer": a.strip()})
        if len(synthetic_qa_pairs) >= num_exemplars:
            break

    return synthetic_qa_pairs

def answer_customer_query(customer_query, few_shot_exemplars, model, tokenizer):
    few_shot_prompt_parts = []
    for exemplar in few_shot_exemplars:
        few_shot_prompt_parts.append(f"Q: {exemplar['query']}\nA: {exemplar['answer']}")
    
    few_shot_context = "\n\n".join(few_shot_prompt_parts)

    full_prompt = (
        f"{few_shot_context}\n\n"
        f"Customer Query: {customer_query}\n"
        "Answer:"
    )

    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    answer_start_index = generated_text.rfind("Answer:")
    if answer_start_index != -1:
        return generated_text[answer_start_index + len("Answer:"):].strip()
    return generated_text.strip()

if __name__ == "__main__":
    print("Loading GPT2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.")

    product_descriptions = {
        "Smartwatch": "The new XYZ Smartwatch features a 1.5-inch AMOLED display, heart rate monitoring, GPS, and 5-day battery life. It's water-resistant up to 50 meters and supports contactless payments.",
        "Wireless Earbuds": "Experience crystal-clear audio with the AuraFlow Wireless Earbuds. Boasting 8 hours of battery life, a comfortable ergonomic design, and active noise cancellation. Comes with a portable charging case providing an additional 24 hours."
    }

    all_synthetic_exemplars = []

    print("\nGenerating synthetic Q&A pairs for products...")
    for product_name, description in product_descriptions.items():
        print(f"\n--- Generating for {product_name} ---")
        exemplars = generate_synthetic_qa(description, model, tokenizer, num_exemplars=2)
        all_synthetic_exemplars.extend(exemplars)
        for qa in exemplars:
            print(f"  Q: {qa['query']}")
            print(f"  A: {qa['answer']}")
    
    print("\nAll synthetic exemplars generated:")
    for i, qa in enumerate(all_synthetic_exemplars):
        print(f"{i+1}. Q: {qa['query']} A: {qa['answer']}")

    print("\nSimulating customer queries with few-shot prompting...")
    customer_queries = [
        "How long does the smartwatch battery last?",
        "Can I swim with the XYZ Smartwatch?",
        "Do the AuraFlow earbuds have noise cancellation?",
        "What is the total battery life of the AuraFlow earbuds including the case?"
    ]

    for query in customer_queries:
        print(f"\nCustomer: {query}")
        answer = answer_customer_query(query, all_synthetic_exemplars, model, tokenizer)
        print(f"Chatbot: {answer}")