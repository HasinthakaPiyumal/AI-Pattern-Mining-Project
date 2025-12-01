import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import Dataset
from trl import SFTTrainer
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. Data Preparation Module ---

def load_dummy_ecommerce_data():
    """Generates a dummy dataset simulating e-commerce product info and customer queries."""
    products = [
        {"id": "P001", "name": "Wireless Bluetooth Headphones", "description": "High-quality sound, comfortable earcups, 20-hour battery life. Compatible with all smartphones. Includes charging cable and carrying case. Model: BH-200. Price: $99.99.", "category": "Electronics"},
        {"id": "P002", "name": "Smart Fitness Tracker", "description": "Monitors heart rate, steps, calories burned, sleep patterns. Water-resistant. Syncs with app. Available in black and blue. Model: FT-101. Price: $49.99.", "category": "Wearables"},
        {"id": "P003", "name": "Portable USB Charger (10000mAh)", "description": "Fast charging for phones and tablets. Dual USB-A ports. LED indicator. Compact design. Model: PC-500. Price: $29.99.", "category": "Electronics"},
        {"id": "P004", "name": "Ergonomic Office Chair", "description": "Adjustable lumbar support, breathable mesh back, swivel function. High-density foam seat. Max weight 250lbs. Model: OC-PRO. Price: $149.99.", "category": "Home Office"},
        {"id": "P005", "name": "Organic Green Tea Variety Pack", "description": "Includes 4 unique blends: Jasmine, Sencha, Matcha, and Peppermint. 20 tea bags per blend. USDA Organic certified. Price: $19.99.", "category": "Groceries"},
        {"id": "P006", "name": "Noise-Cancelling Earbuds", "description": "Premium noise cancellation, secure fit for workouts, touch controls. 8-hour battery. Model: NE-300. Price: $79.99.", "category": "Electronics"},
        {"id": "P007", "name": "4K Ultra HD Smart TV (55 inch)", "description": "Stunning 4K resolution, HDR support, built-in streaming apps. Multiple HDMI ports. Model: TV-550. Price: $499.99.", "category": "Electronics"},
        {"id": "P008", "name": "Stainless Steel Water Bottle (1L)", "description": "Double-walled insulation keeps drinks cold for 24 hours, hot for 12 hours. Leak-proof cap. BPA-free. Price: $15.00.", "category": "Kitchen"},
    ]

    faqs = [
        {"question": "How long does the battery last on the BH-200 headphones?", "answer": "The BH-200 headphones have a 20-hour battery life."},
        {"question": "Is the FT-101 fitness tracker waterproof?", "answer": "Yes, the FT-101 fitness tracker is water-resistant."},
        {"question": "What colors are available for the FT-101 fitness tracker?", "answer": "The FT-101 fitness tracker is available in black and blue."},
        {"question": "Can I charge my tablet with the PC-500 power bank?", "answer": "Yes, the PC-500 portable USB charger is designed for fast charging phones and tablets."},
        {"question": "What is the maximum weight capacity for the OC-PRO office chair?", "answer": "The OC-PRO office chair has a maximum weight capacity of 250lbs."},
        {"question": "Are the green teas organic?", "answer": "Yes, the Organic Green Tea Variety Pack is USDA Organic certified."},
        {"question": "Do the NE-300 earbuds have noise cancellation?", "answer": "Yes, the NE-300 earbuds feature premium noise cancellation."},
        {"question": "What is the resolution of the TV-550 smart TV?", "answer": "The TV-550 smart TV has stunning 4K Ultra HD resolution."},
    ]

    # Combine product descriptions and FAQs into a single corpus for retrieval
    corpus = [p["description"] for p in products] + [f["question"] + " " + f["answer"] for f in faqs]
    return products, faqs, corpus

def generate_rag_training_data(products, faqs, corpus, num_hard_negatives_per_example=1):
    """
    Generates retrieval-augmented QA data with gold contexts and hard-negative contexts.
    """
    training_examples = []
    product_descriptions_map = {p["id"]: p["description"] for p in products}

    # Use TF-IDF for simulating retrieval and finding hard negatives
    vectorizer = TfidfVectorizer().fit(corpus)
    corpus_vectors = vectorizer.transform(corpus)

    for faq in faqs:
        question = faq["question"]
        gold_answer = faq["answer"]

        # Find gold context (from product descriptions or other FAQs that contains the answer)
        gold_contexts = [
            doc for doc in corpus if gold_answer.lower() in doc.lower()
        ]
        if not gold_contexts:
            # Fallback: if answer is not explicitly in corpus, use the FAQ answer itself as context
            gold_contexts = [gold_answer]

        gold_context = random.choice(gold_contexts) # Pick one gold context

        # Simulate retrieval for hard negatives
        query_vector = vectorizer.transform([question])
        similarities = cosine_similarity(query_vector, corpus_vectors).flatten()
        
        # Sort contexts by similarity and filter out gold context and contexts containing the answer
        retrieved_indices_sorted = similarities.argsort()[::-1]
        
        hard_negatives = []
        for idx in retrieved_indices_sorted:
            doc = corpus[idx]
            if gold_answer.lower() not in doc.lower() and doc.strip() != gold_context.strip():
                # This is a hard negative if it doesn't contain the answer but is somewhat similar
                hard_negatives.append(doc)
            if len(hard_negatives) >= num_hard_negatives_per_example:
                break
        
        # If not enough hard negatives, just pick random non-answer docs
        while len(hard_negatives) < num_hard_negatives_per_example:
            random_doc = random.choice(corpus)
            if gold_answer.lower() not in random_doc.lower() and random_doc.strip() != gold_context.strip():
                hard_negatives.append(random_doc)
            # Prevent infinite loop if corpus is too small or all docs contain answer
            if len(set(hard_negatives)) == len(set(doc for doc in corpus if gold_answer.lower() not in doc.lower() and doc.strip() != gold_context.strip())):
                break
        
        hard_negatives_str = " ".join(hard_negatives[:num_hard_negatives_per_example]) if hard_negatives else ""

        # Construct the instruction-tuning prompt
        # We combine gold and hard-negative contexts, letting the model learn to discern.
        # The order can be randomized to prevent positional bias.
        contexts_for_model = [gold_context] + hard_negatives[:num_hard_negatives_per_example]
        random.shuffle(contexts_for_model)
        combined_context = "\n".join(contexts_for_model)

        instruction = "Given the following context, answer the question accurately and concisely. If the answer is not in the context, state that you cannot find the answer."
        input_text = f"Context: {combined_context}\nQuestion: {question}"
        output_text = gold_answer # The LLM should learn to extract this from the relevant parts of combined_context

        training_examples.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    return training_examples

# --- 2. LLM Training Module ---

def train_rag_llm(training_data, model_name="google/flan-t5-small", output_dir="./rag_llm_robustness_model"):
    """
    Fine-tunes an LLM using the retrieval-augmented QA data.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Convert list of dicts to Hugging Face Dataset
    dataset = Dataset.from_list(training_data)

    def formatting_prompts_func(example):
        text = f"{example['instruction']}\n### Input:\n{example['input']}\n### Output:\n{example['output']}"
        return {"text": text}

    formatted_dataset = dataset.map(formatting_prompts_func)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2, # Small batch size for demo
        num_train_epochs=3, # A few epochs for demonstration
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        learning_rate=2e-5,
        evaluation_strategy="no", # For simplicity, no evaluation in this single-file demo
        report_to="none",
    )

    # Use SFTTrainer for supervised fine-tuning
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func, # Pass the formatting function
        max_seq_length=512, # Max sequence length for T5-small
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer

# --- 3. RAG Inference Module (Chatbot) ---

class RAGChatbot:
    def __init__(self, model_path, corpus):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval() # Set model to evaluation mode
        self.corpus = corpus
        # Simple keyword-based retriever for demo. In real-world, use vector DB + embedding model.
        self.vectorizer = TfidfVectorizer().fit(self.corpus)
        self.corpus_vectors = self.vectorizer.transform(self.corpus)

    def retrieve(self, query, top_k=3):
        """Retrieves top_k relevant contexts from the corpus."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.corpus_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        retrieved_contexts = [self.corpus[i] for i in top_indices]
        return retrieved_contexts

    def generate_response(self, query):
        """Generates a response using the RAG model."""
        retrieved_contexts = self.retrieve(query)
        combined_context = "\n".join(retrieved_contexts)

        instruction = "Given the following context, answer the question accurately and concisely. If the answer is not in the context, state that you cannot find the answer."
        input_text = f"Context: {combined_context}\nQuestion: {query}"
        
        # Prepare input for the model
        prompt = f"{instruction}\n### Input:\n{input_text}\n### Output:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                num_beams=4,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to remove potential prompt leakage if model isn't perfectly trained
        if response.startswith(instruction):
            response = response.replace(instruction, "", 1).strip()
        if response.startswith("### Input:"):
            response = re.sub(r"### Input:.*", "", response, flags=re.DOTALL).strip()
        return response

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Irrelevant Context Robustness Training Demo ---")

    # 1. Data Preparation
    print("\n1. Preparing dummy e-commerce data and generating RAG training examples...")
    products, faqs, corpus = load_dummy_ecommerce_data()
    rag_training_data = generate_rag_training_data(products, faqs, corpus, num_hard_negatives_per_example=2)
    
    print(f"Generated {len(rag_training_data)} training examples.")
    print("\nExample Training Data Point:")
    print(json.dumps(rag_training_data[0], indent=2))

    # 2. LLM Training
    print("\n2. Training the RAG LLM (this might take a few minutes)...")
    trained_model, trained_tokenizer = train_rag_llm(rag_training_data, output_dir="./rag_llm_robustness_model")
    print("Training complete. Model saved to ./rag_llm_robustness_model")

    # 3. RAG Inference (Chatbot)
    print("\n3. Initializing RAG Chatbot for inference...")
    chatbot = RAGChatbot(model_path="./rag_llm_robustness_model", corpus=corpus)
    print("Chatbot ready. Type your queries (e.g., 'What is the battery life of the BH-200 headphones?', 'Is the fitness tracker waterproof?', 'Tell me about the organic tea.')")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nCustomer: ")
        if user_query.lower() == 'exit':
            break
        
        print("Chatbot (retrieving and generating response)...")
        response = chatbot.generate_response(user_query)
        print(f"Bot: {response}")

    print("\n--- Demo Finished ---")
