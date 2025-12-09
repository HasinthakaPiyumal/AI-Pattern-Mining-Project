import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

# 1. Simulated Knowledge Base
medical_knowledge_base = [
    "Hypertension, or high blood pressure, is a common condition that can lead to serious health problems, such as heart disease, stroke, kidney failure, and other issues. It is often called a 'silent killer' because it usually has no symptoms.",
    "Kidney disease, also known as nephropathy, is the gradual loss of kidney function. Kidneys filter wastes and excess fluids from the blood, which are then excreted in the urine.",
    "Chronic hypertension can damage the small blood vessels in the kidneys, impairing their ability to filter blood effectively. This can progress to chronic kidney disease over many years.",
    "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy.",
    "Heart failure occurs when the heart muscle doesn't pump blood as well as it should. It can be caused by conditions like narrowed arteries in the heart (coronary artery disease) or high blood pressure.",
    "Treatment for hypertension often includes lifestyle changes such as diet and exercise, and medications like ACE inhibitors, ARBs, diuretics, and beta-blockers."
]

# 2. Embedding Model for Retriever
retriever_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_base_embeddings = retriever_embedding_model.encode(medical_knowledge_base, convert_to_tensor=True)

# 3. Retriever (Simplified using NearestNeighbors)
class SimpleRetriever:
    def __init__(self, knowledge_base_embeddings, knowledge_base_texts, k=3):
        self.knowledge_base_texts = knowledge_base_texts
        self.k = k
        self.nn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
        self.nn_model.fit(knowledge_base_embeddings.cpu().numpy())

    def retrieve(self, query_embedding):
        distances, indices = self.nn_model.kneighbors(query_embedding.cpu().numpy().reshape(1, -1))
        retrieved_passages = [self.knowledge_base_texts[i] for i in indices[0]]
        return retrieved_passages

simple_retriever = SimpleRetriever(knowledge_base_embeddings, medical_knowledge_base)

# 4. Generator (LLM) - T5 for demonstration
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Add special control token for statement reconstruction
special_tokens_dict = {'additional_special_tokens': ['[medical_reconstruct]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Dummy Data Generation
def generate_dummy_qa_data(num_samples=10):
    qa_data = []
    questions = [
        "What causes high blood pressure?",
        "How do kidneys function?",
        "What are the treatments for hypertension?",
        "What happens if hypertension is left untreated?"
    ]
    answers = [
        "High blood pressure, or hypertension, can be caused by various factors, including genetics, lifestyle, and underlying medical conditions. It's often asymptomatic.",
        "Kidneys filter waste products and excess fluids from the blood, producing urine. They are vital for maintaining fluid and electrolyte balance.",
        "Treatments for hypertension include lifestyle modifications (diet, exercise) and medications like ACE inhibitors, ARBs, diuretics, and beta-blockers.",
        "Untreated hypertension can lead to severe complications such as heart disease, stroke, kidney failure, and vision problems due to damage to blood vessels."
    ]
    for _ in range(num_samples):
        question = random.choice(questions)
        answer = random.choice(answers)
        qa_data.append({"question": question, "answer": answer})
    return qa_data

def generate_dummy_reconstruction_data(num_samples=10):
    reconstruction_data = []
    simplified_statements = [
        "Hypertension damages kidneys.",
        "Kidneys clean blood.",
        "Diabetes means high sugar.",
        "Heart failure is bad pumping."
    ]
    comprehensive_statements = [
        "Chronic hypertension can progressively damage the small blood vessels within the kidneys, leading to impaired filtration and potentially chronic kidney disease over time.",
        "The kidneys are vital organs responsible for filtering waste products and excess fluids from the blood to produce urine, maintaining overall fluid and electrolyte balance in the body.",
        "Diabetes mellitus is a metabolic disorder characterized by persistently high blood sugar levels, resulting from either insufficient insulin production or the body's ineffective use of insulin.",
        "Heart failure is a condition where the heart muscle is unable to pump sufficient blood to meet the body's demands, often stemming from conditions like coronary artery disease or prolonged high blood pressure."
    ]
    for _ in range(num_samples):
        simplified = random.choice(simplified_statements)
        comprehensive = random.choice(comprehensive_statements)
        reconstruction_data.append({"simplified": simplified, "comprehensive": comprehensive})
    return reconstruction_data

qa_dataset = generate_dummy_qa_data(20)
reconstruction_dataset = generate_dummy_reconstruction_data(20)

# 5. Multi-task Training Loop (Simplified Demonstration)
optimizer = AdamW(model.parameters(), lr=1e-4)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_qa_loss = 0
    total_reco_loss = 0

    # QA Task Training
    for item in qa_dataset:
        question = item["question"]
        answer = item["answer"]

        query_embedding = retriever_embedding_model.encode(question, convert_to_tensor=True)
        retrieved_contexts = simple_retriever.retrieve(query_embedding)
        context_str = " ".join(retrieved_contexts)

        input_text = f"question: {question} context: {context_str}"
        target_text = answer

        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
        labels = tokenizer(target_text, return_tensors='pt', max_length=512, truncation=True).input_ids

        outputs = model(**inputs, labels=labels)
        qa_loss = outputs.loss
        total_qa_loss += qa_loss.item()

        # In a real scenario, backpropagate and optimize here
        # qa_loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

    # Statement Reconstruction Task Training
    for item in reconstruction_dataset:
        simplified_statement = item["simplified"]
        comprehensive_statement = item["comprehensive"]

        query_embedding = retriever_embedding_model.encode(simplified_statement, convert_to_tensor=True)
        retrieved_contexts = simple_retriever.retrieve(query_embedding)
        context_str = " ".join(retrieved_contexts)

        input_text = f"[medical_reconstruct] {simplified_statement} context: {context_str}"
        target_text = comprehensive_statement

        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
        labels = tokenizer(target_text, return_tensors='pt', max_length=512, truncation=True).input_ids

        outputs = model(**inputs, labels=labels)
        reco_loss = outputs.loss
        total_reco_loss += reco_loss.item()

        # In a real scenario, backpropagate and optimize here
        # reco_loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

    print(f"Epoch {epoch+1}: QA Loss = {total_qa_loss / len(qa_dataset):.4f}, Reconstruction Loss = {total_reco_loss / len(reconstruction_dataset):.4f}")

# 6. Simplified Inference
model.eval()

def clinical_query(question):
    query_embedding = retriever_embedding_model.encode(question, convert_to_tensor=True)
    retrieved_contexts = simple_retriever.retrieve(query_embedding)
    context_str = " ".join(retrieved_contexts)
    input_text = f"question: {question} context: {context_str}"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def reconstruct_statement(simplified_statement):
    query_embedding = retriever_embedding_model.encode(simplified_statement, convert_to_tensor=True)
    retrieved_contexts = simple_retriever.retrieve(query_embedding)
    context_str = " ".join(retrieved_contexts)
    input_text = f"[medical_reconstruct] {simplified_statement} context: {context_str}"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Inference Examples ---")
qa_question = "What are some common treatments for high blood pressure?"
qa_answer = clinical_query(qa_question)
print(f"Question: {qa_question}")
print(f"Answer: {qa_answer}")

reco_statement = "Kidney failure is serious."
reconstructed_output = reconstruct_statement(reco_statement)
print(f"\nSimplified Statement: {reco_statement}")
print(f"Reconstructed: {reconstructed_output}")

reco_statement_2 = "Hypertension harms the heart."
reconstructed_output_2 = reconstruct_statement(reco_statement_2)
print(f"\nSimplified Statement: {reco_statement_2}")
print(f"Reconstructed: {reconstructed_output_2}")
