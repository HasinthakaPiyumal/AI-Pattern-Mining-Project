import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DPRQuestionEncoder, DPRContextEncoder, BartForConditionalGeneration, BartTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import faiss
import numpy as np
import random

# --- Configuration ---
QUESTION_ENCODER_MODEL = "facebook/dpr-question_encoder-single-nq-base"
CONTEXT_ENCODER_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
GENERATOR_MODEL = "facebook/bart-base"
GENERATOR_TOKENIZER = "facebook/bart-base"

BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQUENCE_LENGTH = 128
REINDEX_FREQUENCY = 1 # Re-index after every epoch for demonstration
TOP_K_RETRIEVAL = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Mock Data Generation ---
class MedicalKnowledgeBase:
    def __init__(self, num_passages=100):
        self.passages = [f"Medical passage about disease {i} and its treatment options. This is a detailed explanation focusing on various aspects of health and patient care." for i in range(num_passages)]
        self.ids = list(range(num_passages))

    def get_passages(self):
        return self.passages

    def get_passage_by_id(self, passage_id):
        return self.passages[passage_id]

class MedicalQADataset(Dataset):
    def __init__(self, knowledge_base, num_samples=50):
        self.questions = []
        self.answers = []
        self.positive_passage_ids = []

        for i in range(num_samples):
            q = f"What is the treatment for disease {i % len(knowledge_base.passages)}?"
            pos_id = i % len(knowledge_base.passages)
            ans = f"The treatment for disease {pos_id} involves specific procedures detailed in medical passage {pos_id}."
            
            self.questions.append(q)
            self.answers.append(ans)
            self.positive_passage_ids.append(pos_id)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "answer": self.answers[idx],
            "positive_passage_id": self.positive_passage_ids[idx]
        }

# --- Model Initialization ---
question_encoder = DPRQuestionEncoder.from_pretrained(QUESTION_ENCODER_MODEL).to(device)
context_encoder = DPRContextEncoder.from_pretrained(CONTEXT_ENCODER_MODEL).to(device)
generator = BartForConditionalGeneration.from_pretrained(GENERATOR_MODEL).to(device)
generator_tokenizer = BartTokenizer.from_pretrained(GENERATOR_TOKENIZER)

# --- FAISS Index Management ---
def build_faiss_index(context_encoder_model, knowledge_base_passages, passage_tokenizer, batch_size=32):
    context_encoder_model.eval()
    passage_embeddings = []
    for i in range(0, len(knowledge_base_passages), batch_size):
        batch_passages = knowledge_base_passages[i:i + batch_size]
        inputs = passage_tokenizer(batch_passages, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(device)
        with torch.no_grad():
            embeddings = context_encoder_model(**inputs).pooler_output.cpu().numpy()
        passage_embeddings.append(embeddings)
    
    passage_embeddings = np.vstack(passage_embeddings)
    index = faiss.IndexFlatIP(passage_embeddings.shape[1])
    index.add(passage_embeddings)
    return index

# --- Training Loop Helper Functions ---
def get_negative_passages(positive_passage_id, knowledge_base, num_negatives=4):
    all_passage_ids = set(knowledge_base.ids)
    available_negatives = list(all_passage_ids - {positive_passage_id})
    if len(available_negatives) < num_negatives:
        return [knowledge_base.get_passage_by_id(pid) for pid in available_negatives]
    return [knowledge_base.get_passage_by_id(pid) for pid in random.sample(available_negatives, num_negatives)]

# --- Main Training and Inference --- 

medical_kb = MedicalKnowledgeBase()
qa_dataset = MedicalQADataset(medical_kb)
qa_dataloader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize FAISS index
print("Building initial FAISS index...")
passage_tokenizer_for_faiss = BartTokenizer.from_pretrained(CONTEXT_ENCODER_MODEL)
faiss_index = build_faiss_index(context_encoder, medical_kb.get_passages(), passage_tokenizer_for_faiss)
print("Initial FAISS index built.")

# Optimizer and Scheduler
optimizer = AdamW(list(question_encoder.parameters()) + list(context_encoder.parameters()) + list(generator.parameters()), lr=LEARNING_RATE)
total_steps = len(qa_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
print("Starting End-to-End Training...")
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    question_encoder.train()
    context_encoder.train()
    generator.train()
    total_loss = 0
    for batch_idx, batch in enumerate(qa_dataloader):
        optimizer.zero_grad()

        questions = batch["question"]
        positive_passage_ids = batch["positive_passage_id"]
        answers = batch["answer"]

        # 1. Encode questions
        question_inputs = generator_tokenizer(questions, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(device)
        question_embeddings = question_encoder(**question_inputs).pooler_output

        # 2. Encode positive and negative passages for retrieval loss
        batch_passages_for_retrieval = []
        for i, pos_id in enumerate(positive_passage_ids):
            batch_passages_for_retrieval.append(medical_kb.get_passage_by_id(pos_id.item()))
            negative_passages = get_negative_passages(pos_id.item(), medical_kb, num_negatives=random.randint(1, 4))
            batch_passages_for_retrieval.extend(negative_passages)
        
        passage_inputs_for_retrieval = generator_tokenizer(batch_passages_for_retrieval, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(device)
        passage_embeddings_for_retrieval = context_encoder(**passage_inputs_for_retrieval).pooler_output

        # Arrange embeddings for contrastive loss (simplified)
        # For each question, we have one positive and potentially multiple negatives
        retrieval_scores_list = []
        passage_embedding_idx = 0
        for i, _ in enumerate(questions):
            q_emb = question_embeddings[i:i+1]
            pos_p_emb = passage_embeddings_for_retrieval[passage_embedding_idx:passage_embedding_idx+1]
            passage_embedding_idx += 1
            
            num_negatives_in_batch = len(batch_passages_for_retrieval) - (len(questions) + passage_embedding_idx - 1)
            current_negatives = []
            for _ in range(num_negatives_in_batch):
                current_negatives.append(passage_embeddings_for_retrieval[passage_embedding_idx:passage_embedding_idx+1])
                passage_embedding_idx += 1
            
            if current_negatives:
                all_p_embs = torch.cat([pos_p_emb] + current_negatives, dim=0)
            else:
                all_p_embs = pos_p_emb

            scores = torch.matmul(q_emb, all_p_embs.transpose(0, 1))
            retrieval_scores_list.append(scores)
            
        # This part requires careful handling of targets for a proper contrastive loss
        # For simplicity, we will use a dummy target assuming the first passage is positive
        # In a real scenario, this would be a more complex NCE or InfoNCE loss.
        if retrieval_scores_list:
            retrieval_scores = torch.cat(retrieval_scores_list, dim=0)
            # Dummy labels: assume first item is positive for each question
            retrieval_labels = torch.zeros(retrieval_scores.size(0), dtype=torch.long, device=device)
            retrieval_loss = torch.nn.functional.cross_entropy(retrieval_scores, retrieval_labels)
        else:
            retrieval_loss = torch.tensor(0.0).to(device)

        # 3. Retrieve top-k passages using current FAISS index
        question_embeddings_np = question_embeddings.cpu().detach().numpy()
        D, I = faiss_index.search(question_embeddings_np, TOP_K_RETRIEVAL)

        retrieved_passages_batch = []
        for i_batch in range(len(questions)):
            retrieved_passage_indices = I[i_batch]
            combined_passage_text = " ".join([medical_kb.get_passage_by_id(idx) for idx in retrieved_passage_indices if idx != -1])
            retrieved_passages_batch.append(combined_passage_text)
        
        # 4. Prepare input for generator
        generator_inputs = [q + " [SEP] " + rp for q, rp in zip(questions, retrieved_passages_batch)]
        
        model_inputs = generator_tokenizer(generator_inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(device)
        labels = generator_tokenizer(answers, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQUENCE_LENGTH).input_ids.to(device)

        # 5. Generate and calculate generator loss
        outputs = generator(**model_inputs, labels=labels)
        generator_loss = outputs.loss

        # 6. Combine losses and backpropagate
        # A weighted sum might be used here in a real scenario
        combined_loss = generator_loss + retrieval_loss
        combined_loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += combined_loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(qa_dataloader)} - Loss: {combined_loss.item():.4f} (Gen: {generator_loss.item():.4f}, Ret: {retrieval_loss.item():.4f})")

    avg_loss = total_loss / len(qa_dataloader)
    print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    # Dynamic KB Re-indexing
    if (epoch + 1) % REINDEX_FREQUENCY == 0:
        print(f"Re-encoding Medical Knowledge Base and rebuilding FAISS index after epoch {epoch + 1}...")
        faiss_index = build_faiss_index(context_encoder, medical_kb.get_passages(), passage_tokenizer_for_faiss)
        print("FAISS index rebuilt.")

print("Training complete.")

# --- Inference Pipeline ---
def medrag_inference(query, question_encoder_model, context_encoder_model, generator_model, generator_tokenizer_obj, faiss_idx, knowledge_base, top_k=TOP_K_RETRIEVAL):
    question_encoder_model.eval()
    context_encoder_model.eval()
    generator_model.eval()

    # 1. Encode query
    query_inputs = generator_tokenizer_obj([query], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(device)
    with torch.no_grad():
        query_embedding = question_encoder_model(**query_inputs).pooler_output.cpu().numpy()

    # 2. Retrieve top-k passages
    D, I = faiss_idx.search(query_embedding, top_k)
    retrieved_passages_text = [knowledge_base.get_passage_by_id(idx) for idx in I[0] if idx != -1]
    combined_retrieved_text = " ".join(retrieved_passages_text)

    # 3. Generate answer
    generator_input = query + " [SEP] " + combined_retrieved_text
    generator_model_inputs = generator_tokenizer_obj([generator_input], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(device)
    
    with torch.no_grad():
        generated_ids = generator_model.generate(generator_model_inputs.input_ids, max_length=50, num_beams=2, early_stopping=True)
    
    answer = generator_tokenizer_obj.decode(generated_ids[0], skip_special_tokens=True)
    return answer, retrieved_passages_text

print("\n--- Demonstrating Inference ---")
medical_question = "What is the recommended treatment for disease 5?"
answer, retrieved_passages = medrag_inference(medical_question, question_encoder, context_encoder, generator, generator_tokenizer, faiss_index, medical_kb)

print(f"Question: {medical_question}")
print(f"Retrieved Passages: {retrieved_passages}")
print(f"Generated Answer: {answer}")

medical_question_2 = "Tell me about the causes of disease 90."
answer_2, retrieved_passages_2 = medrag_inference(medical_question_2, question_encoder, context_encoder, generator, generator_tokenizer, faiss_index, medical_kb)

print(f"\nQuestion: {medical_question_2}")
print(f"Retrieved Passages: {retrieved_passages_2}")
print(f"Generated Answer: {answer_2}")

