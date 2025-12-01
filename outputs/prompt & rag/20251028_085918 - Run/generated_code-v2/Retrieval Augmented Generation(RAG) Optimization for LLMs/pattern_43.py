
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import faiss
import numpy as np

# --- 1. Configuration ---
MODEL_NAME = "distilbert-base-uncased" # Using DistilBERT for embeddings
K_NEIGHBORS = 5
LAMBDA = 0.5 # Interpolation weight for kNN distribution
MAX_GENERATION_LENGTH = 50

# --- 2. Medical Corpus Simulation ---
medical_corpus_sentences = [
    "Aspirin is commonly used for pain relief and reducing fever.",
    "Hypertension, or high blood pressure, increases the risk of heart disease.",
    "Diabetes mellitus is a chronic metabolic disease characterized by elevated blood glucose levels.",
    "The human heart has four chambers: two atria and two ventricles.",
    "Vaccination is a safe and effective way to prevent infectious diseases.",
    "Cardiovascular diseases are the leading cause of death globally.",
    "Antibiotics are medications that fight bacterial infections.",
    "Regular exercise and a balanced diet are crucial for maintaining good health.",
    "Symptoms of a common cold include a runny nose, sore throat, and sneezing.",
    "MRI scans use strong magnetic fields and radio waves to create detailed images of organs and tissues.",
    "Early diagnosis and treatment are vital for improving cancer outcomes.",
    "Inflammation is the body's immune response to infection or injury.",
    "Osteoporosis is a condition that causes bones to become weak and brittle.",
    "Kidney failure requires dialysis or a kidney transplant.",
    "Neurological disorders affect the brain, spinal cord, and nerves."
]

# --- 3. Load Language Model and Tokenizer ---
# Using a generic BERT-like model for embeddings
embedding_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME)
embedding_model.eval() # Set to evaluation mode

# --- 4. Build FAISS Index from Medical Corpus ---
print("Building FAISS index from medical corpus...")

corpus_embeddings = []
corpus_token_ids = []
corpus_text_tokens = [] # Store text tokens for debugging/understanding

for sentence in medical_corpus_sentences:
    encoded_input = embedding_tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**encoded_input)
        token_embeddings = outputs.last_hidden_state[0] # Embeddings for tokens in the sentence

    for i in range(len(encoded_input['input_ids'][0])):
        token_id = encoded_input['input_ids'][0][i].item()
        corpus_token_ids.append(token_id)
        corpus_embeddings.append(token_embeddings[i].cpu().numpy())
        corpus_text_tokens.append(embedding_tokenizer.decode(token_id))

corpus_embeddings = np.array(corpus_embeddings).astype('float32')
dimension = corpus_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension) # Using L2 distance for similarity
faiss_index.add(corpus_embeddings)

print(f"FAISS index built with {faiss_index.ntotal} embeddings.")

# --- 5. kNN-LM Inference Function ---

def generate_knn_lm_response(prompt: str, k: int, lambda_weight: float, max_length: int) -> str:
    generated_ids = []
    current_input_ids = embedding_tokenizer.encode(prompt, return_tensors='pt')
    generated_ids.extend(current_input_ids[0].tolist()) # Include prompt in generated text

    print(f"\nPrompt: {prompt}")
    print(f"Generating with kNN-LM (k={k}, lambda={lambda_weight})...")

    for _ in range(max_length):
        # 1. Get embedding of the last token (query token)
        last_token_id = current_input_ids[0, -1].item()
        with torch.no_grad():
            outputs = embedding_model(current_input_ids)
            query_embedding = outputs.last_hidden_state[0, -1, :].cpu().numpy().reshape(1, -1)

        # 2. Search FAISS for k nearest neighbors
        distances, indices = faiss_index.search(query_embedding, k)

        # 3. Derive kNN-induced distribution (simplified as boosting logits)
        knn_logits = torch.full((embedding_tokenizer.vocab_size,), -1e9, dtype=torch.float) # Initialize with very low logits
        
        neighbor_token_ids = []
        for idx in indices[0]:
            if 0 <= idx < len(corpus_token_ids): # Ensure index is valid
                neighbor_token_ids.append(corpus_token_ids[idx])

        unique_neighbors, counts = np.unique(neighbor_token_ids, return_counts=True)
        
        for i, token_id in enumerate(unique_neighbors):
            score = counts[i] # Direct count as score
            knn_logits[token_id] = torch.log(torch.tensor(score + 1, dtype=torch.float)) # Add 1 to avoid log(0)

        # 4. Simulate P_lm logits (very basic for this example - replace with actual CausalLM logits)
        lm_logits_dummy = torch.randn(embedding_tokenizer.vocab_size) * 0.1 # Small random values
        
        # Optional: Add a slight artificial bias for demo to make it less purely random
        if last_token_id == embedding_tokenizer.encode("disease")[1]:
            lm_logits_dummy[embedding_tokenizer.encode("is")[1]] += 2.0
            lm_logits_dummy[embedding_tokenizer.encode("are")[1]] += 1.5

        # 5. Interpolate
        interpolated_logits = (1 - lambda_weight) * lm_logits_dummy + lambda_weight * knn_logits

        # 6. Sample next token (Greedy decoding)
        next_token_id = torch.argmax(interpolated_logits).item()

        # Add to generated sequence and update input
        generated_ids.append(next_token_id)
        current_input_ids = torch.tensor([generated_ids]).to(current_input_ids.device)

        # Check for end-of-sequence token 
        if next_token_id == embedding_tokenizer.sep_token_id or next_token_id == embedding_tokenizer.eos_token_id:
            break

    # Decode the full generated sequence
    return embedding_tokenizer.decode(generated_ids, skip_special_tokens=True)

# --- 6. Example Usage ---
if __name__ == "__main__":
    test_prompt = "Symptoms of "
    response = generate_knn_lm_response(test_prompt, k=K_NEIGHBORS, lambda_weight=LAMBDA, max_length=MAX_GENERATION_LENGTH)
    print(f"\nGenerated Response: {response}")

    test_prompt_2 = "Cardiovascular "
    response_2 = generate_knn_lm_response(test_prompt_2, k=K_NEIGHBORS, lambda_weight=LAMBDA, max_length=MAX_GENERATION_LENGTH)
    print(f"\nGenerated Response 2: {response_2}")
