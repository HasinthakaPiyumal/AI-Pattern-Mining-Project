import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import chromadb
from torch.utils.data import Dataset, DataLoader
import random

# --- 1. Data Ingestion & Knowledge Base Creation ---
class MedicalKnowledgeBase:
    def __init__(self, db_path="./medical_chroma_db", embedding_model_name="all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="medical_passages")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.passage_id_counter = 0

    def add_documents(self, documents):
        # For simplicity, we'll treat each document as a single passage.
        # In a real scenario, documents would be chunked more intelligently.
        new_ids = []
        new_documents = []
        new_embeddings = []

        for doc in documents:
            doc_id = str(self.passage_id_counter)
            self.passage_id_counter += 1
            new_ids.append(doc_id)
            new_documents.append(doc)
            new_embeddings.append(self.embedding_model.encode(doc).tolist())

        if new_ids:
            self.collection.add(documents=new_documents, embeddings=new_embeddings, ids=new_ids)
            print(f"Added {len(new_ids)} documents to the knowledge base.")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return results['documents'][0] if results['documents'] else []

# --- 2. RAG Model Architecture ---
class MedicalRAGModel(torch.nn.Module):
    def __init__(self, knowledge_base: MedicalKnowledgeBase, model_name="google/flan-t5-small", max_length=512):
        super().__init__()
        self.knowledge_base = knowledge_base
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_length = max_length

        # Add control tokens to the tokenizer and model if they don't exist
        self.qa_token = "[QA]"
        self.reconstruct_token = "[RECONSTRUCT_MEDICAL]"
        num_added_tokens = self.tokenizer.add_tokens([self.qa_token, self.reconstruct_token])
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"Added {num_added_tokens} new tokens to the tokenizer and resized model embeddings.")

    def _format_prompt(self, task_token: str, query_or_statement: str, retrieved_context: list[str]) -> str:
        context_str = "\n".join([f"Context: {c}" for c in retrieved_context])
        return f"{task_token} Query: {query_or_statement}\n{context_str}\nAnswer:"

    def generate_answer(self, question: str, k: int = 3) -> str:
        retrieved_passages = self.knowledge_base.retrieve(question, k=k)
        if not retrieved_passages:
            return "No relevant information found in the knowledge base."

        prompt = self._format_prompt(self.qa_token, question, retrieved_passages)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length, truncation=True)
        output_ids = self.model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True)
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Post-process to remove the input prompt part from the generated text
        return generated_text.replace(prompt, "").strip()

    def reconstruct_statement(self, statement: str, k: int = 3) -> str:
        retrieved_passages = self.knowledge_base.retrieve(statement, k=k)
        if not retrieved_passages:
            return "No relevant information found for reconstruction."

        prompt = self._format_prompt(self.reconstruct_token, statement, retrieved_passages)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length, truncation=True)
        output_ids = self.model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True)
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Post-process to remove the input prompt part from the generated text
        return generated_text.replace(prompt, "").strip()

    def forward(self, input_ids, attention_mask, labels=None):
        # Simplified forward pass for training
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

# --- 3. Multi-Task Training Pipeline (Data Preparation) ---
class MultiTaskMedicalDataset(Dataset):
    def __init__(self, qa_data: list[dict], reconstruction_data: list[str], tokenizer, knowledge_base: MedicalKnowledgeBase, max_length=512):
        self.qa_data = qa_data
        self.reconstruction_data = reconstruction_data
        self.tokenizer = tokenizer
        self.knowledge_base = knowledge_base
        self.max_length = max_length
        self.qa_token = "[QA]"
        self.reconstruct_token = "[RECONSTRUCT_MEDICAL]"

    def __len__(self):
        return len(self.qa_data) + len(self.reconstruction_data)

    def __getitem__(self, idx):
        if idx < len(self.qa_data):
            # QA Task
            sample = self.qa_data[idx]
            question = sample["question"]
            true_answer = sample["answer"]
            retrieved_passages = self.knowledge_base.retrieve(question, k=3)
            input_text = MedicalRAGModel._format_prompt(self, self.qa_token, question, retrieved_passages)
            target_text = true_answer # The LLM will learn to generate this after the prompt
        else:
            # Statement Reconstruction Task
            sample_idx = idx - len(self.qa_data)
            statement = self.reconstruction_data[sample_idx]
            retrieved_passages = self.knowledge_base.retrieve(statement, k=3)
            input_text = MedicalRAGModel._format_prompt(self, self.reconstruct_token, statement, retrieved_passages)
            target_text = statement # The LLM will learn to reconstruct this statement

        # Tokenize input and target for training
        model_inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        labels = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt").input_ids

        # Replace padding token id with -100 for `Trainer` to ignore it during loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

# --- Main Execution / Example Usage ----
if __name__ == "__main__":
    # 1. Initialize Knowledge Base and add some dummy medical documents
    kb = MedicalKnowledgeBase()
    medical_docs = [
        "Mitochondrial dysfunction is increasingly recognized as a key player in the pathogenesis of various neurodegenerative disorders, including Parkinson's disease, Alzheimer's disease, and Huntington's disease. Impaired mitochondrial function can lead to increased oxidative stress, energy deficits, and apoptosis.",
        "Alzheimer's disease is characterized by the accumulation of amyloid-beta plaques and neurofibrillary tangles in the brain, leading to progressive cognitive decline. Current treatments primarily focus on symptom management.",
        "Parkinson's disease is a progressive disorder of the nervous system affecting movement. Symptoms gradually appear, sometimes starting with a barely noticeable tremor in one limb. The disease is caused by a loss of dopamine-producing neurons in the substantia nigra.",
        "The human circulatory system is responsible for transporting blood, nutrients, oxygen, carbon dioxide, and hormones throughout the body. It consists of the heart, blood vessels, and blood.",
        "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells for storage or energy. With diabetes, your body either doesn't make enough insulin or can't effectively use the insulin it does make."
    ]
    kb.add_documents(medical_docs)

    # 2. Initialize RAG Model
    # Using a small model for demonstration. For real application, consider larger models (e.g., Llama-2).
    rag_model = MedicalRAGModel(knowledge_base=kb, model_name="google/flan-t5-small")
    
    # Example QA usage
    print("\n--- QA Example ---")
    question = "What is mitochondrial dysfunction associated with?"
    answer = rag_model.generate_answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    question = "What causes Parkinson's disease?"
    answer = rag_model.generate_answer(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")

    # Example Statement Reconstruction usage
    print("\n--- Statement Reconstruction Example ---")
    statement_to_reconstruct = "Mitochondrial issues are linked to brain diseases."
    reconstructed_statement = rag_model.reconstruct_statement(statement_to_reconstruct)
    print(f"Original Statement: {statement_to_reconstruct}")
    print(f"Reconstructed Statement: {reconstructed_statement}")

    statement_to_reconstruct_2 = "High blood sugar is a symptom of diabetes."
    reconstructed_statement_2 = rag_model.reconstruct_statement(statement_to_reconstruct_2)
    print(f"\nOriginal Statement: {statement_to_reconstruct_2}")
    print(f"Reconstructed Statement: {reconstructed_statement_2}")

    # 3. Prepare for Multi-Task Training (Conceptual)
    print("\n--- Preparing Dummy Data for Training ---")
    qa_training_data = [
        {"question": "What is Alzheimer's disease characterized by?", "answer": "Alzheimer's disease is characterized by the accumulation of amyloid-beta plaques and neurofibrillary tangles."}, # Simplified answer
        {"question": "What is the function of the human circulatory system?", "answer": "The human circulatory system transports blood, nutrients, oxygen, carbon dioxide, and hormones."}
    ]
    # Statements not directly from the knowledge base to encourage generalization and retrieval for reconstruction
    reconstruction_training_data = [
        "Oxidative stress and energy deficits can result from impaired mitochondrial function.",
        "Insulin resistance or insufficient production leads to diabetes mellitus."
    ]

    # Create the multi-task dataset
    multi_task_dataset = MultiTaskMedicalDataset(
        qa_data=qa_training_data,
        reconstruction_data=reconstruction_training_data,
        tokenizer=rag_model.tokenizer,
        knowledge_base=kb
    )

    print(f"Total samples in multi-task dataset: {len(multi_task_dataset)}")
    # Example of accessing a sample (e.g., a QA task sample)
    sample = multi_task_dataset[0]
    print("\nExample QA Training Sample:")
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Labels IDs shape:", sample["labels"].shape)
    print("Decoded Input:", rag_model.tokenizer.decode(sample["input_ids"].masked_fill(sample["input_ids"] == rag_model.tokenizer.pad_token_id, rag_model.tokenizer.unk_token_id), skip_special_tokens=False))
    print("Decoded Labels (target):")
    # Labels have -100 for ignored tokens, so decode carefully or replace -100 first
    decoded_labels_ids = sample["labels"].clone()
    decoded_labels_ids[decoded_labels_ids == -100] = rag_model.tokenizer.pad_token_id
    print(rag_model.tokenizer.decode(decoded_labels_ids, skip_special_tokens=True))

    # Example of accessing a sample (e.g., a Reconstruction task sample)
    sample_recon = multi_task_dataset[len(qa_training_data)] # First reconstruction sample
    print("\nExample Reconstruction Training Sample:")
    print("Decoded Input:", rag_model.tokenizer.decode(sample_recon["input_ids"].masked_fill(sample_recon["input_ids"] == rag_model.tokenizer.pad_token_id, rag_model.tokenizer.unk_token_id), skip_special_tokens=False))
    decoded_labels_ids_recon = sample_recon["labels"].clone()
    decoded_labels_ids_recon[decoded_labels_ids_recon == -100] = rag_model.tokenizer.pad_token_id
    print("Decoded Labels (target):")
    print(rag_model.tokenizer.decode(decoded_labels_ids_recon, skip_special_tokens=True))

    print("\n--- Trainer Setup (Conceptual) ---")
    print("To fine-tune the model, you would typically use `transformers.Trainer` like this:")
    print("training_args = TrainingArguments(...)")
    print("trainer = Trainer(model=rag_model, args=training_args, train_dataset=multi_task_dataset, ...) ")
    print("trainer.train()")

