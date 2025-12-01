import random
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def generate_dummy_medical_data(num_documents=100, num_qa_pairs=50):
    documents = []
    for i in range(num_documents):
        doc_id = f"doc_{i}"
        title = f"Medical Article on {random.choice(['Diabetes', 'Hypertension', 'Cardiovascular Disease', 'Cancer Treatment', 'Neurology', 'Infectious Diseases'])}"
        content = f"This document discusses various aspects of {title.split('on ')[1].lower()}. It includes symptoms, diagnoses, and treatment options. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        documents.append({"id": doc_id, "title": title, "content": content})

    qa_pairs = []
    for i in range(num_qa_pairs):
        query = f"What is the latest treatment for {random.choice(['Diabetes', 'Hypertension', 'Asthma', 'Migraine', 'Arthritis'])}?"
        answer = f"According to recent studies, the latest treatment for {query.split('for ')[1].replace('?', '').lower()} involves a combination of medication and lifestyle changes, often incorporating new therapeutic agents that target specific pathways. More details can be found in various medical journals and clinical trials."
        qa_pairs.append({"query": query, "answer": answer})
    
    return documents, qa_pairs


class DocumentEncoder(torch.nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, documents):
        return self.model.encode(documents, convert_to_tensor=True)


class FaissIndex:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.document_ids = []

    def add_documents(self, embeddings, doc_ids):
        self.index.add(embeddings.cpu().numpy())
        self.document_ids.extend(doc_ids)

    def search(self, query_embedding, k=5):
        D, I = self.index.search(query_embedding.cpu().numpy(), k)
        retrieved_docs_info = []
        for i in range(k):
            if I[0, i] != -1:
                doc_id = self.document_ids[I[0, i]]
                score = D[0, i]
                retrieved_docs_info.append({"doc_id": doc_id, "score": float(score)})
        return retrieved_docs_info


class QueryEncoder(torch.nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_name)

    def forward(self, queries):
        return self.model.encode(queries, convert_to_tensor=True)


class Generator(torch.nn.Module):
    def __init__(self, model_name='distilgpt2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.resize_token_embeddings(len(self.tokenizer))
            
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def generate(self, input_ids, attention_mask=None, max_new_tokens=100, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, **kwargs)


class RAGModel(torch.nn.Module):
    def __init__(self, document_encoder, faiss_index, query_encoder_model_name='sentence-transformers/all-MiniLM-L6-v2', generator_model_name='distilgpt2'):
        super().__init__()
        self.document_encoder = document_encoder
        self.faiss_index = faiss_index
        self.query_encoder = QueryEncoder(query_encoder_model_name)
        self.generator = Generator(generator_model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.query_encoder.to(self.device)
        self.generator.to(self.device)
        
        self.tokenizer = self.generator.tokenizer

    def forward(self, queries, target_answers=None, documents_content=None):
        query_embeddings = self.query_encoder(queries).to(self.device)

        retrieved_contexts = []
        for q_embed in query_embeddings:
            search_results = self.faiss_index.search(q_embed.unsqueeze(0), k=3)
            context = [documents_content.get(res['doc_id'], "") for res in search_results]
            retrieved_contexts.append(" ".join(context))

        generator_inputs = [f"Question: {q}\nContext: {c}\nAnswer:" for q, c in zip(queries, retrieved_contexts)]
        
        if target_answers is not None:
            full_inputs = [inp + ' ' + ans + self.tokenizer.eos_token for inp, ans in zip(generator_inputs, target_answers)]
        else:
            full_inputs = generator_inputs
            
        encoded_inputs = self.tokenizer(full_inputs, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)

        labels = None
        if target_answers is not None:
            labels = input_ids.clone()
            answer_start_token_id = self.tokenizer.encode("Answer:", add_special_tokens=False)[0]
            
            for i in range(labels.shape[0]):
                try:
                    answer_start_index = (input_ids[i] == answer_start_token_id).nonzero(as_tuple=True)[0][0].item()
                    labels[i, :answer_start_index] = -100
                except IndexError:
                    labels[i, :] = -100
        
        if target_answers is not None:
            outputs = self.generator(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs.loss
        else:
            generated_ids = self.generator.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                num_beams=4,
                early_stopping=True
            )
            decoded_outputs = []
            for i, gen_ids in enumerate(generated_ids):
                original_input_len = input_ids[i].shape[0]
                decoded_text = self.tokenizer.decode(gen_ids[original_input_len:], skip_special_tokens=True)
                decoded_outputs.append(decoded_text.strip())
            return decoded_outputs


class QADataset(Dataset):
    def __init__(self, qa_pairs):
        self.qa_pairs = qa_pairs

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        return self.qa_pairs[idx]['query'], self.qa_pairs[idx]['answer']


def train_rag_model(model, train_dataloader, documents_content_map, epochs=3, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    optimizer = AdamW(list(model.query_encoder.parameters()) + list(model.generator.parameters()), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_queries, batch_answers in progress_bar:
            optimizer.zero_grad()
            
            loss = model(queries=batch_queries, target_answers=batch_answers, documents_content=documents_content_map)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    print("Training complete.")
    return model


def inference(model, query: str, documents_content_map) -> str:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        answer = model(queries=[query], documents_content=documents_content_map)[0]
    return answer


def main():
    print("--- Starting Medical Information Q&A System Setup ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n1. Generating dummy medical data...")
    documents, qa_pairs = generate_dummy_medical_data(num_documents=200, num_qa_pairs=100)
    documents_content_map = {doc['id']: doc['content'] for doc in documents}
    print(f"Generated {len(documents)} documents and {len(qa_pairs)} QA pairs.")

    print("\n2. Initializing Document Encoder and building FAISS index (fixed component)...")
    doc_encoder = DocumentEncoder()
    doc_embeddings = doc_encoder([doc['content'] for doc in documents])
    faiss_index = FaissIndex(embedding_dim=doc_embeddings.shape[1])
    faiss_index.add_documents(doc_embeddings, [doc['id'] for doc in documents])
    print(f"FAISS index built with {faiss_index.index.ntotal} documents and embedding dimension {doc_embeddings.shape[1]}.")

    print("\n3. Initializing RAG Model (Query Encoder and Generator will be fine-tuned)...")
    rag_model = RAGModel(document_encoder=doc_encoder, faiss_index=faiss_index)
    print("RAG Model initialized, components moved to device.")

    print("\n4. Preparing DataLoader for training...")
    qa_dataset = QADataset(qa_pairs)
    train_dataloader = DataLoader(qa_dataset, batch_size=4, shuffle=True)
    print(f"Training DataLoader created with {len(train_dataloader)} batches.")

    print("\n5. Starting End-to-End Joint Training of RAG model...")
    trained_rag_model = train_rag_model(rag_model, train_dataloader, documents_content_map, epochs=1, learning_rate=5e-5)
    print("Training process completed.")

    print("\n6. Demonstrating Inference with the trained model...")
    test_queries = [
        "What are the typical symptoms and diagnostic methods for diabetes?",
        "Summarize the latest research on cardiovascular disease treatments.",
        "What are some common side effects of chemotherapy?"
    ]

    for i, query in enumerate(test_queries):
        print(f"\n--- Inference Query {i+1} ---")
        print(f"Query: {query}")
        answer = inference(trained_rag_model, query, documents_content_map)
        print(f"Answer: {answer}")
    
    print("\n--- Medical Information Q&A System Demo Complete ---")

if __name__ == "__main__":
    main()