import random
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class RerankerModel(nn.Module):
    """A simple neural network for predicting relevance scores."""
    def __init__(self, embedding_dim):
        super(RerankerModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, 128)  # Query + Document embedding
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output a single relevance score
        self.sigmoid = nn.Sigmoid() # To get a score between 0 and 1

    def forward(self, query_embedding, doc_embedding):
        combined_embedding = torch.cat((query_embedding, doc_embedding), dim=1)
        x = self.fc1(combined_embedding)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.sigmoid(x)

class PredictiveReranker:
    def __init__(self, sbert_model_name='all-MiniLM-L6-v2', embedding_dim=384):
        try:
            self.sbert_model = SentenceTransformer(sbert_model_name)
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer model {sbert_model_name}. Using a mock reranker. Error: {e}")
            self.sbert_model = None

        self.reranker_model = RerankerModel(embedding_dim)
        self.optimizer = optim.Adam(self.reranker_model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss() # Binary Cross-Entropy for relevance scores
        self.is_trained = False
        self.embedding_dim = embedding_dim

    def _create_synthetic_training_data(self, num_samples=1000):
        """Generates synthetic training data for the reranker."""
        print(f"[PredictiveReranker] Generating {num_samples} synthetic training samples...")
        queries = [
            "diabetes management", "influenza symptoms", "cardiovascular prevention",
            "alzheimer research", "antibiotic side effects", "healthy diet tips",
            "exercise benefits", "vaccination schedule", "cancer treatment options",
            "mental health support"
        ]
        documents = [
            "Clinical guidelines for managing type 2 diabetes include dietary changes, regular exercise, and medication such as metformin.",
            "Symptoms of influenza often include fever, body aches, headache, and fatigue. Vaccination is recommended annually.",
            "Cardiovascular disease prevention focuses on controlling blood pressure, cholesterol levels, and maintaining a healthy lifestyle.",
            "The latest research on Alzheimer's disease suggests a complex interplay of genetic and environmental factors.",
            "Common side effects of antibiotics can include nausea, diarrhea, and allergic reactions. Always complete the full course of treatment.",
            "Eating a balanced diet rich in fruits, vegetables, and whole grains is crucial for overall health.",
            "Regular physical activity, including aerobic and strength training, offers numerous health benefits.",
            "Childhood immunization schedules are vital for preventing infectious diseases.",
            "Advances in oncology have led to diverse treatment options for various types of cancer.",
            "Seeking professional help and building a strong support network are important for good mental health."
        ]

        training_data = []
        for _ in range(num_samples):
            query = random.choice(queries)
            doc = random.choice(documents)
            # Assign a relevance label: 1 if keywords overlap significantly, 0 otherwise
            # This is a simplification for synthetic data generation.
            query_keywords = set(query.lower().split())
            doc_keywords = set(doc.lower().split())
            overlap = len(query_keywords.intersection(doc_keywords))
            relevance = 1.0 if overlap >= 2 else 0.0 # Heuristic for relevance
            training_data.append((query, doc, relevance))
        return training_data

    def train_reranker(self, epochs=10, num_synthetic_samples=1000):
        if not self.sbert_model:
            print("[PredictiveReranker] Cannot train without SentenceTransformer model.")
            return

        training_data = self._create_synthetic_training_data(num_synthetic_samples)
        print("[PredictiveReranker] Training reranker model...")

        query_texts = [item[0] for item in training_data]
        doc_texts = [item[1] for item in training_data]
        relevance_labels = torch.tensor([item[2] for item in training_data], dtype=torch.float32).unsqueeze(1)

        # Encode all queries and documents once
        query_embeddings = torch.tensor(self.sbert_model.encode(query_texts), dtype=torch.float32)
        doc_embeddings = torch.tensor(self.sbert_model.encode(doc_texts), dtype=torch.float32)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.reranker_model(query_embeddings, doc_embeddings)
            loss = self.criterion(outputs, relevance_labels)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 1 == 0: # Print loss for every epoch for demonstration
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        self.is_trained = True
        print("[PredictiveReranker] Reranker model trained successfully.")

    def rerank_documents(self, query: str, documents: list[str]) -> list[str]:
        """Reranks documents using the trained predictive model."""
        if not self.sbert_model:
            print("[PredictiveReranker] Skipping reranking (mock mode) as SentenceTransformer is not loaded.")
            # Fallback to a simple heuristic if model not loaded
            if not documents: return []
            reranked_docs = []
            query_lower = query.lower()
            for doc in documents:
                if query_lower in doc.lower():
                    reranked_docs.insert(0, doc)
                else:
                    reranked_docs.append(doc)
            return reranked_docs

        if not self.is_trained:
            print("[PredictiveReranker] Reranker model not trained. Training with default settings...")
            self.train_reranker()

        if not documents:
            return []

        print(f"[PredictiveReranker] Reranking {len(documents)} documents for query: \'{query}\' using trained model.")

        query_embedding = torch.tensor(self.sbert_model.encode([query]), dtype=torch.float32)
        doc_embeddings = torch.tensor(self.sbert_model.encode(documents), dtype=torch.float32)

        scores = []
        for i, doc_embed in enumerate(doc_embeddings):
            # Expand query_embedding to match batch size of 1 for doc_embed
            score = self.reranker_model(query_embedding, doc_embed.unsqueeze(0)).item()
            scores.append({'document': documents[i], 'score': score})

        reranked_docs = [item['document'] for item in sorted(scores, key=lambda x: x['score'], reverse=True)]
        return reranked_docs


if __name__ == "__main__":
    reranker = PredictiveReranker()
    # Train the reranker explicitly for demonstration
    # reranker.train_reranker(epochs=5, num_synthetic_samples=500)

    sample_query = "diabetes management guidelines"
    sample_docs = [
        "Symptoms of influenza often include fever, body aches, headache, and fatigue. Vaccination is recommended annually.",
        "Clinical guidelines for managing type 2 diabetes include dietary changes, regular exercise, and medication such as metformin.",
        "Cardiovascular disease prevention focuses on controlling blood pressure, cholesterol levels, and maintaining a healthy lifestyle.",
        "The latest research on Alzheimer's disease suggests a complex interplay of genetic and environmental factors."
    ]

    print(f"\nOriginal Documents: {sample_docs}")

    # The reranker will automatically train if not already trained when rerank_documents is called
    reranked = reranker.rerank_documents(sample_query, sample_docs)
    print("\nPredictive Reranked Documents:")
    for i, doc in enumerate(reranked):
        print(f"  {i+1}. {doc[:70]}...")

    sample_query_2 = "Alzheimer's research update"
    reranked_2 = reranker.rerank_documents(sample_query_2, sample_docs)
    print("\nPredictive Reranked Documents (Alzheimer's):")
    for i, doc in enumerate(reranked_2):
        print(f"  {i+1}. {doc[:70]}...")
