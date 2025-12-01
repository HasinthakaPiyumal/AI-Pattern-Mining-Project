import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import re

class DataLoader:
    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        return df

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

class ExemplarSelector:
    def select_exemplars(self, embeddings, unlabeled_data_texts, num_exemplars_to_propose):
        if len(embeddings) < num_exemplars_to_propose:
            return list(unlabeled_data_texts)

        kmeans = KMeans(n_clusters=num_exemplars_to_propose, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        
        proposed_exemplars_texts = []
        for i in range(num_exemplars_to_propose):
            cluster_points_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_points_indices) > 0:
                cluster_embeddings = embeddings[cluster_points_indices]
                centroid = kmeans.cluster_centers_[i]
                
                distances = cosine_similarity([centroid], cluster_embeddings)[0]
                closest_point_index_in_cluster = np.argmax(distances) 
                original_index = cluster_points_indices[closest_point_index_in_cluster]
                proposed_exemplars_texts.append(unlabeled_data_texts[original_index])
        
        return proposed_exemplars_texts

class HumanAnnotatorMock:
    def annotate(self, exemplars_texts, possible_labels=None):
        if possible_labels is None:
            possible_labels = ["Billing", "Technical Support", "General Inquiry", "Product Feature Request"]
        
        labeled_exemplars = []
        for text in exemplars_texts:
            label = random.choice(possible_labels)
            labeled_exemplars.append({"text": text, "label": label})
        return labeled_exemplars

class FewShotPromptBuilder:
    def build_prompt(self, labeled_exemplars, new_query):
        prompt_parts = []
        for exemplar in labeled_exemplars:
            prompt_parts.append(f"Example Input: {exemplar['text']}\nExample Output: {exemplar['label']}")
        
        prompt_parts.append(f"New Input: {new_query}\nOutput:")
        return "\n\n".join(prompt_parts)

if __name__ == "__main__":
    # 1. Simulate Data Ingestion and Preprocessing
    data_loader = DataLoader()
    
    # Create a dummy CSV file for demonstration
    dummy_data = {
        "text": [
            "My internet is not working. I need help.",
            "How do I change my billing address?",
            "I want to know about your new product features.",
            "My account is locked, what should I do?",
            "Can I upgrade my subscription plan?",
            "The app keeps crashing on my phone.",
            "What are your operating hours?",
            "I forgot my password, please help.",
            "Tell me about the premium plan benefits.",
            "My payment failed, why?",
            "I need assistance with setting up my device.",
            "Where can I find the user manual?",
            "How to cancel my service?",
            "Is there a discount for new customers?",
            "My device is not connecting to Wi-Fi."
        ]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_filepath = "customer_chats.csv"
    dummy_df.to_csv(dummy_filepath, index=False)
    
    print(f"Loaded {len(dummy_df)} raw chat transcripts.")
    
    unlabeled_df = data_loader.load_data(dummy_filepath)
    unlabeled_df["processed_text"] = unlabeled_df["text"].apply(data_loader.preprocess_text)
    unlabeled_texts = unlabeled_df["processed_text"].tolist()
    
    existing_labeled_data = [
        {"text": "My bill is too high this month.", "label": "Billing"},
        {"text": "I cannot log in to my account.", "label": "Technical Support"},
        {"text": "What is your refund policy?", "label": "General Inquiry"}
    ]
    print(f"Loaded {len(existing_labeled_data)} existing labeled exemplars.")

    # 2. Embedding Generation
    embedding_generator = EmbeddingGenerator()
    unlabeled_embeddings = embedding_generator.generate_embeddings(unlabeled_texts)
    print(f"Generated embeddings for {len(unlabeled_embeddings)} unlabeled texts.")

    # 3. Exemplar Selection
    exemplar_selector = ExemplarSelector()
    num_exemplars_to_propose = 5
    proposed_exemplars_texts = exemplar_selector.select_exemplars(unlabeled_embeddings, unlabeled_texts, num_exemplars_to_propose)
    print(f"Proposed {len(proposed_exemplars_texts)} exemplars for human annotation:")
    for i, text in enumerate(proposed_exemplars_texts):
        print(f"  {i+1}. {text}")

    # 4. Simulate Human Annotation
    human_annotator = HumanAnnotatorMock()
    newly_labeled_exemplars = human_annotator.annotate(proposed_exemplars_texts)
    print(f"Newly labeled {len(newly_labeled_exemplars)} exemplars:")
    for exemplar in newly_labeled_exemplars:
        print(f"  Text: {exemplar['text']} | Label: {exemplar['label']}")

    # 5. Combine Labeled Data
    all_labeled_exemplars = existing_labeled_data + newly_labeled_exemplars
    print(f"Total labeled exemplars after annotation: {len(all_labeled_exemplars)}")

    # 6. Few-Shot Prompt Building
    prompt_builder = FewShotPromptBuilder()
    new_customer_query = "My account is showing incorrect balance."
    few_shot_prompt = prompt_builder.build_prompt(all_labeled_exemplars, new_customer_query)
    
    print("\n--- Generated Few-Shot Prompt ---")
    print(few_shot_prompt)
    print("-----------------------------------")

    # (Optional Conceptual Part) - How an LLM would use this prompt
    # In a real application, you would send this 'few_shot_prompt' to an LLM API (e.g., OpenAI, Gemini, etc.)
    # The LLM would then generate the 'Output:' based on the examples provided.
    print(f"\n(Conceptual): An LLM would classify the query '{new_customer_query}' based on the provided exemplars.")
    print(f"(Conceptual): Expected LLM output for '{new_customer_query}' might be 'Billing' or 'General Inquiry' based on similar exemplars.\n")

    # Clean up dummy file
    import os
    os.remove(dummy_filepath)
