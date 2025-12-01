from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class VoteKExemplarSelector:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.labeled_texts = []
        self.labeled_labels = []
        self.labeled_embeddings = None

    def add_labeled_examples(self, texts, labels):
        new_embeddings = self.embedding_model.encode(texts, convert_to_tensor=True).cpu().numpy()

        self.labeled_texts.extend(texts)
        self.labeled_labels.extend(labels)

        if self.labeled_embeddings is None:
            self.labeled_embeddings = new_embeddings
        else:
            self.labeled_embeddings = np.vstack((self.labeled_embeddings, new_embeddings))

    def propose_unlabeled_exemplars(self, unlabeled_texts, num_proposals):
        if not unlabeled_texts or num_proposals <= 0:
            return []

        if self.labeled_embeddings is None or len(self.labeled_texts) == 0:
            # If no labeled examples, pick random unlabeled texts
            if len(unlabeled_texts) <= num_proposals:
                return unlabeled_texts
            else:
                selected_indices = np.random.choice(len(unlabeled_texts), num_proposals, replace=False)
                return [unlabeled_texts[i] for i in selected_indices]

        unlabeled_embeddings = self.embedding_model.encode(unlabeled_texts, convert_to_tensor=True).cpu().numpy()

        # Candidate indices that haven't been proposed yet
        candidate_indices = list(range(len(unlabeled_texts)))
        proposed_exemplars_texts = []
        current_exemplar_embeddings = self.labeled_embeddings.copy()

        for _ in range(num_proposals):
            if not candidate_indices:
                break

            min_max_similarity = float('inf')
            best_candidate_idx_in_unlabeled = -1
            best_candidate_embedding = None

            for i in candidate_indices:
                candidate_embedding = unlabeled_embeddings[i:i+1] # Ensure 2D array for similarity calculation
                
                # Calculate similarity of current candidate to all current exemplars (labeled + already proposed)
                similarities = cosine_similarity(candidate_embedding, current_exemplar_embeddings)
                max_similarity = np.max(similarities)

                if max_similarity < min_max_similarity:
                    min_max_similarity = max_similarity
                    best_candidate_idx_in_unlabeled = i
                    best_candidate_embedding = candidate_embedding
            
            if best_candidate_idx_in_unlabeled != -1:
                proposed_exemplars_texts.append(unlabeled_texts[best_candidate_idx_in_unlabeled])
                current_exemplar_embeddings = np.vstack((current_exemplar_embeddings, best_candidate_embedding))
                candidate_indices.remove(best_candidate_idx_in_unlabeled)
            else: # Should not happen if candidate_indices is not empty
                break

        return proposed_exemplars_texts

    def get_current_exemplars(self):
        return [{
            "text": text, 
            "label": label
        } for text, label in zip(self.labeled_texts, self.labeled_labels)]

class FewShotPromptingModel:
    def generate_prompt(self, query_text, exemplars):
        prompt_parts = []
        for exemplar in exemplars:
            prompt_parts.append(f"Example: {exemplar['text']} -> {exemplar['label']}")
        
        prompt_parts.append(f"Query: {query_text} ->")
        return "\n".join(prompt_parts)

# Example Usage:
if __name__ == "__main__":
    # 1. Initialize the selector
    selector = VoteKExemplarSelector()

    # 2. Add some initial labeled examples
    initial_labeled_texts = [
        "The printer is not working, showing an error code E305.",
        "My internet connection keeps dropping every few minutes.",
        "I cannot log into my account, password reset link is not arriving."
    ]
    initial_labeled_labels = [
        "Hardware Issue",
        "Network Issue",
        "Account Access Issue"
    ]
    selector.add_labeled_examples(initial_labeled_texts, initial_labeled_labels)
    print("\n--- Initial Labeled Exemplars ---")
    for ex in selector.get_current_exemplars():
        print(f"Text: {ex['text']}, Label: {ex['label']}")

    # 3. Propose diverse unlabeled exemplars for human annotation
    unlabeled_support_tickets = [
        "My screen is flickering randomly, especially when I open videos.",
        "I need to change my billing address, where can I do that?",
        "The software update failed, and now the application crashes on startup.",
        "My mouse is unresponsive, tried restarting the computer.",
        "I got charged twice for my subscription this month.",
        "How do I connect to the company VPN from home?",
        "The new feature 'Dark Mode' is not appearing after the last update.",
        "I can't hear anything from my headphones, but the speakers work."
    ]
    num_proposals = 3
    proposed_tickets = selector.propose_unlabeled_exemplars(unlabeled_support_tickets, num_proposals)
    print(f"\n--- Proposed {num_proposals} Unlabeled Exemplars for Annotation ---")
    for i, text in enumerate(proposed_tickets):
        print(f"Proposal {i+1}: {text}")

    # Simulate human annotation for the proposed tickets
    # For demonstration, let's manually assign labels
    simulated_labels = [
        "Display Issue", 
        "Billing Issue", 
        "Software Bug"
    ] # Assuming these correspond to proposed_tickets

    # 4. Add the newly labeled examples to the selector
    selector.add_labeled_examples(proposed_tickets, simulated_labels)
    print("\n--- All Labeled Exemplars (including newly added) ---")
    for ex in selector.get_current_exemplars():
        print(f"Text: {ex['text']}, Label: {ex['label']}")

    # 5. Use the exemplars for Few-Shot Prompting
    few_shot_model = FewShotPromptingModel()
    current_exemplars = selector.get_current_exemplars()
    
    query = "I'm trying to print a document, but nothing is happening. The printer light is flashing."
    prompt = few_shot_model.generate_prompt(query, current_exemplars)
    print("\n--- Generated Few-Shot Prompt ---")
    print(prompt)
    
    query_2 = "My email client is showing a 'connection refused' error."
    prompt_2 = few_shot_model.generate_prompt(query_2, current_exemplars)
    print("\n--- Generated Few-Shot Prompt for second query ---")
    print(prompt_2)