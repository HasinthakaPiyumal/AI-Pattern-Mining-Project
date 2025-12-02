import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
import random

class MultilingualTicketClassifier:
    def __init__(self, sbert_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 llm_model_name='google/gemma-2b-it'):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sbert_model = SentenceTransformer(sbert_model_name, device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16).to(self.device)
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        self.historical_tickets = []
        self.labels = set()

    def _get_embedding(self, text: str) -> np.ndarray:
        embedding = self.sbert_model.encode(text, convert_to_tensor=False)
        return embedding

    def ingest_ticket(self, text: str, label: str):
        embedding = self._get_embedding(text)
        self.historical_tickets.append({
            'text': text,
            'label': label,
            'embedding': embedding
        })
        self.labels.add(label)

    def _find_semantically_similar_examples(self, query_embedding: np.ndarray, num_examples: int) -> list:
        if not self.historical_tickets:
            return []
        
        embeddings = np.array([ticket['embedding'] for ticket in self.historical_tickets])
        
        similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        top_indices = np.argsort(similarities)[::-1][:num_examples]
        
        return [self.historical_tickets[i] for i in top_indices]

    def _find_label_aligned_examples(self, num_examples: int) -> list:
        if not self.historical_tickets or not self.labels:
            return []

        selected_examples = []
        labels_in_pool = list(self.labels)
        random.shuffle(labels_in_pool)

        label_examples_map = {label: [] for label in self.labels}
        for ticket in self.historical_tickets:
            label_examples_map[ticket['label']].append(ticket)
        
        for label in labels_in_pool:
            if label_examples_map[label] and len(selected_examples) < num_examples:
                selected_examples.append(random.choice(label_examples_map[label]))
        
        if len(selected_examples) < num_examples:
            remaining_needed = num_examples - len(selected_examples)
            available_tickets = [t for t in self.historical_tickets if t not in selected_examples]
            if len(available_tickets) > remaining_needed:
                selected_examples.extend(random.sample(available_tickets, remaining_needed))
            else:
                selected_examples.extend(available_tickets)

        return selected_examples[:num_examples]

    def _construct_prompt(self, input_ticket_text: str, examples: list) -> str:
        prompt_parts = [
            "You are an expert customer support ticket classifier. Classify the following customer support tickets into one of the provided categories. Provide only the category name.\n\nExamples:"
        ]
        
        for ex in examples:
            prompt_parts.append(f"Ticket: \"{ex['text']}\"\nLabel: {ex['label']}\n")
        
        prompt_parts.append(f"New Ticket: \"{input_ticket_text}\"\nLabel:")
        
        return "\n".join(prompt_parts)

    def classify_ticket(self, ticket_text: str, strategy: str = 'semantic', num_examples: int = 3) -> tuple[str, float]:
        if not self.historical_tickets:
            print("Warning: No historical tickets ingested. Classification might be inaccurate.")
            prompt = f"You are an expert customer support ticket classifier. Classify the following customer support ticket into a relevant category. Provide only the category name.\n\nTicket: \"{ticket_text}\"\nLabel:"
            llm_output = self.llm_pipeline(prompt, max_new_tokens=10, num_return_sequences=1, do_sample=False)[0]['generated_text']
            predicted_label = llm_output.split("Label:")[1].strip().split('\n')[0].strip() if "Label:" in llm_output else "UNKNOWN"
            return predicted_label, 0.0

        examples = []
        if strategy == 'semantic':
            query_embedding = self._get_embedding(ticket_text)
            examples = self._find_semantically_similar_examples(query_embedding, num_examples)
        elif strategy == 'label':
            examples = self._find_label_aligned_examples(num_examples)
        elif strategy == 'combined':
            num_semantic = num_examples // 2
            num_label = num_examples - num_semantic

            query_embedding = self._get_embedding(ticket_text)
            semantic_examples = self._find_semantically_similar_examples(query_embedding, num_semantic)
            label_examples = self._find_label_aligned_examples(num_label)
            
            combined_texts = {ex['text'] for ex in semantic_examples}
            examples.extend(semantic_examples)
            for ex in label_examples:
                if ex['text'] not in combined_texts:
                    examples.append(ex)
                    combined_texts.add(ex['text'])
            examples = examples[:num_examples]
        else:
            raise ValueError("Invalid strategy. Choose 'semantic', 'label', or 'combined'.")

        prompt = self._construct_prompt(ticket_text, examples)
        
        llm_output = self.llm_pipeline(prompt, max_new_tokens=10, num_return_sequences=1, do_sample=False)[0]['generated_text']
        
        predicted_label = llm_output.split("Label:")[-1].strip().split('\n')[0].strip()
        
        predicted_label = predicted_label.replace('"', '').replace("'", '').strip()
        
        confidence = 1.0 if predicted_label in self.labels else 0.5
        
        return predicted_label, confidence
