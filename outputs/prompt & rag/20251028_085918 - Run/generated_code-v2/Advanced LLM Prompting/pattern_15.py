import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class MultilingualTicketClassifier:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', in_context_examples=None):
        self.model = SentenceTransformer(model_name)
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.example_embeddings_by_language = {}
        if self.in_context_examples:
            self._preprocess_examples()

    def _preprocess_examples(self):
        for lang in set(ex['language'] for ex in self.in_context_examples):
            lang_examples = [ex for ex in self.in_context_examples if ex['language'] == lang]
            texts = [ex['text'] for ex in lang_examples]
            embeddings = self._get_embeddings(texts)
            self.example_embeddings_by_language[lang] = {'examples': lang_examples, 'embeddings': embeddings}

    def _get_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def _select_examples_semantic(self, input_text_embedding, language, num_examples=3):
        if language not in self.example_embeddings_by_language:
            return []

        lang_data = self.example_embeddings_by_language[language]
        examples = lang_data['examples']
        embeddings = lang_data['embeddings']

        similarities = cosine_similarity(input_text_embedding.reshape(1, -1), embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        selected_examples = [examples[i] for i in sorted_indices[:num_examples]]
        return selected_examples

    def _select_examples_task_based(self, input_ticket_attributes, language, num_examples=3, attribute_key='label'):
        if language not in self.example_embeddings_by_language:
            return []

        lang_data = self.example_embeddings_by_language[language]
        examples = lang_data['examples']

        relevant_examples = [ex for ex in examples if ex.get(attribute_key) == input_ticket_attributes.get(attribute_key)]

        # Take first num_examples from the relevant ones
        selected_examples = relevant_examples[:num_examples]
        return selected_examples

    def _select_examples_hybrid(self, input_text_embedding, input_ticket_attributes, language, num_examples=3, semantic_weight=0.6, attribute_key='label'):
        if language not in self.example_embeddings_by_language:
            return []

        lang_data = self.example_embeddings_by_language[language]
        examples = lang_data['examples']
        embeddings = lang_data['embeddings']

        semantic_similarities = cosine_similarity(input_text_embedding.reshape(1, -1), embeddings)[0]

        # Create a combined score for ranking
        scores = []
        for i, example in enumerate(examples):
            task_score = 1.0 if example.get(attribute_key) == input_ticket_attributes.get(attribute_key) else 0.0
            combined_score = (semantic_weight * semantic_similarities[i]) + ((1 - semantic_weight) * task_score)
            scores.append((combined_score, i))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        sorted_indices = [idx for score, idx in scores]
        
        selected_examples = [examples[i] for i in sorted_indices[:num_examples]]
        return selected_examples

    def _format_prompt(self, input_ticket_text, selected_examples):
        prompt_parts = []
        prompt_parts.append("Classify the following customer support tickets. Provide only the label.")
        prompt_parts.append("\n\nExamples:")
        for ex in selected_examples:
            prompt_parts.append(f"Ticket: {ex['text']}\nLabel: {ex['label']}")
        prompt_parts.append(f"\n\nTicket: {input_ticket_text}\nLabel:")
        return "\n".join(prompt_parts)

    def _call_llm(self, prompt):
        # This is a mock LLM call for demonstration purposes.
        # In a real application, this would interact with an actual LLM API.
        # For simplicity, we'll just return a mock prediction based on keywords.
        if "payment issue" in prompt.lower():
            return "Billing"
        elif "delivery" in prompt.lower() or "shipping" in prompt.lower():
            return "Shipping"
        elif "product defect" in prompt.lower() or "broken" in prompt.lower():
            return "Technical Support"
        else:
            return "General Inquiry"

    def classify_ticket(self, ticket_text, language, strategy='hybrid', **kwargs):
        input_text_embedding = self._get_embeddings([ticket_text])[0]
        input_ticket_attributes = kwargs.get('input_ticket_attributes', {'language': language})

        selected_examples = []
        if strategy == 'semantic':
            selected_examples = self._select_examples_semantic(input_text_embedding, language, kwargs.get('num_examples', 3))
        elif strategy == 'task_based':
            selected_examples = self._select_examples_task_based(input_ticket_attributes, language, kwargs.get('num_examples', 3), kwargs.get('attribute_key', 'label'))
        elif strategy == 'hybrid':
            selected_examples = self._select_examples_hybrid(input_text_embedding, input_ticket_attributes, language, kwargs.get('num_examples', 3), kwargs.get('semantic_weight', 0.6), kwargs.get('attribute_key', 'label'))
        else:
            raise ValueError("Invalid strategy. Choose 'semantic', 'task_based', or 'hybrid'.")

        prompt = self._format_prompt(ticket_text, selected_examples)
        predicted_label = self._call_llm(prompt)
        return predicted_label


# Example Usage:
if __name__ == '__main__':
    # Sample In-Context Examples
    in_context_data = [
        {'text': 'My internet is not working. I have no connection.', 'label': 'Technical Support', 'language': 'en', 'urgency': 'high'},
        {'text': 'I cannot log in to my account.', 'label': 'Account Issues', 'language': 'en', 'urgency': 'medium'},
        {'text': 'My bill is incorrect. I was overcharged.', 'label': 'Billing', 'language': 'en', 'urgency': 'high'},
        {'text': 'Where is my order? It was supposed to arrive yesterday.', 'label': 'Shipping', 'language': 'en', 'urgency': 'medium'},
        {'text': 'Мой интернет не работает. Нет соединения.', 'label': 'Technical Support', 'language': 'ru', 'urgency': 'high'},
        {'text': 'Не могу войти в свой аккаунт.', 'label': 'Account Issues', 'language': 'ru', 'urgency': 'medium'},
        {'text': 'Мой счет неверен. Меня переплатили.', 'label': 'Billing', 'language': 'ru', 'urgency': 'high'},
        {'text': 'Где мой заказ? Он должен был прийти вчера.', 'label': 'Shipping', 'language': 'ru', 'urgency': 'medium'},
        {'text': 'Mi internet no funciona. No tengo conexión.', 'label': 'Technical Support', 'language': 'es', 'urgency': 'high'},
        {'text': 'No puedo iniciar sesión en mi cuenta.', 'label': 'Account Issues', 'language': 'es', 'urgency': 'medium'},
        {'text': 'Mi factura es incorrecta. Me cobraron de más.', 'label': 'Billing', 'language': 'es', 'urgency': 'high'},
        {'text': '¿Dónde está mi pedido? Se suponía que llegaría ayer.', 'label': 'Shipping', 'language': 'es', 'urgency': 'medium'},
    ]

    # Initialize the classifier with examples
    classifier = MultilingualTicketClassifier(in_context_examples=in_context_data)

    print("--- English Tickets ---")
    # English Ticket - Hybrid Strategy
    ticket_en_1 = "I have a problem with my payment, it was declined."
    input_attrs_en_1 = {'label': 'Billing', 'urgency': 'high'}
    predicted_label_en_1 = classifier.classify_ticket(ticket_en_1, 'en', strategy='hybrid', input_ticket_attributes=input_attrs_en_1)
    print(f"Ticket: '{ticket_en_1}'\nPredicted Label (Hybrid): {predicted_label_en_1}\n")

    # English Ticket - Semantic Strategy
    ticket_en_2 = "My connection is completely down."
    predicted_label_en_2 = classifier.classify_ticket(ticket_en_2, 'en', strategy='semantic')
    print(f"Ticket: '{ticket_en_2}'\nPredicted Label (Semantic): {predicted_label_en_2}\n")

    print("--- Russian Tickets ---")
    # Russian Ticket - Task-Based Strategy
    ticket_ru_1 = "Мой аккаунт заблокирован."
    input_attrs_ru_1 = {'label': 'Account Issues'}
    predicted_label_ru_1 = classifier.classify_ticket(ticket_ru_1, 'ru', strategy='task_based', input_ticket_attributes=input_attrs_ru_1)
    print(f"Ticket: '{ticket_ru_1}'\nPredicted Label (Task-Based): {predicted_label_ru_1}\n")

    # Russian Ticket - Hybrid Strategy
    ticket_ru_2 = "Посылка не пришла вовремя."
    input_attrs_ru_2 = {'label': 'Shipping', 'urgency': 'medium'}
    predicted_label_ru_2 = classifier.classify_ticket(ticket_ru_2, 'ru', strategy='hybrid', input_ticket_attributes=input_attrs_ru_2)
    print(f"Ticket: '{ticket_ru_2}'\nPredicted Label (Hybrid): {predicted_label_ru_2}\n")

    print("--- Spanish Tickets ---")
    # Spanish Ticket - Hybrid Strategy
    ticket_es_1 = "Tengo un problema con la entrega de mi producto."
    input_attrs_es_1 = {'label': 'Shipping', 'urgency': 'medium'}
    predicted_label_es_1 = classifier.classify_ticket(ticket_es_1, 'es', strategy='hybrid', input_ticket_attributes=input_attrs_es_1)
    print(f"Ticket: '{ticket_es_1}'\nPredicted Label (Hybrid): {predicted_label_es_1}\n")

    # Spanish Ticket - Semantic Strategy
    ticket_es_2 = "Mi internet no funciona para nada."
    predicted_label_es_2 = classifier.classify_ticket(ticket_es_2, 'es', strategy='semantic')
    print(f"Ticket: '{ticket_es_2}'\nPredicted Label (Semantic): {predicted_label_es_2}\n")