import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import random

class USP_Chatbot:
    def __init__(self, 
                 unlabeled_data_path="unlabeled_data.txt", 
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 sentiment_model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        
        self.unlabeled_data_path = unlabeled_data_path
        self.exemplar_store = [] # Stores {'text': '...', 'embedding': np.array([...])}
        self.conversation_history = [] # Stores {'query': '...', 'response': '...'}

        # Load models
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model_name)
        
        # Simulate LLM for exemplar generation (can be replaced with actual API calls)
        self.llm_tokenizer = AutoTokenizer.from_pretrained("gpt2") 
        self.llm_model = None # Placeholder, would load an actual LLM or use API

        # Weights for scoring function
        self.scoring_weights = {
            "relevance": 0.4,
            "contextual_fit": 0.3,
            "information_density": 0.1,
            "sentiment_analysis": 0.1,
            "historical_effectiveness": 0.1
        }

        self._ingest_and_process_unlabeled_data()
        self._generate_initial_exemplars()

    def _ingest_and_process_unlabeled_data(self):
        """Simulates ingestion and preprocessing of unlabeled data."""
        print("Ingesting and processing unlabeled data...")
        try:
            with open(self.unlabeled_data_path, "r", encoding="utf-8") as f:
                raw_data = f.readlines()
            self.processed_unlabeled_data = [line.strip() for line in raw_data if line.strip()]
            print(f"Ingested {len(self.processed_unlabeled_data)} lines of unlabeled data.")
        except FileNotFoundError:
            print(f"Warning: {self.unlabeled_data_path} not found. Starting with empty unlabeled data.")
            self.processed_unlabeled_data = []

    def _generate_initial_exemplars(self, num_exemplars=50):
        """Generates initial exemplars from processed unlabeled data. (Simplified)"""
        print("Generating initial exemplars...")
        if not self.processed_unlabeled_data:
            print("No unlabeled data to generate exemplars from.")
            return

        # In a real scenario, an LLM would generate diverse exemplars.
        # Here, we'll just use snippets of the unlabeled data as exemplars.
        selected_snippets = random.sample(self.processed_unlabeled_data, min(num_exemplars, len(self.processed_unlabeled_data)))
        
        for snippet in selected_snippets:
            embedding = self.embedding_model.encode(snippet)
            self.exemplar_store.append({"text": snippet, "embedding": embedding})
        print(f"Generated {len(self.exemplar_store)} initial exemplars.")

    def _update_exemplar_store(self, new_unlabeled_data):
        """Simulates updating the exemplar store with new data. (Simplified)"""
        print("Updating exemplar store with new data...")
        new_snippets = [data.strip() for data in new_unlabeled_data if data.strip()]
        for snippet in new_snippets:
            embedding = self.embedding_model.encode(snippet)
            self.exemplar_store.append({"text": snippet, "embedding": embedding})
        print(f"Added {len(new_snippets)} new exemplars.")

    def _get_conversation_context(self):
        """Retrieves relevant historical conversation context. (Simplified)"""
        if not self.conversation_history:
            return ""
        # Combine last few turns for context
        context_turns = self.conversation_history[-3:] # Last 3 turns
        context_str = " ".join([f"User: {t['query']} Bot: {t['response']}" for t in context_turns])
        return context_str

    def _calculate_relevance_score(self, query_embedding, exemplar_embedding):
        """Calculates cosine similarity as relevance score."""
        return np.dot(query_embedding, exemplar_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(exemplar_embedding))

    def _calculate_contextual_fit_score(self, exemplar_text, conversation_context):
        """Assesses how well the exemplar aligns with conversation context. (Simplified)"""
        if not conversation_context:
            return 0.5 # Neutral score if no context
        
        # Simple keyword overlap or embedding similarity
        exemplar_keywords = set(exemplar_text.lower().split())
        context_keywords = set(conversation_context.lower().split())
        overlap = len(exemplar_keywords.intersection(context_keywords))
        return overlap / len(exemplar_keywords.union(context_keywords)) if len(exemplar_keywords.union(context_keywords)) > 0 else 0.0

    def _calculate_information_density_score(self, exemplar_text):
        """Evaluates conciseness and completeness. (Simplified)"""
        word_count = len(exemplar_text.split())
        # Assign higher scores for moderately dense responses
        if 10 <= word_count <= 50: 
            return 0.9
        elif 5 <= word_count < 10 or 50 < word_count <= 100:
            return 0.7
        else:
            return 0.3

    def _calculate_sentiment_analysis_score(self, exemplar_text):
        """Ensures appropriate tone using a sentiment model."""
        result = self.sentiment_analyzer(exemplar_text)[0]
        if result['label'] == 'POSITIVE':
            return result['score']
        elif result['label'] == 'NEUTRAL': # Assuming 'NEUTRAL' is not a direct label, handling example
            return 0.5 # Neutral score if not positive/negative
        else: # NEGATIVE
            return 1 - result['score'] # Invert score for negative sentiment, higher means less negative

    def _calculate_historical_effectiveness_score(self, exemplar_text):
        """Placeholder for learning from past interactions. (Simplified)"""
        # In a real system, this would involve tracking which exemplars led to good outcomes.
        # For now, a random score or a default.
        return random.uniform(0.4, 0.8)

    def _score_exemplar(self, query_embedding, exemplar, conversation_context):
        """Applies the complicated scoring function to a single exemplar."""
        relevance = self._calculate_relevance_score(query_embedding, exemplar["embedding"])
        contextual_fit = self._calculate_contextual_fit_score(exemplar["text"], conversation_context)
        information_density = self._calculate_information_density_score(exemplar["text"])
        sentiment_analysis = self._calculate_sentiment_analysis_score(exemplar["text"])
        historical_effectiveness = self._calculate_historical_effectiveness_score(exemplar["text"])

        total_score = (
            self.scoring_weights["relevance"] * relevance +
            self.scoring_weights["contextual_fit"] * contextual_fit +
            self.scoring_weights["information_density"] * information_density +
            self.scoring_weights["sentiment_analysis"] * sentiment_analysis +
            self.scoring_weights["historical_effectiveness"] * historical_effectiveness
        )
        return total_score

    def get_response(self, query):
        """Generates a response for a given query using USP."""
        if not self.exemplar_store:
            return "I'm sorry, I don't have any information to help you at the moment. Please try again later."

        query_embedding = self.embedding_model.encode(query)
        conversation_context = self._get_conversation_context()

        scored_exemplars = []
        for exemplar in self.exemplar_store:
            score = self._score_exemplar(query_embedding, exemplar, conversation_context)
            scored_exemplars.append((exemplar["text"], score))

        # Sort by score in descending order
        scored_exemplars.sort(key=lambda x: x[1], reverse=True)

        best_response = scored_exemplars[0][0] if scored_exemplars else "No suitable response found."
        
        self.conversation_history.append({"query": query, "response": best_response})
        return best_response

    def provide_feedback(self, query, response, feedback_type="implicit"): # Simplified feedback mechanism
        """Simulates receiving feedback on a response."""
        print(f"Feedback received for query: '{query}', response: '{response}', type: {feedback_type}")
        # In a real system, this would update historical effectiveness scores or fine-tune models.
        # For demonstration, we just print it.

# --- Chatbot Simulation --- #
def run_chatbot_simulation():
    # Create a dummy unlabeled data file for demonstration
    with open("unlabeled_data.txt", "w", encoding="utf-8") as f:
        f.write("Our product warranty covers manufacturing defects for one year.\n")
        f.write("To reset your password, visit the 'Forgot Password' link on the login page.\n")
        f.write("Shipping usually takes 3-5 business days within the US.\n")
        f.write("For technical support, please contact our helpline at 1-800-TECH-HELP.\n")
        f.write("We offer a 30-day money-back guarantee on all purchases.\n")
        f.write("The new software update includes performance improvements and bug fixes.\n")
        f.write("You can track your order status using the tracking number provided in your email.\n")
        f.write("Our return policy allows returns within 14 days with original packaging.\n")
        f.write("Customer satisfaction is our top priority. How can I help you today?\n")
        f.write("Explore our comprehensive knowledge base for common issues and solutions.\n")
        f.write("We value your feedback to improve our services.\n")
        f.write("Our support team is available Monday to Friday, 9 AM to 5 PM EST.\n")
        f.write("The latest product features are detailed in our online user manual.\n")
        f.write("Payments can be made via credit card, PayPal, or bank transfer.\n")
        f.write("Is there anything else I can assist you with?\n")

    chatbot = USP_Chatbot()

    print("\n--- USP Chatbot Simulation Started ---")
    print("Type 'quit' to exit.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'quit':
            break

        response = chatbot.get_response(user_query)
        print(f"Bot: {response}")
        
        # Simulate implicit feedback
        if "thank you" in user_query.lower():
            chatbot.provide_feedback(user_query, response, feedback_type="positive_implicit")

    print("--- USP Chatbot Simulation Ended ---")

if __name__ == "__main__":
    run_chatbot_simulation()