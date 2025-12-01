import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import logging
from tqdm import tqdm

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Data Module (data_utils.py) ---
class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path
        if self.data_path is None:
            logger.info("Using dummy dataset for demonstration.")
            self._create_dummy_data()
        else:
            self.load_data()

    def _create_dummy_data(self):
        self.df = pd.DataFrame({
            'ticket_id': range(1, 11),
            'text': [
                "My internet is not working at all.",
                "I can't access my email account.",
                "My laptop screen is broken, need repair.",
                "Printer is offline and won't print.",
                "Need help resetting my password.",
                "Slow internet speed, pages loading very slowly.",
                "Software installation failed, error code 0x80070005.",
                "My phone battery drains too fast.",
                "Payment gateway is down, customers can't checkout.",
                "How do I configure my new router?"
            ],
            'category': [
                "Network Issue",
                "Account Access",
                "Hardware Repair",
                "Printer Support",
                "Account Access",
                "Network Issue",
                "Software Issue",
                "Hardware Repair",
                "Billing/Payment",
                "Network Issue"
            ]
        })

    def load_data(self):
        # In a real scenario, load data from CSV, DB, etc.
        # For this example, we'll just use the dummy data if data_path is not specified
        if self.data_path:
            try:
                self.df = pd.read_csv(self.data_path)
                logger.info(f"Data loaded from {self.data_path}")
            except FileNotFoundError:
                logger.error(f"Data file not found at {self.data_path}. Using dummy data instead.")
                self._create_dummy_data()
        else:
            self._create_dummy_data()

    def get_train_data(self):
        return self.df['text'].tolist(), self.df['category'].tolist()

# --- 2. Embedding Model Module (embedding_model.py) ---
class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"SentenceTransformer model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            # Fallback or raise error
            raise

    def embed(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        logger.info("Embeddings generated.")
        return embeddings

# --- 3. KNN Exemplar Selection Module (knn_selector.py) ---
class KNNExemplarSelector:
    def __init__(self, train_embeddings, train_categories, k=3):
        self.train_embeddings = train_embeddings
        self.train_categories = train_categories
        self.k = k
        self.nn_model = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.nn_model.fit(self.train_embeddings)
        logger.info(f"KNN model initialized with k={self.k}.")

    def select_exemplars(self, query_embedding, train_texts):
        distances, indices = self.nn_model.kneighbors(query_embedding.reshape(1, -1))
        exemplars = []
        for i in indices[0]:
            exemplars.append({
                'text': train_texts[i],
                'category': self.train_categories[i]
            })
        logger.info(f"Selected {len(exemplars)} exemplars using KNN.")
        return exemplars

# --- 4 & 5. Few-Shot Prompting Module & LLM Integration (few_shot_classifier.py) ---
class FewShotClassifier:
    def __init__(self, llm_client_type='dummy', model_name='gpt-3.5-turbo'):
        self.llm_client_type = llm_client_type
        self.model_name = model_name
        # In a real application, initialize OpenAI client or local Transformers pipeline
        # For dummy, we'll just have a placeholder
        if self.llm_client_type == 'dummy':
            logger.warning("Using dummy LLM client. No actual LLM calls will be made.")
        else:
            logger.info(f"Initializing LLM client for {self.llm_client_type} with model {self.model_name}")
            # Example for OpenAI (requires openai package and API key)
            # import openai
            # self.llm_client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")
            pass # Placeholder for actual LLM initialization

    def _format_exemplar(self, exemplar):
        return f"Ticket: '{exemplar['text']}'\nCategory: '{exemplar['category']}'"

    def _construct_prompt(self, exemplars, new_ticket_text):
        instruction = (
            "You are an expert customer support ticket categorizer. "
            "Classify the given new ticket into one of the provided categories based on the examples. "
            "Respond only with the predicted category."
        )

        exemplar_string = "\n\n".join([self._format_exemplar(e) for e in exemplars])

        prompt = (
            f"{instruction}\n\n"
            f"--- Examples ---\n"
            f"{exemplar_string}\n\n"
            f"--- New Ticket to Classify ---\n"
            f"Ticket: '{new_ticket_text}'\nCategory: "
        )
        return prompt

    def classify_ticket(self, exemplars, new_ticket_text):
        prompt = self._construct_prompt(exemplars, new_ticket_text)
        logger.info(f"Generated prompt for LLM:\n{prompt}")

        if self.llm_client_type == 'dummy':
            # Simulate LLM response based on keywords or a simple rule
            if "internet" in new_ticket_text.lower() or "network" in new_ticket_text.lower():
                predicted_category = "Network Issue"
            elif "account" in new_ticket_text.lower() or "password" in new_ticket_text.lower():
                predicted_category = "Account Access"
            elif "screen" in new_ticket_text.lower() or "hardware" in new_ticket_text.lower() or "phone battery" in new_ticket_text.lower():
                predicted_category = "Hardware Repair"
            elif "printer" in new_ticket_text.lower():
                predicted_category = "Printer Support"
            elif "software" in new_ticket_text.lower() or "installation" in new_ticket_text.lower():
                predicted_category = "Software Issue"
            elif "payment" in new_ticket_text.lower() or "billing" in new_ticket_text.lower():
                predicted_category = "Billing/Payment"
            elif "router" in new_ticket_text.lower():
                predicted_category = "Network Issue"
            else:
                predicted_category = "Unknown Category"
            logger.info(f"Dummy LLM predicted: '{predicted_category}'")
            return predicted_category
        else:
            try:
                # Example for OpenAI API call
                # response = self.llm_client.chat.completions.create(
                #     model=self.model_name,
                #     messages=[
                #         {"role": "system", "content": "You are an expert customer support ticket categorizer."},
                #         {"role": "user", "content": prompt}
                #     ],
                #     max_tokens=50 # Expect a short category name
                # )
                # predicted_category = response.choices[0].message.content.strip()
                # logger.info(f"LLM predicted: '{predicted_category}'")
                # return predicted_category
                return "Actual LLM call placeholder"
            except Exception as e:
                logger.error(f"Error during LLM API call: {e}")
                return "Error in Classification"

# --- 6. Main Application Logic (main.py) ---
def main():
    logger.info("Starting Intelligent Customer Support Ticket Categorization application.")

    # 1. Load Data
    data_loader = DataLoader()
    train_texts, train_categories = data_loader.get_train_data()

    # 2. Initialize Embedding Model
    embedding_model = EmbeddingModel()

    # 3. Generate Embeddings for Training Data
    logger.info("Generating embeddings for training data...")
    train_embeddings = embedding_model.embed(train_texts)
    logger.info("Finished generating embeddings for training data.")

    # 4. Initialize KNN Exemplar Selector
    knn_selector = KNNExemplarSelector(train_embeddings, train_categories, k=3)

    # 5. Initialize Few-Shot Classifier with LLM Integration
    # Use 'dummy' for local testing without an actual LLM API key
    # Or change to 'openai' if you have the API key and 'openai' package installed
    few_shot_classifier = FewShotClassifier(llm_client_type='dummy')

    # --- Simulate a new incoming ticket ---
    new_ticket_text = "My VPN is not connecting, I can't work from home."
    logger.info(f"\nProcessing new ticket: '{new_ticket_text}'")

    # a. Generate embedding for new ticket
    new_ticket_embedding = embedding_model.embed(new_ticket_text)

    # b. Use KNN to find exemplars
    selected_exemplars = knn_selector.select_exemplars(new_ticket_embedding, train_texts)
    logger.info(f"Selected exemplars: {selected_exemplars}")

    # c. Classify the new ticket using few-shot prompting
    predicted_category = few_shot_classifier.classify_ticket(selected_exemplars, new_ticket_text)
    logger.info(f"Final predicted category for ticket '{new_ticket_text}': '{predicted_category}'")

    # Another example
    new_ticket_text_2 = "My email is constantly getting spam, how to filter it?"
    logger.info(f"\nProcessing new ticket: '{new_ticket_text_2}'")
    new_ticket_embedding_2 = embedding_model.embed(new_ticket_text_2)
    selected_exemplars_2 = knn_selector.select_exemplars(new_ticket_embedding_2, train_texts)
    predicted_category_2 = few_shot_classifier.classify_ticket(selected_exemplars_2, new_ticket_text_2)
    logger.info(f"Final predicted category for ticket '{new_ticket_text_2}': '{predicted_category_2}'")

if __name__ == '__main__':
    main()
