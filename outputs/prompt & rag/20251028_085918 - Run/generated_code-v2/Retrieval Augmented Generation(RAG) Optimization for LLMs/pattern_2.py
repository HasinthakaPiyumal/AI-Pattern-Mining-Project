import random
import numpy as np
from sklearn.linear_model import LogisticRegression

class MockVectorDB:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, k=3):
        # Simple keyword matching for mock retrieval
        query_words = set(query.lower().split())
        scores = []
        for doc_id, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words.intersection(doc_words))
            scores.append((overlap, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:k]]

class MockLLM:
    def generate(self, query, context):
        # Simulate LLM response and self-rated confidence
        if "diagnosis" in query.lower() or "symptoms" in query.lower():
            response = f"Based on the context, a possible diagnosis is: {query.replace('What is the diagnosis for ', '').replace('What are the symptoms of ', '')}. Further tests may be required."
            confidence = random.uniform(0.5, 0.95) # High confidence for generation
        else:
            response = f"I am processing your request regarding: {query}. The context provided is: {' '.join(context[:2])}..."
            confidence = random.uniform(0.4, 0.8) # Moderate confidence
        return response, confidence

def mock_context_autorater(retrieved_context):
    # Simulate context sufficiency: if context has more than 1 document, it's sufficient
    # Or if a specific keyword is present
    if len(retrieved_context) > 1 and any("treatment" in doc.lower() for doc in retrieved_context):
        return 1  # Sufficient
    return 0  # Insufficient

class HallucinationPredictor:
    def __init__(self, threshold=0.5):
        self.model = LogisticRegression()
        self.threshold = threshold
        self._is_trained = False

    def train(self, X_train, y_train):
        # Mock training: in a real scenario, X_train would be features
        # (self-rated confidence, context sufficiency) and y_train would be
        # binary labels (hallucination/no hallucination)
        if not X_train.shape[0] > 0 or not y_train.shape[0] > 0:
            print("Warning: No data for training. Initializing with dummy values.")
            # Create dummy data for the model to be 'fitted'
            X_train = np.array([[0.8, 1], [0.3, 0], [0.9, 1]])
            y_train = np.array([0, 1, 0]) # 0 for no hallucination, 1 for hallucination

        self.model.fit(X_train, y_train)
        self._is_trained = True

    def predict_likelihood(self, confidence, context_sufficiency):
        if not self._is_trained:
            # If not trained, provide a default prediction or raise an error
            print("Warning: Hallucination Predictor not trained. Returning default likelihood.")
            return random.uniform(0.1, 0.6) # Random likelihood if not trained
        
        features = np.array([[confidence, context_sufficiency]])
        # The model predicts probability of the 'positive' class (hallucination = 1)
        if hasattr(self.model, 'predict_proba'):
            likelihood = self.model.predict_proba(features)[:, 1][0]
        else:
            # Fallback for models without predict_proba (shouldn't happen for LogisticRegression)
            likelihood = self.model.predict(features)[0] # This would be 0 or 1 directly
        return likelihood

class MedicalDiagnosisAssistant:
    def __init__(self, hallucination_threshold=0.6):
        self.medical_documents = [
            "Symptoms of common cold include runny nose, sore throat, and cough. Treatments often involve rest and fluids.",
            "Influenza (flu) typically presents with fever, body aches, chills, and fatigue. Vaccination is key for prevention.",
            "Diabetes mellitus is characterized by high blood sugar levels. Management includes diet, exercise, and medication.",
            "Hypertension, or high blood pressure, often has no symptoms. Regular monitoring and lifestyle changes are crucial.",
            "Allergic reactions can manifest as rashes, itching, swelling, and difficulty breathing. Antihistamines are common treatments.",
            "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, chills, and difficulty breathing.",
            "Migraine headaches are severe headaches often accompanied by throbbing pain, sensitivity to light and sound, and nausea. Rest in a dark, quiet room can help.",
            "Gastric ulcers are open sores that develop on the lining of the stomach. Symptoms include burning stomach pain, bloating, heartburn, and nausea. Treatment often involves medication to reduce stomach acid."
        ]
        self.vector_db = MockVectorDB(self.medical_documents)
        self.llm = MockLLM()
        self.hallucination_predictor = HallucinationPredictor(threshold=hallucination_threshold)
        # Mock training the hallucination predictor
        # In a real system, this would come from a dataset of (confidence, context_sufficiency, actual_hallucination_label)
        mock_X_train = np.array([
            [0.9, 1], [0.85, 1], [0.7, 0], [0.6, 1], [0.92, 1], [0.5, 0], [0.75, 1], [0.4, 0]
        ])
        mock_y_train = np.array([0, 0, 1, 1, 0, 1, 0, 1]) # 0: No hallucination, 1: Hallucination
        self.hallucination_predictor.train(mock_X_train, mock_y_train)

    def diagnose(self, patient_query):
        print(f"\nUser query: {patient_query}")

        # 1. Context Retrieval
        retrieved_context = self.vector_db.retrieve(patient_query)
        print(f"Retrieved context: {[doc[:50] + '...' for doc in retrieved_context]}")

        # 2. LLM Generation and Self-rated Confidence
        llm_response, self_rated_confidence = self.llm.generate(patient_query, retrieved_context)
        print(f"LLM initial response: {llm_response[:80]}...")
        print(f"LLM self-rated confidence: {self_rated_confidence:.2f}")

        # 3. Context Sufficiency Assessment
        context_sufficiency = mock_context_autorater(retrieved_context)
        print(f"Context sufficiency (1=sufficient, 0=insufficient): {context_sufficiency}")

        # 4. Hallucination Likelihood Prediction
        hallucination_likelihood = self.hallucination_predictor.predict_likelihood(self_rated_confidence, context_sufficiency)
        print(f"Predicted hallucination likelihood: {hallucination_likelihood:.2f}")

        # 5. Abstention Decision
        if hallucination_likelihood > self.hallucination_predictor.threshold:
            print(f"\n---> ABSTAINED: High likelihood of hallucination ({hallucination_likelihood:.2f} > {self.hallucination_predictor.threshold:.2f}). Human review recommended due to lack of confidence or insufficient context.")
            return "ABSTAINED: Insufficient information or high hallucination risk. Please consult a human expert."
        else:
            print(f"\n---> GENERATED: Diagnosis/Suggestion: {llm_response}")
            return llm_response

# --- Example Usage ---
if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant(hallucination_threshold=0.5)

    print("\n--- Scenario 1: Sufficient context, good confidence ---")
    assistant.diagnose("What are the symptoms and common treatment for common cold?")

    print("\n--- Scenario 2: Less relevant context, moderate confidence ---")
    assistant.diagnose("Tell me about very rare tropical disease symptoms.")

    print("\n--- Scenario 3: Query with some but not full context in mock DB, potentially leading to abstention ---")
    assistant.diagnose("What is the best medication for severe chronic migraines?")

    print("\n--- Scenario 4: Query that might push towards abstention ---")
    assistant.diagnose("Latest research on novel cancer therapies for stage 4 glioblastoma.")
