import numpy as np
import random
from sklearn.linear_model import LogisticRegression

class RAGSystem:
    def retrieve_context(self, query: str) -> str:
        simulated_contexts = {
            "fever and cough": "Medical literature on respiratory infections, common cold, and flu.",
            "chest pain": "Cardiology guidelines, information on angina, myocardial infarction, and GERD.",
            "headache severe": "Neurology articles, migraine information, and possible causes of severe headaches.",
            "skin rash itchy": "Dermatology resources on eczema, allergic reactions, and fungal infections.",
            "diabetes management": "Endocrinology journals, insulin therapy guidelines, and dietary recommendations for diabetes."
        }
        return simulated_contexts.get(query.lower(), "General medical knowledge base.")

    def generate_diagnosis(self, query: str, context: str) -> str:
        if "fever and cough" in query.lower() and "respiratory infections" in context.lower():
            return "Possible diagnosis: Viral respiratory infection."
        elif "chest pain" in query.lower() and "angina" in context.lower():
            return "Possible diagnosis: Angina pectoris, further investigation needed."
        elif "headache severe" in query.lower() and "migraine" in context.lower():
            return "Possible diagnosis: Migraine with aura."
        elif "skin rash itchy" in query.lower() and "eczema" in context.lower():
            return "Possible diagnosis: Atopic dermatitis."
        elif "diabetes management" in query.lower() and "insulin therapy" in context.lower():
            return "Recommendation: Review current insulin regimen and blood glucose monitoring."
        return f"Based on the query '{query}' and context, a general diagnostic statement."

class ConfidenceEstimator:
    def estimate_confidence(self, diagnosis_text: str, context: str) -> float:
        if "further investigation needed" in diagnosis_text:
            return random.uniform(0.4, 0.6)
        elif "General medical knowledge base" in context:
            return random.uniform(0.5, 0.7)
        return random.uniform(0.7, 0.95)

class ContextAutorater:
    def assess_sufficiency(self, query: str, context: str) -> int:
        if "General medical knowledge base" in context:
            return 0
        return 1 if random.random() > 0.3 else 0

class AbstentionPredictor:
    def __init__(self, threshold: float = 0.6):
        self.model = LogisticRegression(solver='liblinear')
        self.threshold = threshold

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        print("Abstention predictor trained.")

    def predict_abstain(self, confidence: float, context_sufficiency: int) -> bool:
        features = np.array([[confidence, context_sufficiency]])
        probabilities = self.model.predict_proba(features)[0]
        p_safe_to_respond = probabilities[1]

        if p_safe_to_respond < self.threshold:
            return True
        else:
            return False

def main():
    rag_system = RAGSystem()
    confidence_estimator = ConfidenceEstimator()
    context_autorater = ContextAutorater()
    abstention_predictor = AbstentionPredictor(threshold=0.65)

    X_train = []
    y_train = []

    for _ in range(50):
        X_train.append([random.uniform(0.8, 0.95), 1])
        y_train.append(1)

    for _ in range(50):
        X_train.append([random.uniform(0.4, 0.6), 0])
        y_train.append(0)

    for _ in range(20):
        X_train.append([random.uniform(0.7, 0.9), 0])
        y_train.append(random.choice([0, 0, 1]))

    for _ in range(20):
        X_train.append([random.uniform(0.4, 0.7), 1])
        y_train.append(random.choice([0, 0, 1]))
    
    for _ in range(10):
        X_train.append([random.uniform(0.4, 0.95), random.choice([0,1])])
        y_train.append(random.choice([0,1]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    abstention_predictor.train(X_train, y_train)

    print("\n--- Medical Diagnosis Assistant Ready ---")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your medical query: ")
        if query.lower() == 'exit':
            break

        context = rag_system.retrieve_context(query)
        diagnosis = rag_system.generate_diagnosis(query, context)
        confidence = confidence_estimator.estimate_confidence(diagnosis, context)
        context_sufficiency = context_autorater.assess_sufficiency(query, context)

        print(f"\nInternal Process:")
        print(f"  Retrieved Context: {context}")
        print(f"  Initial Diagnosis (LLM): {diagnosis}")
        print(f"  LLM Self-rated Confidence: {confidence:.2f}")
        print(f"  Context Sufficiency (0=Insufficient, 1=Sufficient): {context_sufficiency}")

        should_abstain = abstention_predictor.predict_abstain(confidence, context_sufficiency)

        if should_abstain:
            print("\nAssistant Response: I am unable to provide a definitive diagnosis at this moment due to insufficient context or low confidence in the generated response. Please provide more information or consult a human expert.")
        else:
            print(f"\nAssistant Response: {diagnosis}")
            print(f"  Supporting Note: This diagnosis is provided with a confidence of {confidence:.2f} and based on the sufficiency of retrieved context.")

if __name__ == "__main__":
    main()