"""A dummy black-box credit risk classification model."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class CreditRiskModel:
    """A placeholder for a black-box credit risk classification model."""
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(random_state=random_state)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains the black-box model."""
        print("Training black-box credit risk model...")
        self.model.fit(X_train, y_train)
        print("Black-box model training complete.")

    def predict_proba(self, X_inference: pd.DataFrame) -> pd.DataFrame:
        """Predicts probabilities for credit default."""
        if not hasattr(self.model, 'classes_'):
            raise RuntimeError("Model not trained. Please call .train() first.")
        probabilities = self.model.predict_proba(X_inference)
        return pd.DataFrame(probabilities, columns=self.model.classes_, index=X_inference.index)

    def get_feature_names(self):
        return self.model.feature_names_in_.tolist() if hasattr(self.model, 'feature_names_in_') else []
