import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class DataGenerationModule:
    def __init__(self):
        self.historical_queries = [
            "What is the status of my order?",
            "How do I return a product?",
            "Can I change my shipping address after placing an order?",
            "Tell me about your privacy policy.",
            "My product arrived damaged, what should I do?",
            "What are the specifications of the new XYZ laptop model?",
            "How do I troubleshoot connection issues with my smart speaker?",
            "Where can I find my order history?",
            "What payment methods do you accept?",
            "I want to cancel my subscription.",
            "What's the difference between product A and product B?",
            "Can I use my gift card for international purchases?",
            "How long does shipping usually take to Europe?",
            "The website is not loading correctly on my browser.",
            "I forgot my password, how can I reset it?"
        ]
        self.faq_keywords = ["return", "shipping", "payment", "cancel", "password"]
        self.complex_keywords = ["troubleshoot", "specifications", "damaged", "difference", "international purchase"]

    def _llm_strategy_outcome_simulator(self, query):
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in self.faq_keywords) and not any(keyword in query_lower for keyword in self.complex_keywords):
            return {"no_retrieval": False, "faq_retrieval": True, "multi_document": True}
        elif any(keyword in query_lower for keyword in self.complex_keywords):
            return {"no_retrieval": False, "faq_retrieval": False, "multi_document": True}
        else:
            return {"no_retrieval": True, "faq_retrieval": True, "multi_document": True}

    def _labeling_logic(self, query, outcomes):
        if outcomes["no_retrieval"]:
            return "simple"
        elif outcomes["faq_retrieval"]:
            return "moderate"
        elif outcomes["multi_document"]:
            return "complex"
        return "unknown"

    def generate_dataset(self):
        data = []
        for query in self.historical_queries:
            outcomes = self._llm_strategy_outcome_simulator(query)
            label = self._labeling_logic(query, outcomes)
            data.append({"query": query, "complexity_label": label})
        
        df = pd.DataFrame(data)

        df.loc[df["query"].str.contains("status of my order|where can i find my order history", case=False), "complexity_label"] = "simple"
        df.loc[df["query"].str.contains("troubleshoot connection issues|specifications of the new|product arrived damaged|difference between product", case=False), "complexity_label"] = "complex"
        df.loc[df["query"].str.contains("how do i return|change my shipping address|privacy policy|payment methods|cancel my subscription|forgot my password", case=False), "complexity_label"] = "moderate"

        return df

class QueryComplexityClassifierModule:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = RandomForestClassifier(random_state=42)

    def train_classifier(self, df):
        X = df["query"]
        y = df["complexity_label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier.fit(X_train_vec, y_train)

        y_pred = self.classifier.predict(X_test_vec)

        return self.vectorizer, self.classifier

class AdaptiveChatbotModule:
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def _predict_complexity(self, query):
        query_vec = self.vectorizer.transform([query])
        return self.classifier.predict(query_vec)[0]

    def _select_strategy(self, complexity_label):
        if complexity_label == "simple":
            return "simple_response"
        elif complexity_label == "moderate":
            return "faq_based_response"
        elif complexity_label == "complex":
            return "multi_step_response"
        return "default_response"

    def _generate_response(self, strategy, query):
        if strategy == "simple_response":
            return f"Simple answer for: '{query}' - Your query is straightforward and directly answered."
        elif strategy == "faq_based_response":
            return f"FAQ-based answer for: '{query}' - Retrieving information from our knowledge base."
        elif strategy == "multi_step_response":
            return f"Multi-step detailed answer for: '{query}' - Synthesizing information from multiple sources for a comprehensive reply."
        else:
            return f"Default response for: '{query}' - Unable to determine complexity, providing a general answer."

    def get_chatbot_response(self, query):
        complexity = self._predict_complexity(query)
        strategy = self._select_strategy(complexity)
        response = self._generate_response(strategy, query)
        return response

if __name__ == "__main__":
    data_gen = DataGenerationModule()
    training_df = data_gen.generate_dataset()

    classifier_module = QueryComplexityClassifierModule()
    trained_vectorizer, trained_classifier = classifier_module.train_classifier(training_df)

    chatbot = AdaptiveChatbotModule(trained_vectorizer, trained_classifier)

    print("\n--- Adaptive Chatbot Test --- ")
    test_queries = [
        "Hello, what is my order status?",
        "How do I return a defective item?",
        "I need help troubleshooting my smart TV connection.",
        "What is your refund policy?",
        "Tell me about the new features in your premium plan."
    ]

    for query in test_queries:
        response = chatbot.get_chatbot_response(query)
        print(f"Query: {query}\nResponse: {response}\n")
