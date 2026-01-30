"""
Intent Classification Module
Uses TF-IDF + Logistic Regression for supervised learning
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import os


class IntentClassifier:
    """
    Supervised Intent Classifier for logistics queries
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            lowercase=True
        )
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.is_trained = False

    def preprocess_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def train(self, dataset_path):
        """
        Train the intent classifier

        Args:
            dataset_path (str): Path to CSV file with 'text' and 'intent'
        """
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} samples")
        print(f"Intents: {df['intent'].unique().tolist()}")

        # Preprocess
        df['text'] = df['text'].apply(self.preprocess_text)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'],
            df['intent'],
            test_size=0.2,
            random_state=42,
            stratify=df['intent']
        )

        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train model
        print("Training model...")
        self.model.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 50)
        print(f"Training Accuracy: {accuracy:.2%}")
        print("=" * 50)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self.is_trained = True

        return {
            "accuracy": accuracy,
            "report": classification_report(y_test, y_pred, output_dict=True)
        }

    def predict(self, text):
        """Predict intent for a single query"""
        if not self.is_trained:
            raise Exception("Model not trained or loaded!")

        text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([text])

        intent = self.model.predict(text_vec)[0]
        confidence = max(self.model.predict_proba(text_vec)[0])

        return {
            "intent": intent,
            "confidence": round(confidence, 2)
        }

    def save(self, model_path):
        """Save trained model"""
        if not self.is_trained:
            raise Exception("Cannot save untrained model!")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "model": self.model
                },
                f
            )

        print(f"Model saved to {model_path}")

    def load(self, model_path):
        """Load trained model"""
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.is_trained = True

        print(f"Model loaded from {model_path}")


# =======================
# TRAINING ENTRY POINT
# =======================
if __name__ == "__main__":
    print("=" * 60)
    print("INTENT CLASSIFIER TRAINING")
    print("=" * 60)

    classifier = IntentClassifier()

    # Train model
    classifier.train("data/dataset.csv")

    # Save model
    classifier.save("models/intent_model.pkl")

    # Test predictions
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    test_queries = [
        "Bhai price batao Mumbai to Delhi 5kg",
        "Pickup karna hai urgent",
        "Mera order track karo",
        "COD available hai kya"
    ]

    for query in test_queries:
        result = classifier.predict(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']})")
