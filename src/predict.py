"""
Inference module for SMS Spam Classification.
Provides an easy-to-use SMSClassifier class and CLI interface.
"""

import os
import sys
import argparse
import joblib
from pathlib import Path

# Add current dir to path
sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "spam_classifier_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"


class SMSClassifier:
    """Wrapper for trained model and TF-IDF vectorizer."""

    def __init__(self, auto_train: bool = True):
        self.model = None
        self.vectorizer = None
        self._load_or_train(auto_train)

    def _load_or_train(self, auto_train: bool):
        if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        elif auto_train:
            print("[INFO] Model not found on disk. Initiating training pipeline...")
            from train import train_pipeline
            self.model, self.vectorizer, _ = train_pipeline()
        else:
            raise FileNotFoundError(
                f"Trained models not found at {MODELS_DIR}. Run train.py first."
            )

    def predict(self, text: str) -> dict:
        """
        Classifies an input text.
        Returns prediction label, probability scores, and analysis details.
        """
        if not text or not text.strip():
            return {
                "label": "HAM",
                "is_spam": False,
                "spam_probability": 0.0,
                "ham_probability": 1.0,
                "confidence": 1.0,
                "cleaned_text": "",
            }

        cleaned = clean_text(text)
        features = self.vectorizer.transform([cleaned])

        pred_class = int(self.model.predict(features)[0])

        # Get probabilities if available
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            ham_prob = float(probs[0])
            spam_prob = float(probs[1])
        else:
            spam_prob = 1.0 if pred_class == 1 else 0.0
            ham_prob = 1.0 - spam_prob

        is_spam = pred_class == 1
        confidence = spam_prob if is_spam else ham_prob

        # Extract highlighted suspicious words
        spam_indicators = [
            "free", "win", "winner", "prize", "urgent", "cash", "claim",
            "award", "lottery", "gift", "card", "guaranteed", "call", "bonus",
            "txt", "money_token", "http_token", "phone_token", "kyc", "discount"
        ]
        tokens = cleaned.split()
        matched_indicators = [t for t in set(tokens) if t in spam_indicators]

        return {
            "label": "SPAM" if is_spam else "HAM",
            "is_spam": is_spam,
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4),
            "confidence": round(confidence, 4),
            "cleaned_text": cleaned,
            "flagged_keywords": matched_indicators
        }


def main():
    parser = argparse.ArgumentParser(description="Classify an SMS as Spam or Ham.")
    parser.add_argument(
        "--message", "-m",
        type=str,
        help="The SMS text message to analyze",
        default="WINNER!! You have won a £1,000 cash prize! Call 09061701461 to claim."
    )
    args = parser.parse_args()

    classifier = SMSClassifier()
    result = classifier.predict(args.message)

    print("\n" + "=" * 45)
    print(" SMS SPAM CLASSIFIER RESULT")
    print("=" * 45)
    print(f"Original SMS : {args.message}")
    print(f"Cleaned Text : {result['cleaned_text']}")
    print(f"Prediction   : {result['label']}")
    print(f"Confidence   : {result['confidence']:.2%}")
    print(f"Spam Score   : {result['spam_probability']:.2%}")
    print(f"Ham Score    : {result['ham_probability']:.2%}")
    if result["flagged_keywords"]:
        print(f"Trigger Words: {', '.join(result['flagged_keywords'])}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
