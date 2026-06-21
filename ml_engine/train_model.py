"""
Trains a TF-IDF + SVM classifier on the synthetic training data.
Use this as a comparison baseline in your viva against Sentence Transformers.

Run:
    python ml_engine/training_data_generator.py   # generate data first
    python ml_engine/train_model.py               # train and evaluate

Saves model to: ml_engine/tfidf_svm_model.pkl
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

DATA_PATH  = Path(__file__).parent.parent / "data" / "training_data.csv"
MODEL_PATH = Path(__file__).parent / "tfidf_svm_model.pkl"


def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. "
            "Run ml_engine/generate_training_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} training samples across {df['category'].nunique()} categories.")

    X = df["text"].tolist()
    y = df["category"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000)),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    print("\n── Test Set Classification Report ──")
    print(classification_report(y_test, y_pred))

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to {MODEL_PATH}")
    return pipeline


def predict(text: str) -> tuple[str, float]:
    """Load saved model and predict category for a single text."""
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    category = pipeline.predict([text])[0]
    # LinearSVC doesn't output probabilities natively; use decision function
    scores = pipeline.decision_function([text])[0]
    classes = pipeline.classes_
    confidence = float(
        (scores.max() - scores.min()) / (scores.max() - scores.min() + 1e-9)
    )
    return category, round(confidence, 3)


if __name__ == "__main__":
    train()
