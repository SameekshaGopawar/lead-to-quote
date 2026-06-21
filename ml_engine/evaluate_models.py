"""
Fair model comparison: TF-IDF + SVM vs Sentence Transformers

Uses paraphrased test sentences that were NOT in the training data.
Both models are tested on the exact same inputs.

Run:
    python ml_engine/evaluate_models.py

Requirements:
    - ml_engine/tfidf_svm_model.pkl must exist (run train_model.py first)
    - sentence-transformers must be installed
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ml_engine.project_classifier import CATEGORY_PROTOTYPES

MODEL_PATH = Path(__file__).parent / "tfidf_svm_model.pkl"

# ── PARAPHRASED TEST SET ──────────────────────────────────────────────────────
# These sentences are deliberately written differently from the training data.
# Neither model has seen these exact phrasings before.
PARAPHRASED_TEST_SET = [
    # CRM & Sales System
    ("Track our clients and manage deal stages for the sales department.", "CRM & Sales System"),
    ("We want software to follow up with prospects and log customer calls.", "CRM & Sales System"),
    ("Build a tool to monitor our revenue pipeline and close rates.", "CRM & Sales System"),

    # Healthcare Portal
    ("Digital solution for scheduling consultations with physicians.", "Healthcare Portal"),
    ("We need a system where patients can view their test results online.", "Healthcare Portal"),
    ("Software to manage ward admissions and discharge records.", "Healthcare Portal"),

    # E-Commerce Platform
    ("We want customers to browse and purchase items through our website.", "E-Commerce Platform"),
    ("Build an online retail platform with a checkout and payment system.", "E-Commerce Platform"),
    ("We need a digital storefront with product listings and order management.", "E-Commerce Platform"),

    # Education Platform
    ("Platform for teachers to upload lessons and students to submit assignments.", "Education Platform"),
    ("We need a system to deliver online training modules to our employees.", "Education Platform"),
    ("Build a portal where students can track their academic progress.", "Education Platform"),

    # Project Management Tool
    ("Software to assign tasks to team members and track completion.", "Project Management Tool"),
    ("We need to monitor construction milestones and site progress.", "Project Management Tool"),
    ("Tool for planning sprints and managing workload across teams.", "Project Management Tool"),

    # Cloud Infrastructure
    ("We need to host our application on remote servers with auto-scaling.", "Cloud Infrastructure"),
    ("Set up our deployment pipeline with containerisation and monitoring.", "Cloud Infrastructure"),
    ("Migrate our local servers to a cloud environment with backup.", "Cloud Infrastructure"),

    # Cybersecurity & Compliance
    ("We want an independent review of our systems for security gaps.", "Cybersecurity & Compliance"),
    ("Assess our network for weaknesses and provide a remediation plan.", "Cybersecurity & Compliance"),
    ("We need to meet data protection regulations for our industry.", "Cybersecurity & Compliance"),

    # AI & Automation
    ("We want to reduce manual data entry using intelligent automation.", "AI & Automation"),
    ("Build a conversational assistant for our customer service team.", "AI & Automation"),
    ("Use machine intelligence to predict which leads are likely to convert.", "AI & Automation"),

    # Business Intelligence
    ("We need visual reports showing our monthly sales performance.", "Business Intelligence"),
    ("Build an executive summary dashboard with key business metrics.", "Business Intelligence"),
    ("We want to analyse patterns in our customer purchase history.", "Business Intelligence"),

    # Mobile Application
    ("Develop a smartphone application for both Apple and Android users.", "Mobile Application"),
    ("We need a mobile solution for our field staff to log work orders.", "Mobile Application"),
    ("Build a consumer-facing app with GPS and in-app purchases.", "Mobile Application"),

    # Web Application
    ("We need an internal tool accessible through a web browser.", "Web Application"),
    ("Build a multi-user web system with role-based permissions.", "Web Application"),
    ("We want a custom online platform to manage our business processes.", "Web Application"),

    # Office & Productivity
    ("Set up corporate email and shared calendar for our staff.", "Office & Productivity"),
    ("We need cloud-based document storage and team collaboration tools.", "Office & Productivity"),
    ("Migrate our team to Microsoft 365 with Teams and SharePoint.", "Office & Productivity"),

    # IT Support & Maintenance
    ("We need a team to handle technical issues for our employees.", "IT Support & Maintenance"),
    ("Provide round-the-clock monitoring and maintenance for our servers.", "IT Support & Maintenance"),
    ("We want a managed service contract covering our entire IT infrastructure.", "IT Support & Maintenance"),
]


def evaluate_svm(texts, labels):
    if not MODEL_PATH.exists():
        print("SVM model not found. Run python ml_engine/train_model.py first.\n")
        return None, None

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    predictions = pipeline.predict(texts)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, zero_division=0)
    return accuracy, report, predictions


def evaluate_sentence_transformers(texts, labels):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Build prototype embeddings
    prototype_embeddings = {}
    for category, sentences in CATEGORY_PROTOTYPES.items():
        embs = model.encode(sentences, convert_to_numpy=True)
        prototype_embeddings[category] = embs.mean(axis=0, keepdims=True)

    predictions = []
    for text in texts:
        query_emb = model.encode([text], convert_to_numpy=True)
        scores = {
            cat: float(cosine_similarity(query_emb, proto)[0][0])
            for cat, proto in prototype_embeddings.items()
        }
        predictions.append(max(scores, key=scores.get))

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, zero_division=0)
    return accuracy, report, predictions


def run():
    texts  = [t for t, _ in PARAPHRASED_TEST_SET]
    labels = [l for _, l in PARAPHRASED_TEST_SET]

    print("=" * 65)
    print("MODEL COMPARISON — Paraphrased Test Set (unseen sentences)")
    print(f"Total test samples: {len(texts)} across 13 categories")
    print("=" * 65)

    # SVM
    print("\n── TF-IDF + SVM ──────────────────────────────────────────────")
    svm_acc, svm_report, svm_preds = evaluate_svm(texts, labels)
    if svm_acc is not None:
        print(f"Accuracy: {svm_acc:.2%}")
        print(svm_report)

    # Sentence Transformers
    print("\n── Sentence Transformers (all-MiniLM-L6-v2) ─────────────────")
    st_acc, st_report, st_preds = evaluate_sentence_transformers(texts, labels)
    print(f"Accuracy: {st_acc:.2%}")
    print(st_report)

    # Side-by-side summary
    print("\n── Summary ───────────────────────────────────────────────────")
    print(f"{'Model':<35} {'Accuracy':>10}")
    print("-" * 47)
    if svm_acc is not None:
        print(f"{'TF-IDF + SVM':<35} {svm_acc:>10.2%}")
    print(f"{'Sentence Transformers':<35} {st_acc:>10.2%}")

    # Show where SVM failed but ST succeeded
    if svm_preds is not None:
        print("\n── Cases where SVM failed, Sentence Transformers succeeded ──")
        found = False
        for i, (text, true_label) in enumerate(zip(texts, labels)):
            if svm_preds[i] != true_label and st_preds[i] == true_label:
                print(f"\n  Input : \"{text}\"")
                print(f"  True  : {true_label}")
                print(f"  SVM   : {svm_preds[i]}  ✗")
                print(f"  ST    : {st_preds[i]}  ✓")
                found = True
        if not found:
            print("  None — both models agreed on all examples.")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    run()
