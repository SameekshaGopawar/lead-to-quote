"""
Category classifier using Sentence Transformers.

Approach: zero-shot semantic similarity.
Each project category has a set of prototype sentences that describe it.
The input requirements text is encoded into a 384-dim embedding.
Cosine similarity is computed against the mean prototype embedding per category.
The category with the highest similarity score is returned.

No training data required. Works because all-MiniLM-L6-v2 understands
semantic meaning, not just keyword overlap.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── CATEGORY PROTOTYPES ───────────────────────────────────────────────────────
# Multiple prototype sentences per category improve embedding coverage.
# These describe what a typical requirement in this category sounds like.
CATEGORY_PROTOTYPES = {
    "CRM & Sales System": [
        "We need a CRM to manage our sales leads and customer pipeline.",
        "Build a customer relationship management system with sales tracking.",
        "We want to track our leads, deals, and sales team performance.",
        "Need a platform to manage customer contacts and sales funnel.",
    ],
    "Healthcare Portal": [
        "Build a patient management portal for our clinic.",
        "We need a healthcare system with appointment booking for doctors.",
        "Hospital management software with medical records and patient history.",
        "Online doctor consultation and appointment scheduling platform.",
    ],
    "E-Commerce Platform": [
        "We want to build an online store with shopping cart and payments.",
        "E-commerce website with product catalog, inventory, and checkout.",
        "We need an online shop with payment gateway integration.",
        "Build a retail website where customers can browse and buy products.",
    ],
    "Education Platform": [
        "We need a learning management system for students and teachers.",
        "Online education platform with courses, quizzes, and certificates.",
        "School management software with student tracking and assignments.",
        "Build an e-learning platform with video content and progress tracking.",
    ],
    "Project Management Tool": [
        "We need project management software with task tracking and timelines.",
        "Construction site management with milestone tracking and reporting.",
        "Build a tool to manage team tasks, deadlines, and project workflows.",
        "We want a platform to track project progress and team collaboration.",
    ],
    "Cloud Infrastructure": [
        "We need cloud hosting and server infrastructure for our application.",
        "Set up cloud deployment with auto-scaling and monitoring.",
        "Migrate our on-premise servers to cloud infrastructure.",
        "We need DevOps setup with CI/CD pipeline and cloud hosting.",
    ],
    "Cybersecurity & Compliance": [
        "We need a security audit and vulnerability assessment for our systems.",
        "Implement firewall and network security for our organization.",
        "We require compliance audit and penetration testing services.",
        "Cybersecurity assessment and data protection implementation.",
    ],
    "AI & Automation": [
        "We want to automate our business workflows using AI.",
        "Build a chatbot and AI-powered automation for our operations.",
        "Implement machine learning models for business process automation.",
        "We need NLP and AI integration to automate data processing.",
    ],
    "Business Intelligence": [
        "We need a business intelligence dashboard with data analytics.",
        "Build a reporting platform with charts and data visualisation.",
        "We want real-time analytics and KPI tracking dashboards.",
        "Data analytics platform to visualise sales and business metrics.",
    ],
    "Mobile Application": [
        "We need a mobile app for Android and iOS.",
        "Build a cross-platform mobile application using Flutter.",
        "Develop a mobile app for our customers to use on their phones.",
        "We want a native mobile app with push notifications and offline mode.",
    ],
    "Web Application": [
        "We need a custom web application and portal.",
        "Build a full-stack web platform for our business.",
        "We want a web-based system that our team can access online.",
        "Develop a web portal with user login and admin dashboard.",
    ],
    "Office & Productivity": [
        "We need Microsoft 365 and Office setup for our team.",
        "Set up email, collaboration tools, and productivity software.",
        "We need Microsoft Teams and SharePoint configured for our staff.",
        "Office productivity suite setup and migration.",
    ],
    "IT Support & Maintenance": [
        "We need ongoing IT support and helpdesk services.",
        "Managed IT support and infrastructure maintenance for our company.",
        "We want a support contract covering server and network maintenance.",
        "IT helpdesk and monitoring services for our organization.",
    ],
}

_model = None
_prototype_embeddings: dict[str, np.ndarray] = {}


def _load_model():
    global _model, _prototype_embeddings
    if _model is not None:
        return
    _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for category, sentences in CATEGORY_PROTOTYPES.items():
        embeddings = _model.encode(sentences, convert_to_numpy=True)
        _prototype_embeddings[category] = embeddings.mean(axis=0, keepdims=True)


def classify(text: str) -> tuple[str, float]:
    """
    Classifies requirements text into a project category.

    Returns:
        (category_label, confidence_score)  where confidence is 0.0–1.0
    """
    _load_model()

    query_embedding = _model.encode([text], convert_to_numpy=True)

    scores = {}
    for category, proto_embedding in _prototype_embeddings.items():
        sim = cosine_similarity(query_embedding, proto_embedding)[0][0]
        scores[category] = float(sim)

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    # Normalise score to 0–1 range (cosine similarity is already -1 to 1,
    # but for positive text it stays roughly 0.1–0.9)
    confidence = round((best_score + 1) / 2, 3)

    return best_category, confidence
