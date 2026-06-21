"""
Service recommender using two strategies:

Strategy A (primary): Sentence Transformer semantic similarity
  — Encodes requirements text and each service description into embeddings.
  — Returns services whose embedding similarity exceeds a threshold.
  — Fast, works offline, no training needed.

Strategy B (fallback / comparison): HuggingFace Zero-Shot Classification
  — Uses facebook/bart-large-mnli to score requirement text against service labels.
  — More accurate but slower (large model, ~1.6GB).
  — Enable by passing use_zero_shot=True.

In a viva, you can show both strategies and compare confidence scores.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_QUANTITIES = {
    "Managed IT Support"       : 10,
    "Cloud Hosting Basic"      : 1,
    "Cloud Hosting Pro"        : 1,
    "Network Firewall"         : 1,
    "Cybersecurity Audit"      : 1,
    "Custom Web App Dev"       : 80,
    "Mobile App Dev"           : 60,
    "Office 365 Setup"         : 10,
    "Emergency Support"        : 2,
    "AI Integration Consult"   : 10,
    "Server Maintenance"       : 1,
    "Database Design"          : 1,
    "UI/UX Design"             : 20,
    "QA and Testing"           : 20,
    "SEO and Digital Marketing": 3,
}

SIMILARITY_THRESHOLD = 0.15

_st_model = None
_zs_pipeline = None


def _load_st_model():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _st_model


def recommend(text: str, catalog_df, use_zero_shot: bool = False) -> list[dict]:
    """
    Returns a list of recommended service dicts, each with:
      service, category, reason, quantity, unit, unit_price, total, confidence
    """
    if use_zero_shot:
        return _recommend_zero_shot(text, catalog_df)
    return _recommend_semantic(text, catalog_df)


def _recommend_semantic(text: str, catalog_df) -> list[dict]:
    """
    Primary strategy: cosine similarity between requirements embedding
    and each service's (name + description) embedding.
    """
    model = _load_st_model()

    query_emb = model.encode([text], convert_to_numpy=True)

    service_texts = [
        f"{row['service_name']} — {row['description']}"
        for _, row in catalog_df.iterrows()
    ]
    service_embs = model.encode(service_texts, convert_to_numpy=True)

    sims = cosine_similarity(query_emb, service_embs)[0]

    results = []
    for i, (_, row) in enumerate(catalog_df.iterrows()):
        score = float(sims[i])
        if score < SIMILARITY_THRESHOLD:
            continue

        name = row["service_name"]
        qty_default = DEFAULT_QUANTITIES.get(name, int(row["min_qty"]))
        qty = max(int(row["min_qty"]), min(qty_default, int(row["max_qty"])))
        unit_price = float(row["unit_price"])
        total = round(qty * unit_price, 2)

        results.append({
            "service"   : name,
            "category"  : row["category"],
            "reason"    : f"Semantically matched to your requirements (confidence {score:.0%}).",
            "quantity"  : qty,
            "unit"      : row["unit"],
            "unit_price": unit_price,
            "total"     : total,
            "confidence": round(score, 3),
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Fallback if nothing matched
    if not results:
        fallback = catalog_df[catalog_df["service_name"].str.contains("Custom Web App", case=False)]
        if not fallback.empty:
            row = fallback.iloc[0]
            qty = max(int(row["min_qty"]), min(40, int(row["max_qty"])))
            unit_price = float(row["unit_price"])
            results.append({
                "service"   : row["service_name"],
                "category"  : row["category"],
                "reason"    : "General custom development — no specific services matched above threshold.",
                "quantity"  : qty,
                "unit"      : row["unit"],
                "unit_price": unit_price,
                "total"     : round(qty * unit_price, 2),
                "confidence": 0.0,
            })

    return results


def _recommend_zero_shot(text: str, catalog_df) -> list[dict]:
    """
    Fallback/comparison strategy: HuggingFace zero-shot classification.
    Uses facebook/bart-large-mnli to score service labels against requirements.
    Slower but more powerful for ambiguous requirements.
    """
    global _zs_pipeline
    from transformers import pipeline as hf_pipeline

    if _zs_pipeline is None:
        _zs_pipeline = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

    candidate_labels = catalog_df["service_name"].tolist()
    result = _zs_pipeline(text, candidate_labels=candidate_labels, multi_label=True)

    label_scores = dict(zip(result["labels"], result["scores"]))

    matched = []
    for _, row in catalog_df.iterrows():
        name = row["service_name"]
        score = label_scores.get(name, 0.0)
        if score < 0.25:
            continue

        qty_default = DEFAULT_QUANTITIES.get(name, int(row["min_qty"]))
        qty = max(int(row["min_qty"]), min(qty_default, int(row["max_qty"])))
        unit_price = float(row["unit_price"])

        matched.append({
            "service"   : name,
            "category"  : row["category"],
            "reason"    : f"Zero-shot classification confidence: {score:.0%}.",
            "quantity"  : qty,
            "unit"      : row["unit"],
            "unit_price": unit_price,
            "total"     : round(qty * unit_price, 2),
            "confidence": round(score, 3),
        })

    matched.sort(key=lambda x: x["confidence"], reverse=True)
    return matched
