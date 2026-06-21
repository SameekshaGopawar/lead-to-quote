"""
ML-powered quote generation engine.

Category detection: Sentence Transformer semantic similarity (all-MiniLM-L6-v2)
Service recommendation: Sentence Transformer semantic similarity
Timeline estimation: Rule-based (business logic — ML not appropriate)
Lead scoring: Rule-based (structured numeric data — ML not appropriate)
Summary generation: Template-based (deterministic, not NLP)
"""

import hashlib
from datetime import date, timedelta

from ml_engine.text_cleaner import build_input
from ml_engine.project_classifier import classify as classify_category
from ml_engine.service_matcher import recommend as recommend_services
from services_catalog import load_catalog, recalculate_totals


# ── TIMELINE RULES ────────────────────────────────────────────────────────────
# Kept rule-based: timeline depends on price and service count — structured
# business logic, not language patterns. ML would just re-learn these thresholds.
TIMELINE_PHASES = {
    "short": {
        "label": "4–6 weeks",
        "phases": [
            {"phase": "Discovery & Planning", "duration": "1 week",  "description": "Gather requirements, finalize scope and technical architecture."},
            {"phase": "Development",          "duration": "3 weeks", "description": "Core feature implementation and integration."},
            {"phase": "Testing & Delivery",   "duration": "1 week",  "description": "QA testing, bug fixes, and final handover."},
        ],
    },
    "medium": {
        "label": "6–10 weeks",
        "phases": [
            {"phase": "Discovery & Planning", "duration": "1 week",  "description": "Requirements gathering, wireframes, and project kickoff."},
            {"phase": "Design",               "duration": "1 week",  "description": "UI/UX design, prototyping, and client approval."},
            {"phase": "Development",          "duration": "6 weeks", "description": "Full-stack development, API integration, and database setup."},
            {"phase": "Testing & Delivery",   "duration": "2 weeks", "description": "QA, UAT, performance testing, and deployment."},
        ],
    },
    "long": {
        "label": "12–16 weeks",
        "phases": [
            {"phase": "Discovery & Planning",   "duration": "2 weeks", "description": "Deep-dive requirements, architecture design, and sprint planning."},
            {"phase": "Design",                 "duration": "2 weeks", "description": "Full UI/UX design system, prototyping, and sign-off."},
            {"phase": "Development — Phase 1",  "duration": "5 weeks", "description": "Core modules, authentication, database, and primary features."},
            {"phase": "Development — Phase 2",  "duration": "4 weeks", "description": "Secondary features, integrations, and admin panels."},
            {"phase": "Testing & Delivery",     "duration": "3 weeks", "description": "QA, UAT, load testing, documentation, and production deployment."},
        ],
    },
}

SUMMARY_TEMPLATES = [
    (
        "Thank you for reaching out, {name}. Based on our review of {company}'s requirements, "
        "we are pleased to present this tailored proposal for a {category}. "
        "Our recommended solution encompasses {service_count} carefully selected services "
        "with an estimated delivery timeline of {timeline}, designed to meet your {industry} industry needs. "
        "We are confident this proposal delivers strong value within your stated budget and look forward to partnering with you."
    ),
    (
        "Dear {name}, we appreciate the opportunity to support {company} with your upcoming project. "
        "After thoroughly analysing your requirements, we have prepared a comprehensive quote for a {category}. "
        "The proposed engagement includes {service_count} services totalling {total}, "
        "with an estimated completion of {timeline}. "
        "Our team is ready to begin immediately upon your approval."
    ),
    (
        "We are delighted to present this quotation to {name} at {company}. "
        "Your requirement for a {category} has been carefully assessed by our team. "
        "We recommend {service_count} services tailored specifically to the {industry} sector, "
        "with delivery estimated at {timeline}. "
        "Please review the line items below and do not hesitate to reach out with any questions."
    ),
]


def _estimate_timeline(items: list[dict], total: float) -> dict:
    if total < 5000 or len(items) <= 2:
        return TIMELINE_PHASES["short"]
    if total < 20000 or len(items) <= 5:
        return TIMELINE_PHASES["medium"]
    return TIMELINE_PHASES["long"]


def _score_lead(budget: float, total: float, requirements: str, priority: str) -> tuple[int, str]:
    score = 5
    parts = []

    if budget > 0:
        ratio = budget / total if total > 0 else 0
        if ratio >= 1.2:
            score += 2; parts.append("budget comfortably covers the quote")
        elif ratio >= 0.8:
            score += 1; parts.append("budget closely matches the quote")
        elif ratio < 0.5:
            score -= 2; parts.append("budget is significantly below the estimated cost")

    word_count = len(requirements.split())
    if word_count >= 15:
        score += 1; parts.append("detailed requirements provided")
    elif word_count < 6:
        score -= 1; parts.append("requirements are brief")

    if priority == "High":
        score += 1; parts.append("high priority engagement")
    elif priority == "Low":
        score -= 1

    score = max(1, min(10, score))
    reason = (", ".join(parts).capitalize() + ".") if parts else "Score based on available information."
    return score, reason


def _discount_percent(total: float, priority: str) -> float:
    if total >= 20000: return 10.0
    if total >= 10000: return 7.0
    if total >= 5000:  return 5.0
    return 0.0


def _build_summary(lead: dict, category: str, items: list[dict], timeline: str, total: float) -> str:
    idx = int(hashlib.md5(lead.get("customer_name", "x").encode()).hexdigest(), 16) % len(SUMMARY_TEMPLATES)
    return SUMMARY_TEMPLATES[idx].format(
        name          = lead.get("customer_name", ""),
        company       = lead.get("company", "your company"),
        category      = category,
        industry      = lead.get("industry", "your"),
        service_count = len(items),
        timeline      = timeline,
        total         = f"${total:,.0f}",
    )


def generate_quote(lead: dict, use_zero_shot: bool = False) -> dict:
    """
    ML-powered quote generation pipeline.

    Steps:
      1. Preprocess requirements text
      2. Sentence Transformer category classification
      3. Sentence Transformer service recommendation
      4. Rule-based: pricing, discount, timeline (business logic)
      5. Rule-based: lead scoring (structured numeric data)
      6. Template summary generation
    """
    catalog_df, errors = load_catalog()
    if errors:
        raise ValueError(f"Catalog error: {errors[0]}")

    requirements = str(lead.get("requested_items", ""))
    industry     = str(lead.get("industry", ""))
    budget       = float(lead.get("budget", 0))
    priority     = str(lead.get("priority", "Medium"))
    today        = date.today()
    valid_until  = today + timedelta(days=30)

    # Step 1 — Preprocess
    clean_text = build_input(requirements, industry)

    # Step 2 — ML: Category classification (Sentence Transformer)
    category, category_confidence = classify_category(clean_text)

    # Step 3 — ML: Service recommendation (Sentence Transformer)
    items = recommend_services(clean_text, catalog_df, use_zero_shot=use_zero_shot)

    # Trim to budget (remove lowest-confidence services if total exceeds 2× budget)
    if budget > 0:
        while len(items) > 1:
            current_total = sum(i["total"] for i in items)
            if current_total <= budget * 2:
                break
            items.pop()

    # Step 4 — Rule-based: pricing
    discount_pct = _discount_percent(sum(i["total"] for i in items), priority)
    totals       = recalculate_totals(items, discount_pct)

    # Step 5 — Rule-based: timeline
    tl_data  = _estimate_timeline(items, totals["total"])
    timeline = tl_data["label"]
    phases   = tl_data["phases"]

    # Step 6 — Rule-based: lead score
    lead_score, score_reason = _score_lead(budget, totals["total"], requirements, priority)

    # Step 7 — Template summary
    summary = _build_summary(lead, category, items, timeline, totals["total"])

    return {
        "customer_name"        : lead.get("customer_name", ""),
        "customer_email"       : lead.get("customer_email", ""),
        "company"              : lead.get("company", ""),
        "industry"             : industry,
        "quote_date"           : str(today),
        "valid_until"          : str(valid_until),
        "project_category"     : category,
        "category_confidence"  : category_confidence,
        "lead_score"           : lead_score,
        "lead_score_reason"    : score_reason,
        "recommended_services" : items,
        "items"                : items,
        "estimated_timeline"   : timeline,
        "timeline_breakdown"   : phases,
        "quote_summary"        : summary,
        "catalog_warnings"     : [],
        **totals,
    }
