"""
Local quote generation engine — no API keys, no internet required.
Uses keyword matching, scoring rules, and template-based summary generation.
"""

import re
from datetime import date, timedelta

from services_catalog import load_catalog, validate_quote_services, recalculate_totals

# ── PROJECT CATEGORY RULES ────────────────────────────────────────────────────
# Each entry: (category_label, [keywords])
CATEGORY_RULES = [
    ("CRM & Sales System",          ["crm", "lead", "sales", "pipeline", "customer relationship"]),
    ("Healthcare Portal",           ["health", "medical", "patient", "clinic", "doctor", "appointment", "hospital"]),
    ("E-Commerce Platform",         ["ecommerce", "e-commerce", "shop", "store", "cart", "payment", "inventory"]),
    ("Education Platform",          ["education", "lms", "course", "student", "learning", "school", "academy", "training"]),
    ("Project Management Tool",     ["project management", "task", "timeline", "milestone", "site", "construction"]),
    ("Cloud Infrastructure",        ["cloud", "hosting", "server", "vps", "infrastructure", "devops"]),
    ("Cybersecurity & Compliance",  ["security", "firewall", "audit", "vulnerability", "compliance", "penetration"]),
    ("AI & Automation",             ["ai", "automation", "workflow", "bot", "machine learning", "nlp"]),
    ("Business Intelligence",       ["dashboard", "report", "analytics", "bi", "data", "visuali"]),
    ("Mobile Application",          ["mobile", "android", "ios", "app", "flutter", "react native"]),
    ("Web Application",             ["web app", "web application", "portal", "platform", "system"]),
    ("Office & Productivity",       ["office", "microsoft", "365", "email", "collaboration", "productivity"]),
    ("IT Support & Maintenance",    ["support", "helpdesk", "maintenance", "managed", "monitoring"]),
]

# ── SERVICE MATCH RULES ───────────────────────────────────────────────────────
# Each entry: (service_name_substring, [trigger_keywords], default_qty)
SERVICE_MATCH_RULES = [
    ("Managed IT Support",          ["support", "helpdesk", "managed", "maintenance", "monitor"],   10),
    ("Cloud Hosting Basic",         ["cloud", "hosting", "vps", "server", "deploy", "basic"],        1),
    ("Cloud Hosting Pro",           ["cloud pro", "high traffic", "production", "scale", "large"],   1),
    ("Network Firewall",            ["firewall", "network security", "vpn", "network"],              1),
    ("Cybersecurity Audit",         ["security audit", "vulnerability", "penetration", "compliance", "audit"], 1),
    ("Custom Web App Dev",          ["web app", "dashboard", "portal", "crm", "system", "platform", "custom"], 80),
    ("Mobile App Dev",              ["mobile", "android", "ios", "app", "flutter"],                 60),
    ("Office 365 Setup",            ["office", "microsoft", "365", "email", "outlook"],             10),
    ("Emergency Support",           ["urgent", "emergency", "critical", "immediate", "asap"],        2),
    ("AI Integration Consult",      ["ai", "automation", "bot", "machine learning", "workflow", "nlp"], 10),
    ("Server Maintenance",          ["server", "maintenance", "uptime", "backup"],                   1),
    ("Database Design",             ["database", "db", "sql", "data model", "schema"],               1),
    ("UI/UX Design",                ["design", "ui", "ux", "user interface", "figma", "wireframe", "branding"], 20),
    ("QA and Testing",              ["test", "qa", "quality", "bug", "automation testing"],         20),
    ("SEO and Digital Marketing",   ["seo", "marketing", "digital", "social media", "google ads"],   3),
]

# ── TIMELINE RULES ────────────────────────────────────────────────────────────
TIMELINE_PHASES = {
    "short":  {
        "label": "4–6 weeks",
        "phases": [
            {"phase": "Discovery & Planning",  "duration": "1 week",  "description": "Gather requirements, finalize scope and technical architecture."},
            {"phase": "Development",           "duration": "3 weeks", "description": "Core feature implementation and integration."},
            {"phase": "Testing & Delivery",    "duration": "1 week",  "description": "QA testing, bug fixes, and final handover."},
        ],
    },
    "medium": {
        "label": "6–10 weeks",
        "phases": [
            {"phase": "Discovery & Planning",  "duration": "1 week",  "description": "Requirements gathering, wireframes, and project kickoff."},
            {"phase": "Design",                "duration": "1 week",  "description": "UI/UX design, prototyping, and client approval."},
            {"phase": "Development",           "duration": "6 weeks", "description": "Full-stack development, API integration, and database setup."},
            {"phase": "Testing & Delivery",    "duration": "2 weeks", "description": "QA, UAT, performance testing, and deployment."},
        ],
    },
    "long":   {
        "label": "12–16 weeks",
        "phases": [
            {"phase": "Discovery & Planning",  "duration": "2 weeks", "description": "Deep-dive requirements, architecture design, and sprint planning."},
            {"phase": "Design",                "duration": "2 weeks", "description": "Full UI/UX design system, prototyping, and sign-off."},
            {"phase": "Development — Phase 1", "duration": "5 weeks", "description": "Core modules, authentication, database, and primary features."},
            {"phase": "Development — Phase 2", "duration": "4 weeks", "description": "Secondary features, integrations, and admin panels."},
            {"phase": "Testing & Delivery",    "duration": "3 weeks", "description": "QA, UAT, load testing, documentation, and production deployment."},
        ],
    },
}

# ── QUOTE SUMMARY TEMPLATES ───────────────────────────────────────────────────
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


def _normalize(text: str) -> str:
    return text.lower().strip()


def _detect_category(requirements: str, industry: str) -> str:
    text = _normalize(f"{requirements} {industry}")
    for category, keywords in CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            return category
    return "Custom IT Solution"


def _match_services(requirements: str, industry: str, budget: float, catalog_df) -> list[dict]:
    text = _normalize(f"{requirements} {industry}")
    catalog_names = {row["service_name"].lower(): row for _, row in catalog_df.iterrows()}

    matched = []
    seen    = set()

    for service_substr, keywords, default_qty in SERVICE_MATCH_RULES:
        if not any(kw in text for kw in keywords):
            continue

        # Find the catalog row whose name contains this substring
        catalog_row = next(
            (row for name, row in catalog_names.items() if service_substr.lower() in name),
            None,
        )
        if catalog_row is None or catalog_row["service_name"] in seen:
            continue

        qty = max(int(catalog_row["min_qty"]), min(default_qty, int(catalog_row["max_qty"])))
        matched.append({
            "service"   : catalog_row["service_name"],
            "category"  : catalog_row["category"],
            "reason"    : f"Recommended based on your requirement for {keywords[0].replace('_', ' ')}.",
            "quantity"  : qty,
            "unit"      : catalog_row["unit"],
            "unit_price": float(catalog_row["unit_price"]),
            "total"     : round(qty * float(catalog_row["unit_price"]), 2),
        })
        seen.add(catalog_row["service_name"])

    # If nothing matched, fall back to Custom Web App Dev
    if not matched:
        fallback = next(
            (row for _, row in catalog_df.iterrows() if "Custom Web App" in row["service_name"]),
            None,
        )
        if fallback is not None:
            qty = max(int(fallback["min_qty"]), min(40, int(fallback["max_qty"])))
            matched.append({
                "service"   : fallback["service_name"],
                "category"  : fallback["category"],
                "reason"    : "General custom development to address your requirements.",
                "quantity"  : qty,
                "unit"      : fallback["unit"],
                "unit_price": float(fallback["unit_price"]),
                "total"     : round(qty * float(fallback["unit_price"]), 2),
            })

    # Trim to budget — remove lowest-priority items if total exceeds 2× budget
    if budget > 0:
        while len(matched) > 1:
            current_total = sum(i["total"] for i in matched)
            if current_total <= budget * 2:
                break
            matched.pop()

    return matched


def _estimate_timeline(items: list[dict], total: float) -> dict:
    if total < 5000 or len(items) <= 2:
        return TIMELINE_PHASES["short"]
    if total < 20000 or len(items) <= 5:
        return TIMELINE_PHASES["medium"]
    return TIMELINE_PHASES["long"]


def _score_lead(budget: float, total: float, requirements: str, priority: str) -> tuple[int, str]:
    score = 5
    reason_parts = []

    # Budget fit
    if budget > 0:
        ratio = budget / total if total > 0 else 0
        if ratio >= 1.2:
            score += 2
            reason_parts.append("budget comfortably covers the quote")
        elif ratio >= 0.8:
            score += 1
            reason_parts.append("budget closely matches the quote")
        elif ratio < 0.5:
            score -= 2
            reason_parts.append("budget is significantly below the estimated cost")

    # Requirement clarity
    word_count = len(requirements.split())
    if word_count >= 15:
        score += 1
        reason_parts.append("detailed requirements provided")
    elif word_count < 6:
        score -= 1
        reason_parts.append("requirements are brief")

    # Priority
    if priority == "High":
        score += 1
        reason_parts.append("high priority engagement")
    elif priority == "Low":
        score -= 1

    score = max(1, min(10, score))
    reason = (", ".join(reason_parts).capitalize() + ".") if reason_parts else "Score based on available information."
    return score, reason


def _discount_percent(total: float, priority: str) -> float:
    if total >= 20000:
        return 10.0
    if total >= 10000:
        return 7.0
    if total >= 5000:
        return 5.0
    return 0.0


def _build_summary(lead: dict, category: str, items: list[dict], timeline: str, total: float) -> str:
    import hashlib
    # Pick a template deterministically based on customer name
    idx = int(hashlib.md5(lead.get("customer_name", "x").encode()).hexdigest(), 16) % len(SUMMARY_TEMPLATES)
    template = SUMMARY_TEMPLATES[idx]
    return template.format(
        name          = lead.get("customer_name", ""),
        company       = lead.get("company", "your company"),
        category      = category,
        industry      = lead.get("industry", "your"),
        service_count = len(items),
        timeline      = timeline,
        total         = f"${total:,.0f}",
    )


def generate_quote(lead: dict, services_df=None) -> dict:
    """
    Fully local quote engine — no API calls, no internet required.
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

    # Step 1 — Categorise
    category = _detect_category(requirements, industry)

    # Step 2 — Match services from catalog
    items = _match_services(requirements, industry, budget, catalog_df)

    # Step 3 — Pricing
    discount_pct = _discount_percent(sum(i["total"] for i in items), priority)
    totals       = recalculate_totals(items, discount_pct)

    # Step 4 — Timeline
    tl_data  = _estimate_timeline(items, totals["total"])
    timeline = tl_data["label"]
    phases   = tl_data["phases"]

    # Step 5 — Lead score
    lead_score, score_reason = _score_lead(budget, totals["total"], requirements, priority)

    # Step 6 — Summary
    summary = _build_summary(lead, category, items, timeline, totals["total"])

    return {
        "customer_name"       : lead.get("customer_name", ""),
        "customer_email"      : lead.get("customer_email", ""),
        "company"             : lead.get("company", ""),
        "industry"            : industry,
        "quote_date"          : str(today),
        "valid_until"         : str(valid_until),
        "project_category"    : category,
        "lead_score"          : lead_score,
        "lead_score_reason"   : score_reason,
        "recommended_services": items,
        "items"               : items,
        "estimated_timeline"  : timeline,
        "timeline_breakdown"  : phases,
        "quote_summary"       : summary,
        "catalog_warnings"    : [],
        **totals,
    }
