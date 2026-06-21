"""
Synthetic training data generator for the category classifier.

Generates ~200 labeled (requirements_text, category) pairs by combining
seed phrases, industry contexts, and random variation.

Run this script directly:
    python ml_engine/generate_training_data.py

Output: data/training_data.csv
Columns: text, category

Use this to:
  1. Train a TF-IDF + SVM classifier as a comparison baseline.
  2. Fine-tune the sentence transformer (optional, advanced).
  3. Show a full ML training pipeline in your viva.
"""

import csv
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

# ── SEED PHRASES PER CATEGORY ─────────────────────────────────────────────────
SEEDS = {
    "CRM & Sales System": [
        "We need a CRM system to track our leads and customers.",
        "Build a sales pipeline management tool with deal tracking.",
        "Customer relationship management with email integration and reporting.",
        "We want to manage our sales team performance and customer data.",
        "Lead tracking and opportunity management platform for our sales team.",
        "CRM with contact management, follow-up reminders, and analytics.",
        "Sales automation tool with pipeline stages and customer history.",
        "We need to track customer interactions and sales conversions.",
    ],
    "Healthcare Portal": [
        "Patient portal with appointment booking and medical records.",
        "Hospital management system with doctor scheduling and patient history.",
        "We need a telemedicine platform for online doctor consultations.",
        "Clinic management software with billing and prescription management.",
        "Healthcare portal with lab reports, appointments, and patient login.",
        "Online doctor booking and patient health record management system.",
        "We want a hospital management platform with ward and staff management.",
        "Electronic health records system for our medical clinic.",
    ],
    "E-Commerce Platform": [
        "Online store with product catalog, cart, and payment gateway.",
        "E-commerce website with inventory management and order tracking.",
        "We need a shopping platform with multiple payment options.",
        "Build a retail website where customers can buy our products online.",
        "Marketplace platform with vendor management and payment processing.",
        "E-commerce site with product reviews, wishlist, and discount coupons.",
        "We want an online shop with delivery tracking and customer accounts.",
        "Multi-vendor e-commerce platform with analytics and inventory.",
    ],
    "Education Platform": [
        "Learning management system with course creation and student tracking.",
        "Online education platform with video lessons and quizzes.",
        "School management software with attendance, grades, and parent portal.",
        "E-learning platform with certificates and progress tracking.",
        "We need an LMS for training employees with assessments and reports.",
        "Online course platform with instructor dashboards and student analytics.",
        "Education portal with live classes, assignments, and discussion forums.",
        "Student management system with fee tracking and academic records.",
    ],
    "Project Management Tool": [
        "Project management tool with task tracking and team collaboration.",
        "We need construction site management with milestone tracking.",
        "Build a project management platform with Gantt charts and deadlines.",
        "Team task management with project timelines and progress reports.",
        "We want a tool to manage multiple projects and team workloads.",
        "Construction management platform with site inspection and reporting.",
        "Project tracking software with resource allocation and time tracking.",
        "Task management system with sprint planning and kanban board.",
    ],
    "Cloud Infrastructure": [
        "We need cloud hosting and server setup for our application.",
        "Migrate our on-premise servers to AWS or Azure cloud infrastructure.",
        "Set up cloud deployment with load balancing and auto-scaling.",
        "DevOps and CI/CD pipeline with containerisation using Docker.",
        "We need VPS hosting with monitoring and backup solutions.",
        "Cloud infrastructure setup with database hosting and CDN.",
        "Server configuration and cloud migration for our web platform.",
        "We want cloud hosting with 99.9% uptime and disaster recovery.",
    ],
    "Cybersecurity & Compliance": [
        "Security audit and vulnerability assessment for our IT systems.",
        "We need penetration testing and network security review.",
        "Firewall setup and cybersecurity implementation for our company.",
        "GDPR compliance audit and data protection policy implementation.",
        "We require a security assessment and risk management report.",
        "Network security hardening and intrusion detection system setup.",
        "Cybersecurity training and phishing simulation for our staff.",
        "We need SOC 2 compliance audit and security controls implementation.",
    ],
    "AI & Automation": [
        "We want to automate our business workflows using AI and ML.",
        "Build a chatbot for customer support using NLP.",
        "AI-powered document processing and data extraction automation.",
        "Machine learning model to predict customer churn for our business.",
        "We need AI integration to automate invoice processing and data entry.",
        "Workflow automation using RPA and AI for our operations team.",
        "NLP-based text classification for customer feedback analysis.",
        "Build an AI recommendation engine for our e-commerce platform.",
    ],
    "Business Intelligence": [
        "Business intelligence dashboard with sales and revenue analytics.",
        "We need real-time reporting and KPI tracking for management.",
        "Data visualisation platform with charts and interactive reports.",
        "Build a BI tool to analyse customer behaviour and business metrics.",
        "We want an analytics dashboard connected to our database.",
        "Power BI or Tableau-like reporting platform for our business data.",
        "Data warehouse setup with ETL pipelines and dashboard reporting.",
        "Executive dashboard with financial reports and performance metrics.",
    ],
    "Mobile Application": [
        "We need a mobile app for Android and iOS users.",
        "Build a cross-platform mobile application using Flutter or React Native.",
        "Mobile app with push notifications, user login, and offline mode.",
        "We want a food delivery app with real-time tracking.",
        "iOS and Android app for our retail customers with loyalty programme.",
        "Mobile application with GPS tracking and in-app payments.",
        "We need a mobile app for field staff with offline data sync.",
        "Cross-platform mobile app with social login and user profiles.",
    ],
    "Web Application": [
        "We need a custom web application for internal use by our team.",
        "Build a web portal with user authentication and admin dashboard.",
        "Full-stack web application with database and REST API.",
        "We want a web-based platform to manage our business operations.",
        "Custom web system with role-based access and reporting module.",
        "Web portal for customer self-service and account management.",
        "Build a multi-tenant web application for our SaaS product.",
        "Internal web tool to manage inventory and employee records.",
    ],
    "Office & Productivity": [
        "We need Microsoft 365 setup and email migration for our team.",
        "Set up Microsoft Teams, SharePoint, and OneDrive for collaboration.",
        "Office 365 deployment with Outlook and Teams configuration.",
        "We want to migrate from on-premise email to cloud Office 365.",
        "Productivity tools setup including email, calendar, and file storage.",
        "Microsoft 365 administration and user management for our company.",
        "We need SharePoint intranet and document management system.",
        "Office 365 setup with Teams, OneDrive, and security policies.",
    ],
    "IT Support & Maintenance": [
        "We need ongoing IT support and managed helpdesk services.",
        "Managed IT services with server monitoring and maintenance.",
        "We want a support contract for network and infrastructure maintenance.",
        "IT helpdesk with remote support and on-site technical assistance.",
        "Server maintenance and backup management for our organization.",
        "We need IT infrastructure monitoring and proactive support.",
        "Managed services for network, servers, and end-user support.",
        "IT support contract with SLA and 24/7 monitoring.",
    ],
}

AUGMENTATIONS = [
    "for our company",
    "for our team",
    "for our business",
    "as soon as possible",
    "with a modern UI",
    "with detailed reporting",
    "with API integration",
    "that is scalable",
    "that is secure",
    "with mobile responsiveness",
    "with user authentication",
    "with an admin panel",
]


def generate(output_path: str = None, samples_per_category: int = 15) -> str:
    if output_path is None:
        output_path = str(Path(__file__).parent.parent / "data" / "training_data.csv")

    rows = []
    for category, phrases in SEEDS.items():
        base = list(phrases)
        while len(base) < samples_per_category:
            phrase = random.choice(phrases)
            aug = random.choice(AUGMENTATIONS)
            base.append(f"{phrase} {aug}")

        for text in base[:samples_per_category]:
            rows.append({"text": text, "category": category})

    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} training samples → {output_path}")
    return output_path


if __name__ == "__main__":
    generate()
