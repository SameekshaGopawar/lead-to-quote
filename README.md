---
title: Lead To Quote
emoji: 📋
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Lead-to-Quote Module

🔗 **GitHub Repository:** https://github.com/SameekshaGopawar/lead-to-quote

An ML-powered lead-to-quote generation system built with Python and Streamlit. Upload a CSV of leads, and the system uses Sentence Transformers (all-MiniLM-L6-v2) to classify the project category, recommend services from a catalog via semantic similarity, estimate timeline and pricing, and generate a downloadable PDF quote.

## Resume Highlights

- Built an ML-powered Lead-to-Quote web application using Python, Streamlit, Sentence Transformers, and ReportLab — classifies project categories and recommends services using all-MiniLM-L6-v2 semantic embeddings and cosine similarity, deployed on HuggingFace Spaces
- Engineered a two-model NLP pipeline replacing rule-based keyword matching with Sentence Transformer semantic similarity for project classification and service recommendation, with TF-IDF + SVM as a trained comparison baseline
- Designed and deployed a multi-tab Streamlit interface with CSV validation, filterable lead tables, ML confidence scores, real-time quote preview, and in-browser PDF download across a modular Python architecture

## Features

- Upload leads via CSV
- Automatic field validation
- Browse a services catalog with pricing
- ML engine classifies project category with confidence score
- Semantic service recommendation using NLP
- Itemised pricing with automatic discounts
- Timeline estimation with phase breakdown
- Professional PDF quote generation
- Download PDF instantly in the browser

## Tech Stack

| Library | Purpose |
|---------|---------|
| Streamlit | Web UI |
| Sentence Transformers | NLP embeddings for classification and recommendation |
| Transformers | HuggingFace model support |
| PyTorch | Deep learning backend |
| Scikit-learn | TF-IDF + SVM baseline classifier |
| Pandas | CSV parsing and data handling |
| ReportLab | PDF generation |

## Project Structure

```
Lead_To_Quote_Project/
├── app.py                          # Streamlit UI
├── ml_quote_engine.py              # ML-powered quote engine
├── lead_processor.py               # CSV ingestion and validation
├── services_catalog.py             # Catalog loader and price enforcer
├── pdf_generator.py                # ReportLab PDF builder
├── requirements.txt
├── ml_engine/
│   ├── text_cleaner.py             # Text preprocessing
│   ├── project_classifier.py       # Sentence Transformer category classifier
│   ├── service_matcher.py          # Semantic service recommendation
│   ├── training_data_generator.py  # Synthetic training data generator
│   └── train_model.py              # TF-IDF + SVM baseline training script
├── data/
│   ├── leads.csv                   # Sample leads
│   └── services_catalog.csv
```

## CSV Format

Your leads CSV must contain these columns:

| Column | Description |
|--------|-------------|
| customer_name | Full name of the contact |
| customer_email | Business email |
| company | Company name |
| industry | Technology / Healthcare / Education / Construction / Retail / Finance / Other |
| budget | Numeric budget in dollars |
| requested_items | Free text description of requirements |
| priority | High / Medium / Low |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How the ML Engine Works

1. **Preprocess** — clean and normalise requirements text
2. **Classify** — all-MiniLM-L6-v2 encodes text → cosine similarity against category prototypes
3. **Recommend** — semantic similarity between requirements and catalog service descriptions
4. **Price** — pulled directly from `services_catalog.csv` (catalog enforces all prices)
5. **Timeline** — rule-based: short / medium / long based on scope and price
6. **Discount** — automatic: 5% (≥$5k), 7% (≥$10k), 10% (≥$20k)
7. **Summary** — professional template filled with customer and project details
