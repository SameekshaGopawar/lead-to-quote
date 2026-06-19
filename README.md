# Lead-to-Quote Module

🔗 **Live Demo:** https://lead-to-quote-i7ccr6yos93jkvbhthmwas.streamlit.app/

🔗 **GitHub Repository:** https://github.com/SameekshaGopawar/lead-to-quote

An autonomous lead-to-quote generation system built with Python and Streamlit. Upload a CSV of leads, and the system automatically categorises the project, recommends services from a catalog, estimates timeline and pricing, and generates a downloadable PDF quote — all offline, no API keys required.

## Resume Highlights

- Built a fully autonomous Lead-to-Quote web application using Python, Streamlit, and ReportLab that ingests CSV leads, categorises projects via a local rule-based AI engine, and generates professional PDF quotes with itemised pricing and timeline breakdowns — deployed live on Streamlit Community Cloud
- Engineered a local quote generation engine with zero external API dependencies, implementing keyword-based project classification, 15-rule service matching against a dynamic catalog, automated discount logic, and professional summary generation using Python template rendering
- Designed and deployed a multi-tab Streamlit interface with CSV validation, filterable lead tables, real-time quote preview, and in-browser PDF download — integrating pandas for data processing and ReportLab for document generation across a modular 6-file Python architecture

## Features

- Upload leads via CSV
- Automatic field validation
- Browse a services catalog with pricing
- Local AI engine categorises projects and recommends services
- Itemised pricing with automatic discounts
- Timeline estimation with phase breakdown
- Professional PDF quote generation
- Download PDF instantly in the browser

## Tech Stack

| Library | Purpose |
|---------|---------|
| Streamlit | Web UI |
| Pandas | CSV parsing and data handling |
| ReportLab | PDF generation |
| Python-dotenv | Environment variable management |

## Project Structure

```
Lead_To_Quote_Project/
├── app.py                  # Streamlit UI
├── ai_quote_generator.py   # Local quote engine
├── lead_processor.py       # CSV ingestion and validation
├── services_catalog.py     # Catalog loader and service matcher
├── pdf_generator.py        # ReportLab PDF builder
├── requirements.txt
├── data/
│   ├── leads.csv           # Sample leads
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

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repository and set **Main file path** to `app.py`
5. Click **Deploy**

## How the Quote Engine Works

1. **Categorise** — keyword matching maps requirements to a project type
2. **Match services** — 15 service rules scan requirements against the catalog
3. **Price** — pulled directly from `services_catalog.csv`
4. **Timeline** — rule-based: short / medium / long based on scope
5. **Discount** — automatic: 5% (≥$5k), 7% (≥$10k), 10% (≥$20k)
6. **Summary** — professional template filled with customer and project details
