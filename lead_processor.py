import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = [
    "customer_name",
    "customer_email",
    "company",
    "industry",
    "budget",
    "requested_items",
    "priority",
]

VALID_PRIORITIES = {"High", "Medium", "Low"}
VALID_INDUSTRIES = {"Technology", "Healthcare", "Education", "Construction", "Retail", "Finance", "Other"}


def load_leads(file) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Accepts a file path or Streamlit UploadedFile.
    Returns (dataframe, errors, warnings).
    Errors block processing. Warnings are shown but do not block.
    """
    errors = []
    warnings = []

    # ── Read CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return pd.DataFrame(), [f"Could not read file: {e}"], []

    if df.empty:
        return pd.DataFrame(), ["The CSV file is empty."], []

    # ── Normalize column names ────────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Check required columns ────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        return df, errors, warnings

    # ── Drop rows missing critical fields ─────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["customer_name", "customer_email", "requested_items"])
    dropped = before - len(df)
    if dropped:
        warnings.append(f"{dropped} row(s) removed — missing name, email, or requirements.")

    if df.empty:
        errors.append("No valid leads found after removing incomplete rows.")
        return df, errors, warnings

    df = df.reset_index(drop=True)

    # ── Validate and clean each column ────────────────────────────────────────
    # Budget: convert to number
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    bad_budget = df["budget"].isna().sum()
    if bad_budget:
        warnings.append(f"{bad_budget} lead(s) have invalid budget values — defaulting to 0.")
    df["budget"] = df["budget"].fillna(0).astype(float)

    # Priority: normalize and validate
    df["priority"] = df["priority"].astype(str).str.strip().str.capitalize()
    bad_priority = ~df["priority"].isin(VALID_PRIORITIES)
    if bad_priority.any():
        df.loc[bad_priority, "priority"] = "Medium"
        warnings.append(f"{bad_priority.sum()} lead(s) had invalid priority — defaulted to Medium.")

    # Industry: normalize and validate
    df["industry"] = df["industry"].astype(str).str.strip().str.capitalize()
    bad_industry = ~df["industry"].isin(VALID_INDUSTRIES)
    if bad_industry.any():
        df.loc[bad_industry, "industry"] = "Other"
        warnings.append(f"{bad_industry.sum()} lead(s) had unrecognised industry — set to Other.")

    # Email: basic format check
    invalid_email = ~df["customer_email"].astype(str).str.contains(r"@.*\.", regex=True)
    if invalid_email.any():
        warnings.append(f"{invalid_email.sum()} lead(s) have potentially invalid email addresses.")

    # Status: default to New if not present
    if "status" not in df.columns:
        df["status"] = "New"
    else:
        df["status"] = df["status"].fillna("New")

    # Lead score: default to unscored
    if "lead_score" not in df.columns:
        df["lead_score"] = None

    # Quote total: default to unquoted
    if "quote_total" not in df.columns:
        df["quote_total"] = None

    return df, errors, warnings


def load_services(path: str = str(Path(__file__).parent / "data" / "services_catalog.csv")) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def get_lead_by_index(df: pd.DataFrame, idx: int) -> dict:
    return df.iloc[idx].to_dict()


def get_validation_summary(df: pd.DataFrame) -> dict:
    """Returns stats about the loaded leads for display in the UI."""
    return {
        "total": len(df),
        "new": int((df["status"] == "New").sum()),
        "quoted": int((df["status"] == "Quote Sent").sum()),
        "high_priority": int((df["priority"] == "High").sum()),
        "avg_budget": float(df["budget"].mean()),
        "max_budget": float(df["budget"].max()),
    }


def mark_lead_done(df: pd.DataFrame, idx: int, quote_total: float) -> pd.DataFrame:
    df = df.copy()
    df.at[idx, "status"] = "Quote Sent"
    df.at[idx, "quote_total"] = quote_total
    return df


def export_leads(df: pd.DataFrame, path: str = "output/leads_updated.csv") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
