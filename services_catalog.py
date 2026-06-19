import pandas as pd
from pathlib import Path

CATALOG_PATH = str(Path(__file__).parent / "data" / "services_catalog.csv")

REQUIRED_CATALOG_COLS = [
    "service_name", "category", "description",
    "unit_price", "unit", "min_qty", "max_qty",
]


def load_catalog(path: str = CATALOG_PATH) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and validate the services catalog CSV.
    Returns (dataframe, list_of_errors).
    """
    errors = []

    if not Path(path).exists():
        return pd.DataFrame(), [f"Catalog file not found: {path}"]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return pd.DataFrame(), [f"Could not read catalog: {e}"]

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_CATALOG_COLS if c not in df.columns]
    if missing:
        errors.append(f"Catalog missing columns: {', '.join(missing)}")
        return df, errors

    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
    df["min_qty"]    = pd.to_numeric(df["min_qty"],    errors="coerce").fillna(1)
    df["max_qty"]    = pd.to_numeric(df["max_qty"],    errors="coerce").fillna(9999)
    df["service_name"] = df["service_name"].astype(str).str.strip()
    df["category"]     = df["category"].astype(str).str.strip()

    return df.reset_index(drop=True), errors


def get_service_names(df: pd.DataFrame) -> list[str]:
    return sorted(df["service_name"].tolist())


def get_categories(df: pd.DataFrame) -> list[str]:
    return sorted(df["category"].unique().tolist())


def get_services_by_category(df: pd.DataFrame) -> dict[str, list[dict]]:
    result = {}
    for cat, group in df.groupby("category"):
        result[cat] = group.to_dict(orient="records")
    return result


def get_catalog_for_ai(df: pd.DataFrame) -> str:
    """
    Formats the catalog as a strict text block for the AI prompt.
    Each line is one service the AI is allowed to pick from.
    """
    lines = ["AVAILABLE SERVICES (you may ONLY recommend services from this list):"]
    lines.append("-" * 70)
    for _, row in df.iterrows():
        lines.append(
            f"  Service : {row['service_name']}\n"
            f"  Category: {row['category']}\n"
            f"  Desc    : {row['description']}\n"
            f"  Price   : ${row['unit_price']:,.0f} {row['unit']}\n"
            f"  Qty     : min {int(row['min_qty'])} — max {int(row['max_qty'])}\n"
        )
    lines.append("-" * 70)
    lines.append("Do NOT recommend any service not listed above.")
    return "\n".join(lines)


def validate_quote_services(quote: dict, df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """
    Cross-checks AI-recommended services against the catalog.
    Returns (validated_items, list_of_warnings).
    Items not found in the catalog are removed with a warning.
    Prices are snapped to catalog prices so AI cannot hallucinate costs.
    """
    catalog_lookup = {
        row["service_name"].lower(): row
        for _, row in df.iterrows()
    }

    validated = []
    warnings  = []

    services = quote.get("recommended_services") or quote.get("items") or []

    for item in services:
        name       = item.get("service", "")
        name_lower = name.lower()

        if name_lower not in catalog_lookup:
            warnings.append(f"Removed '{name}' — not found in services catalog.")
            continue

        catalog_row = catalog_lookup[name_lower]
        qty = max(
            int(catalog_row["min_qty"]),
            min(int(item.get("quantity", catalog_row["min_qty"])), int(catalog_row["max_qty"])),
        )
        unit_price = float(catalog_row["unit_price"])
        total      = round(qty * unit_price, 2)

        validated.append({
            "service"    : catalog_row["service_name"],
            "category"   : catalog_row["category"],
            "reason"     : item.get("reason") or item.get("description") or catalog_row["description"],
            "quantity"   : qty,
            "unit"       : catalog_row["unit"],
            "unit_price" : unit_price,
            "total"      : total,
        })

    return validated, warnings


def recalculate_totals(items: list[dict], discount_percent: float = 0) -> dict:
    subtotal        = sum(i["total"] for i in items)
    discount_amount = round(subtotal * discount_percent / 100, 2)
    total           = round(subtotal - discount_amount, 2)
    return {
        "subtotal"        : subtotal,
        "discount_percent": discount_percent,
        "discount_amount" : discount_amount,
        "total"           : total,
        "estimated_price" : total,
    }
