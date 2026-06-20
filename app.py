import streamlit as st
import pandas as pd
from pathlib import Path

from lead_processor import (
    load_leads, get_lead_by_index,
    get_validation_summary, mark_lead_done, export_leads,
)
from ai_quote_generator import generate_quote
from pdf_generator import generate_pdf
from services_catalog import load_catalog, get_categories, get_services_by_category

st.set_page_config(
    page_title="Lead to Quote",
    page_icon="📋",
    layout="wide",
)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
for key, default in [
    ("leads_df", None),
    ("quote", None),
    ("pdf_bytes", None),
    ("pdf_filename", None),
    ("pdf_key", None),
    ("selected_idx", 0),
    ("quote_idx", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Lead to Quote")
    st.caption("Upload your leads CSV to begin.")

    uploaded = st.file_uploader("Upload leads.csv", type=["csv"])
    if uploaded:
        df, errors, warnings = load_leads(uploaded)
        for e in errors:
            st.error(f"Error: {e}")
        for w in warnings:
            st.warning(w)
        if not errors and not df.empty:
            st.session_state.leads_df = df
            st.session_state.quote = None
            st.session_state.pdf_path = None
            st.success(f"{len(df)} leads loaded.")

    st.divider()
    st.markdown("**Required CSV columns**")
    st.code(
        "customer_name\ncustomer_email\ncompany\nindustry\nbudget\nrequested_items\npriority",
        language="text",
    )
    sample_path = Path(__file__).parent / "data" / "leads.csv"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            st.download_button(
                "Download sample leads.csv", data=f,
                file_name="leads_sample.csv", mime="text/csv",
                use_container_width=True,
            )

# ── GUARD ─────────────────────────────────────────────────────────────────────
df = st.session_state.leads_df

st.title("Lead to Quote")

if df is None:
    st.info("Upload a leads CSV using the sidebar to get started.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Upload**\nDrop your leads CSV. Every field is validated automatically.")
    c2.markdown("**2. Generate**\nAI analyses requirements and builds an itemised quote from your catalog.")
    c3.markdown("**3. Download**\nGet a professional branded PDF quote in one click.")
    st.stop()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_leads, tab_catalog, tab_quote = st.tabs(["Leads", "Services Catalog", "Generate Quote"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — LEADS
# ════════════════════════════════════════════════════════════════════════════════
with tab_leads:
    summary = get_validation_summary(df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Leads",   summary["total"])
    m2.metric("New",           summary["new"])
    m3.metric("Quoted",        summary["quoted"])
    m4.metric("High Priority", summary["high_priority"])
    m5.metric("Avg Budget",    f"${summary['avg_budget']:,.0f}")

    st.divider()
    st.subheader("All Leads")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_priority = st.selectbox("Priority", ["All", "High", "Medium", "Low"])
    with col_f2:
        filter_status = st.selectbox("Status", ["All", "New", "Quote Sent"])
    with col_f3:
        filter_industry = st.selectbox(
            "Industry", ["All"] + sorted(df["industry"].unique().tolist())
        )

    filtered = df.copy()
    if filter_priority != "All":
        filtered = filtered[filtered["priority"] == filter_priority]
    if filter_status != "All":
        filtered = filtered[filtered["status"] == filter_status]
    if filter_industry != "All":
        filtered = filtered[filtered["industry"] == filter_industry]

    display_cols = ["customer_name", "company", "industry", "budget", "priority", "status", "quote_total"]
    show_cols    = [c for c in display_cols if c in filtered.columns]

    def highlight_status(val):
        if val == "Quote Sent": return "background-color: #052e16; color: #34d399;"
        if val == "New":        return "background-color: #1e3a5f; color: #60a5fa;"
        return ""

    def highlight_priority(val):
        if val == "High":   return "color: #f87171; font-weight: bold;"
        if val == "Medium": return "color: #fbbf24;"
        return "color: #9ca3af;"

    styled = (
        filtered[show_cols]
        .style
        .map(highlight_status, subset=["status"])
        .map(highlight_priority, subset=["priority"])
        .format({
            "budget"     : "${:,.0f}",
            "quote_total": lambda x: f"${x:,.0f}" if pd.notna(x) else "—",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=False)
    st.caption(f"Showing {len(filtered)} of {len(df)} leads")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — SERVICES CATALOG
# ════════════════════════════════════════════════════════════════════════════════
with tab_catalog:
    st.subheader("Services Catalog")
    st.caption("The AI may only recommend services from this catalog. Prices are enforced — the AI cannot change them.")

    cat_df, cat_errors = load_catalog()

    if cat_errors:
        for e in cat_errors:
            st.error(e)
    else:
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Total Services", len(cat_df))
        cm2.metric("Categories",     cat_df["category"].nunique())
        cm3.metric("Lowest Price",   f"${cat_df['unit_price'].min():,.0f}")
        cm4.metric("Highest Price",  f"${cat_df['unit_price'].max():,.0f}")

        st.divider()

        categories   = ["All"] + get_categories(cat_df)
        selected_cat = st.selectbox("Filter by category", categories)
        filtered_cat = cat_df if selected_cat == "All" else cat_df[cat_df["category"] == selected_cat]

        display_cat_cols = ["service_name", "category", "description", "unit_price", "unit", "min_qty", "max_qty"]
        st.dataframe(
            filtered_cat[display_cat_cols].style.format({
                "unit_price": "${:,.0f}",
                "min_qty"   : "{:.0f}",
                "max_qty"   : "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.markdown("#### Browse by Category")
        for cat_name, services in get_services_by_category(cat_df).items():
            with st.expander(f"{cat_name}  —  {len(services)} services"):
                for svc in services:
                    c1, c2, c3 = st.columns([3, 4, 2])
                    c1.markdown(f"**{svc['service_name']}**")
                    c2.markdown(f"<small>{svc['description']}</small>", unsafe_allow_html=True)
                    c3.markdown(f"**${svc['unit_price']:,.0f}** / {svc['unit']}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — GENERATE QUOTE
# ════════════════════════════════════════════════════════════════════════════════
with tab_quote:
    st.subheader("Generate Quote")

    lead_options = [
        f"{i}  —  {row['customer_name']}  ({row['company']})"
        for i, row in df.iterrows()
    ]
    st.markdown("""
    <style>
    div[data-testid="stSelectbox"] > div:first-child {
        max-width: 420px;
        border: 1.5px solid #a78bfa;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    selected_label = st.selectbox("Select a lead", lead_options)
    selected_idx   = int(selected_label.split("  —  ")[0])
    st.session_state.selected_idx = selected_idx
    lead = get_lead_by_index(df, selected_idx)

    # Lead detail card
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Name:** {lead['customer_name']}")
            st.markdown(f"**Email:** {lead['customer_email']}")
            st.markdown(f"**Company:** {lead['company']}")
            st.markdown(f"**Industry:** {lead['industry']}")
        with c2:
            st.markdown(f"**Budget:** ${float(lead['budget']):,.0f}")
            st.markdown(f"**Priority:** {lead['priority']}")
            st.markdown(f"**Status:** {lead['status']}")
        st.markdown("**Requirements**")
        st.info(lead["requested_items"])

    if st.button("Generate Quote with AI", type="primary", use_container_width=True):
        with st.spinner("Analysing requirements and matching catalog services..."):
            try:
                quote = generate_quote(lead)
                st.session_state.quote      = quote
                st.session_state.quote_idx  = selected_idx
                st.session_state.pdf_path   = None
            except Exception as e:
                st.error(f"Error: {e}")

    # ── QUOTE OUTPUT ──────────────────────────────────────────────────────────
    quote = st.session_state.quote
    if quote and st.session_state.get("quote_idx") == selected_idx:
        st.divider()

        # OUTPUT 1: Project Category
        st.markdown("#### Project Category")
        st.info(f"**{quote['project_category']}**")

        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Lead Score",      f"{quote['lead_score']} / 10")
        q2.metric("Estimated Price", f"${quote['estimated_price']:,.0f}")
        q3.metric("Subtotal",        f"${quote['subtotal']:,.0f}")
        q4.metric("Discount",        f"{quote.get('discount_percent', 0)}%")
        q5.metric("Total",           f"${quote['total']:,.0f}")
        st.caption(f"Score reason: {quote.get('lead_score_reason', '')}")

        st.divider()

        # OUTPUT 2: Recommended Services (catalog-verified)
        st.markdown("#### Recommended Services")
        st.caption("All services below are verified against the catalog. Prices are catalog prices.")
        services = quote.get("recommended_services", quote.get("items", []))
        if services:
            svc_df = pd.DataFrame(services)
            cols   = [c for c in ["service", "category", "reason", "quantity", "unit", "unit_price", "total"]
                      if c in svc_df.columns]
            fmt = {}
            if "unit_price" in svc_df.columns: fmt["unit_price"] = "${:,.0f}"
            if "total"      in svc_df.columns: fmt["total"]      = "${:,.0f}"
            st.dataframe(svc_df[cols].style.format(fmt), use_container_width=True, hide_index=True)

        st.divider()

        # OUTPUT 3: Estimated Timeline
        st.markdown("#### Estimated Timeline")
        st.success(f"Total: **{quote['estimated_timeline']}**")
        for i, phase in enumerate(quote.get("timeline_breakdown", []), 1):
            with st.expander(f"Phase {i}: {phase.get('phase', '')}  —  {phase.get('duration', '')}"):
                st.write(phase.get("description", ""))

        st.divider()

        # OUTPUT 4: Estimated Price
        st.markdown("#### Estimated Price")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Subtotal", f"${quote['subtotal']:,.0f}")
        pc2.metric(
            f"Discount ({quote.get('discount_percent', 0)}%)",
            f"-${quote.get('discount_amount', 0):,.0f}",
        )
        pc3.metric(
            "Final Price", f"${quote['total']:,.0f}",
            delta=f"-${quote.get('discount_amount', 0):,.0f} saved" if quote.get("discount_amount") else None,
        )

        st.divider()

        # OUTPUT 5: Professional Quote Summary
        st.markdown("#### Professional Quote Summary")
        with st.container(border=True):
            st.markdown(quote.get("quote_summary", ""))

        st.divider()

        # Generate PDF bytes once and cache in session state keyed to quote
        quote_key = f"pdf_{quote['customer_name']}_{quote['quote_date']}"
        if st.session_state.get("pdf_key") != quote_key:
            try:
                st.session_state.pdf_bytes    = generate_pdf(quote)
                st.session_state.pdf_filename = f"Quote_{quote['customer_name'].replace(' ', '_')}.pdf"
                st.session_state.pdf_key      = quote_key
            except Exception as e:
                import traceback
                st.error(f"PDF generation failed: {e}")
                st.code(traceback.format_exc())

        if st.session_state.get("pdf_bytes"):
            st.download_button(
                label="Download PDF Quote",
                data=st.session_state.pdf_bytes,
                file_name=st.session_state.pdf_filename,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
