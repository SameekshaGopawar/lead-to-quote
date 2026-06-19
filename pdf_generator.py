from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_RIGHT, TA_CENTER, TA_LEFT

BRAND_DARK   = colors.HexColor("#1e1b4b")
BRAND_MID    = colors.HexColor("#4338ca")
BRAND_LIGHT  = colors.HexColor("#e0e7ff")
GREY_TEXT    = colors.HexColor("#6b7280")
BLACK        = colors.HexColor("#111827")
WHITE        = colors.white
GREEN        = colors.HexColor("#16a34a")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", fontSize=22, textColor=WHITE,
                                fontName="Helvetica-Bold", spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontSize=10, textColor=WHITE,
                                   fontName="Helvetica", spaceAfter=2),
        "label": ParagraphStyle("label", fontSize=8, textColor=GREY_TEXT,
                                fontName="Helvetica", spaceAfter=2),
        "value": ParagraphStyle("value", fontSize=11, textColor=BLACK,
                                fontName="Helvetica-Bold", spaceAfter=2),
        "small": ParagraphStyle("small", fontSize=9, textColor=GREY_TEXT,
                                fontName="Helvetica"),
        "body": ParagraphStyle("body", fontSize=10, textColor=BLACK,
                               fontName="Helvetica", leading=14),
        "section": ParagraphStyle("section", fontSize=12, textColor=BRAND_DARK,
                                  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=8),
        "total_label": ParagraphStyle("total_label", fontSize=12, textColor=BLACK,
                                      fontName="Helvetica-Bold", alignment=TA_RIGHT),
        "total_value": ParagraphStyle("total_value", fontSize=14, textColor=BRAND_DARK,
                                      fontName="Helvetica-Bold", alignment=TA_RIGHT),
        "right": ParagraphStyle("right", fontSize=9, textColor=BLACK,
                                fontName="Helvetica", alignment=TA_RIGHT),
    }


def generate_pdf(quote: dict) -> bytes:
    """Returns PDF as bytes — no file system required."""
    from io import BytesIO
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    s = _styles()
    story = []

    # ── HEADER BANNER ──────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("SERVICE QUOTE", s["title"]),
        Paragraph(f"Prepared for {quote['company']}", s["subtitle"]),
    ]]
    header_table = Table(header_data, colWidths=[doc.width * 0.6, doc.width * 0.4])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BRAND_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8 * mm))

    # ── META ROW ───────────────────────────────────────────────────────────────
    meta_data = [[
        Paragraph("PREPARED FOR", s["label"]),
        Paragraph("COMPANY", s["label"]),
        Paragraph("QUOTE DATE", s["label"]),
        Paragraph("VALID UNTIL", s["label"]),
        Paragraph("LEAD SCORE", s["label"]),
    ], [
        Paragraph(quote["customer_name"], s["value"]),
        Paragraph(quote["company"], s["value"]),
        Paragraph(quote["quote_date"], s["value"]),
        Paragraph(quote["valid_until"], s["value"]),
        Paragraph(f"{quote['lead_score']} / 10", s["value"]),
    ]]
    meta_table = Table(meta_data, colWidths=[doc.width / 5] * 5)
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BRAND_LIGHT),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BRAND_LIGHT]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6 * mm))

    # ── SERVICES TABLE ─────────────────────────────────────────────────────────
    story.append(Paragraph("Services Included", s["section"]))

    col_w = [doc.width * r for r in [0.30, 0.30, 0.15, 0.12, 0.13]]
    headers = ["Service", "Description", "Qty / Unit", "Unit Price", "Total"]
    rows = [headers]
    for item in quote["items"]:
        desc = item.get("reason") or item.get("description") or ""
        rows.append([
            Paragraph(item["service"], s["body"]),
            Paragraph(desc, s["small"]),
            Paragraph(f"{item['quantity']} {item['unit']}", s["small"]),
            Paragraph(f"${item['unit_price']:,.0f}", s["right"]),
            Paragraph(f"${item['total']:,.0f}", s["right"]),
        ])

    svc_table = Table(rows, colWidths=col_w, repeatRows=1)
    svc_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BRAND_MID),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("TOPPADDING",    (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#f9fafb")]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
        ("TOPPADDING",    (0, 1), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(svc_table)
    story.append(Spacer(1, 5 * mm))

    # ── TOTALS ─────────────────────────────────────────────────────────────────
    totals_rows = [
        ["Subtotal", f"${quote['subtotal']:,.0f}"],
    ]
    if quote.get("discount_percent", 0) > 0:
        totals_rows.append([
            f"Discount ({quote['discount_percent']}%)",
            f"-${quote['discount_amount']:,.0f}",
        ])
    totals_rows.append(["TOTAL", f"${quote['total']:,.0f}"])

    totals_table = Table(
        totals_rows,
        colWidths=[doc.width * 0.75, doc.width * 0.25],
    )
    totals_style = [
        ("ALIGN",         (0, 0), (-1, -1), "RIGHT"),
        ("FONTNAME",      (0, 0), (-1, -2), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -2), 10),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, -1), (-1, -1), 13),
        ("TEXTCOLOR",     (0, -1), (-1, -1), BRAND_DARK),
        ("LINEABOVE",     (0, -1), (-1, -1), 1.2, BRAND_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if quote.get("discount_percent", 0) > 0:
        totals_style.append(("TEXTCOLOR", (0, 1), (-1, 1), GREEN))
    totals_table.setStyle(TableStyle(totals_style))
    story.append(totals_table)
    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 4 * mm))

    # ── NOTES + TIMELINE ───────────────────────────────────────────────────────
    note     = quote.get("quote_summary") or quote.get("notes") or ""
    timeline = quote.get("estimated_timeline") or quote.get("timeline") or "To be confirmed"
    notes_data = [[
        Paragraph(f"<b>Personal Note</b><br/>{note}", s["body"]),
        Paragraph(f"<b>Estimated Timeline</b><br/>{timeline}", s["body"]),
    ]]
    notes_table = Table(notes_data, colWidths=[doc.width * 0.6, doc.width * 0.4])
    notes_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), colors.HexColor("#f5f3ff")),
        ("BACKGROUND",    (1, 0), (1, 0), colors.HexColor("#f0fdf4")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(notes_table)
    story.append(Spacer(1, 6 * mm))

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    footer_data = [[
        Paragraph("This quote was auto-generated by the Lead-to-Quote system.", s["small"]),
        Paragraph(f"Contact: {quote['customer_email']}", s["small"]),
    ]]
    footer_table = Table(footer_data, colWidths=[doc.width * 0.6, doc.width * 0.4])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f3f4f6")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(footer_table)

    doc.build(story)
    return buffer.getvalue()
