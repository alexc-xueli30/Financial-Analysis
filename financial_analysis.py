import os
import sys
import time
import tempfile
from datetime import date, datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

_NAVY       = colors.HexColor("#1B2A4A")
_BLUE       = colors.HexColor("#2E6DA4")
_LIGHT_BG   = colors.HexColor("#F5F7FA")
_ALT_ROW    = colors.HexColor("#EEF2F7")
_RED        = colors.HexColor("#C0392B")
_RED_BG     = colors.HexColor("#FDEDEC")
_AMBER      = colors.HexColor("#D68910")
_AMBER_BG   = colors.HexColor("#FEF9E7")
_GREEN      = colors.HexColor("#1E8449")
_GREEN_BG   = colors.HexColor("#EAFAF1")
_BORDER     = colors.HexColor("#D5DCE8")
_MUTED      = colors.HexColor("#7F8C8D")
_DARK       = colors.HexColor("#2C3E50")

_PAGE_W, _PAGE_H = letter          # 612 × 792 pts
_MARGIN          = 48              # 0.67 inch
_CW              = _PAGE_W - 2 * _MARGIN   # content width: 516 pts


_SEC_CONTACT = os.environ.get("SEC_CONTACT", "sec-analyzer contact@example.com")
HEADERS = {"User-Agent": _SEC_CONTACT}
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_REVENUE_CONCEPTS = [
    "RevenuesNetOfInterestExpense",                         # investment banks (GS, MS) — least precise, set first so standard tags override
    "SalesRevenueNet",                                      # pre-2018 tag
    "Revenues",                                             # general tag (overwrites banking fallback where both exist)
    "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606 (2018+)
]

# Equity: broad → precise so the precise tag wins when both are present.
# StockholdersEquityIncluding... covers companies like Visa post-2011 that
# switched away from the plain StockholdersEquity tag.
_EQUITY_CONCEPTS = [
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "StockholdersEquity",
]

# Net income: ProfitLoss is broader (includes minority interests); NetIncomeLoss
# is the standard tag and overrides when present.
_NET_INCOME_CONCEPTS = [
    "ProfitLoss",
    "NetIncomeLoss",
]


class TickerNotFoundError(Exception):
    """Ticker symbol not found in SEC's company database."""

class SECDataError(Exception):
    """SEC API returned an error or unreadable response."""

class InsufficientDataError(Exception):
    """Not enough overlapping annual filings to compute ratios."""


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

_ticker_data_cache: dict | None = None


def _get_ticker_data() -> dict:
    global _ticker_data_cache
    if _ticker_data_cache is None:
        try:
            resp = requests.get(_SEC_TICKERS_URL, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SECDataError(f"Could not fetch SEC ticker list: {exc}") from exc
        _ticker_data_cache = resp.json()
    return _ticker_data_cache


def get_cik(ticker: str) -> str:
    """Return zero-padded 10-digit CIK for *ticker*, or raise TickerNotFoundError."""
    ticker_upper = ticker.upper().strip()
    for entry in _get_ticker_data().values():
        if entry["ticker"] == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    raise TickerNotFoundError(
        f"'{ticker}' not found in SEC's company database. "
        "Verify the ticker is correct and the company files with the SEC."
    )


def resolve_company(query: str) -> tuple[str, str]:
    """
    Resolve a ticker symbol or company name to (cik, ticker).
    Tries exact ticker match first, then case-insensitive company name substring.
    Returns (cik, resolved_ticker). Raises TickerNotFoundError if ambiguous or missing.
    """
    query_stripped = query.strip()
    query_upper = query_stripped.upper()
    query_lower = query_stripped.lower()
    data = _get_ticker_data()

    for entry in data.values():
        if entry["ticker"] == query_upper:
            return str(entry["cik_str"]).zfill(10), entry["ticker"]

    matches = [
        entry for entry in data.values()
        if query_lower in entry.get("title", "").lower()
    ]

    if not matches:
        raise TickerNotFoundError(
            f"'{query}' not found — try a ticker symbol (e.g. AAPL) or the full company name."
        )

    if len(matches) == 1:
        e = matches[0]
        return str(e["cik_str"]).zfill(10), e["ticker"]

    starts = [e for e in matches if e.get("title", "").lower().startswith(query_lower)]
    if len(starts) == 1:
        e = starts[0]
        return str(e["cik_str"]).zfill(10), e["ticker"]

    top = starts if starts else matches
    suggestions = " | ".join(f"{e['ticker']} ({e.get('title', '')})" for e in top[:5])
    raise TickerNotFoundError(
        f"Multiple companies match '{query}' — use a ticker symbol. "
        f"Suggestions: {suggestions}"
    )


def get_company_facts(cik: str) -> dict:
    """Fetch XBRL company facts for *cik*, or raise SECDataError."""
    url = _COMPANYFACTS_URL.format(cik=cik)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        if resp.status_code == 404:
            raise SECDataError(
                f"No EDGAR XBRL data found for CIK {cik}. "
                "The company may not file XBRL-tagged financials."
            ) from exc
        raise SECDataError(f"SEC API returned HTTP {resp.status_code}: {exc}") from exc
    except requests.RequestException as exc:
        raise SECDataError(f"Network error fetching company facts: {exc}") from exc

    try:
        return resp.json()
    except ValueError as exc:
        raise SECDataError("SEC returned a non-JSON response for company facts.") from exc


# ---------------------------------------------------------------------------
# XBRL concept extraction  (same logic as original, refactored to take
# a single company's facts dict rather than the global company_facts[ticker])
# ---------------------------------------------------------------------------

def get_annual_flow_values(facts_data: dict, concept: str) -> dict:
    """
    Extract annual (10-K) flow values for *concept* from one company's facts.
    Keeps only entries whose start→end span is 350–380 days (full-year flows),
    keyed by filing end date. Deduplicated by latest filing date.
    """
    try:
        entries = facts_data["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        return {}

    annual: dict[str, tuple] = {}
    for entry in entries:
        if entry.get("form") != "10-K":
            continue
        start = entry.get("start")
        end = entry.get("end")
        if not start or not end:
            continue
        try:
            duration = (date.fromisoformat(end) - date.fromisoformat(start)).days
        except ValueError:
            continue
        if 350 <= duration <= 380:
            filed = entry.get("filed", "")
            if end not in annual or filed > annual[end][1]:
                annual[end] = (entry["val"], filed)

    return {end: val for end, (val, _) in annual.items()}


def get_annual_point_values(facts_data: dict, concept: str) -> dict:
    """
    Extract annual (10-K) point-in-time values for *concept* from one company's facts.
    Keyed by filing end date, deduplicated by latest filing date.
    """
    try:
        entries = facts_data["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        return {}

    annual: dict[str, tuple] = {}
    for entry in entries:
        if entry.get("form") != "10-K":
            continue
        end = entry.get("end")
        if not end:
            continue
        filed = entry.get("filed", "")
        if end not in annual or filed > annual[end][1]:
            annual[end] = (entry["val"], filed)

    return {end: val for end, (val, _) in annual.items()}


def get_revenue_values(facts_data: dict) -> dict:
    """Merge revenue across possible XBRL tag names (SalesRevenueNet → Revenues → ASC 606)."""
    merged: dict = {}
    for concept in _REVENUE_CONCEPTS:
        merged.update(get_annual_flow_values(facts_data, concept))
    return merged


def get_equity_values(facts_data: dict) -> dict:
    """
    Merge shareholders equity across tag variants.
    Processes broad tag first so the precise tag (StockholdersEquity) wins
    for years where both exist.  Visa, for example, used StockholdersEquity
    only through 2011 then switched to the Including... variant.
    """
    merged: dict = {}
    for concept in _EQUITY_CONCEPTS:
        merged.update(get_annual_point_values(facts_data, concept))
    return merged


def get_net_income_values(facts_data: dict) -> dict:
    """
    Merge net income across tag variants (ProfitLoss → NetIncomeLoss).
    NetIncomeLoss is preferred when present.
    """
    merged: dict = {}
    for concept in _NET_INCOME_CONCEPTS:
        merged.update(get_annual_flow_values(facts_data, concept))
    return merged


# ---------------------------------------------------------------------------
# Ratio computation
# ---------------------------------------------------------------------------

def build_ratio_table(facts_data: dict) -> pd.DataFrame:
    """
    Compute annual financial ratios from *facts_data*.
    Raises InsufficientDataError if there aren't at least 2 usable years.
    """
    revenues = get_revenue_values(facts_data)
    net_income = get_net_income_values(facts_data)
    assets = get_annual_point_values(facts_data, "Assets")
    equity = get_equity_values(facts_data)
    current_assets = get_annual_point_values(facts_data, "AssetsCurrent")
    current_liabilities = get_annual_point_values(facts_data, "LiabilitiesCurrent")

    liabilities = {y: assets[y] - equity[y] for y in assets if y in equity}

    # Core intersection: revenue, net income, assets, equity are always required.
    # Current assets/liabilities are optional — financial institutions don't report them.
    core_years = sorted(
        set(revenues)
        & set(net_income)
        & set(assets)
        & set(equity)
    )

    if not core_years:
        raise InsufficientDataError(
            "No years with complete data across all required concepts "
            "(Revenue, NetIncomeLoss, Assets, Equity). "
            "The company may use non-standard XBRL tags or have incomplete filings."
        )

    has_current = bool(current_assets and current_liabilities)

    rows = []
    for y in core_years:
        rev = revenues[y]
        eq = equity[y]

        # Skip years with zero denominators rather than crashing
        if rev == 0 or assets.get(y, 0) == 0:
            continue

        cl = current_liabilities.get(y)
        ca = current_assets.get(y)
        if has_current and (cl is None or cl == 0):
            continue

        current_ratio = (ca / cl) if (ca is not None and cl) else float("nan")

        rows.append({
            "year": int(y[:4]),
            "revenue": rev,
            "net_income": net_income[y],
            "net_margin": net_income[y] / rev,
            "current_ratio": current_ratio,
            "debt_to_equity": liabilities.get(y, 0) / eq if eq != 0 else float("nan"),
            "roe": net_income[y] / eq if eq != 0 else float("nan"),
            "roa": net_income[y] / assets[y],
        })

    if len(rows) < 2:
        raise InsufficientDataError(
            f"Only {len(rows)} usable year(s) after filtering zero-denominator rows — "
            "need at least 2 to produce a meaningful report."
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Risk flags  (unchanged from original)
# ---------------------------------------------------------------------------

def flag_risks(df: pd.DataFrame) -> list[tuple[int, str]]:
    flags: list[tuple[int, str]] = []
    df = df.sort_values("year").reset_index(drop=True)

    for i in range(len(df)):
        row = df.iloc[i]
        year = int(row["year"])

        if row["net_income"] < 0:
            flags.append((year, "Negative net income"))
        if not pd.isna(row["current_ratio"]) and row["current_ratio"] < 1.0:
            flags.append((year, "Current ratio below 1.0 (liquidity risk)"))
        if row["debt_to_equity"] > 3.0:
            flags.append((year, "High leverage (debt/equity > 3.0)"))
        if i >= 2:
            last3 = df.iloc[i - 2 : i + 1]["net_margin"]
            if last3.is_monotonic_decreasing:
                flags.append((year, "Net margin declined 3 consecutive years"))

    return flags


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_currency(val: float) -> str:
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1e12:
        return f"${val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"

def fmt_pct(val: float) -> str:
    return "N/A" if pd.isna(val) else f"{val*100:.1f}%"

def fmt_ratio(val: float) -> str:
    return "N/A" if pd.isna(val) else f"{val:.2f}x"


# ---------------------------------------------------------------------------
# Chart generation  (one combined figure, two styled subplots)
# ---------------------------------------------------------------------------

def make_charts(df: pd.DataFrame, ticker: str, output_dir: str) -> str:
    """Save a combined side-by-side chart PNG to *output_dir*. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)

    BLUE_HEX  = "#2E6DA4"
    RED_HEX   = "#C0392B"
    GRID_COL  = "#E8EDF4"
    SPINE_COL = "#CCCCCC"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.patch.set_facecolor("white")

    years = df["year"].tolist()

    # --- Net Margin ---
    margin_pct = df["net_margin"] * 100
    ax1.plot(years, margin_pct, marker="o", color=BLUE_HEX, linewidth=2, markersize=5, zorder=3)
    ax1.fill_between(years, margin_pct, alpha=0.12, color=BLUE_HEX)
    ax1.axhline(0, color=SPINE_COL, linewidth=0.8, linestyle="--", zorder=2)
    ax1.set_title("Net Margin", fontsize=11, fontweight="bold", color="#1B2A4A", pad=8)
    ax1.set_xlabel("Fiscal Year", fontsize=8, color="#7F8C8D")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.tick_params(labelsize=8, colors="#555555")
    ax1.set_facecolor("white")
    ax1.grid(axis="y", color=GRID_COL, linewidth=0.8, zorder=1)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax1.spines[spine].set_color(SPINE_COL)

    # --- Debt / Equity ---
    de = df["debt_to_equity"]
    ax2.plot(years, de, marker="o", color=RED_HEX, linewidth=2, markersize=5, zorder=3)
    ax2.fill_between(years, de, alpha=0.12, color=RED_HEX)
    ax2.axhline(3.0, color="#E67E22", linewidth=1, linestyle="--", alpha=0.8,
                zorder=2, label="Risk threshold (3.0×)")
    ax2.set_title("Debt / Equity", fontsize=11, fontweight="bold", color="#1B2A4A", pad=8)
    ax2.set_xlabel("Fiscal Year", fontsize=8, color="#7F8C8D")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.tick_params(labelsize=8, colors="#555555")
    ax2.set_facecolor("white")
    ax2.grid(axis="y", color=GRID_COL, linewidth=0.8, zorder=1)
    ax2.legend(fontsize=7, framealpha=0.7)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax2.spines[spine].set_color(SPINE_COL)

    plt.tight_layout(pad=1.8)
    chart_path = os.path.join(output_dir, f"{ticker}_charts.png")
    plt.savefig(chart_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    return chart_path


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def _section_header(text: str) -> list:
    """Return [label Paragraph, thin rule] as a visual section divider."""
    style = ParagraphStyle(
        "SectionHead",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=_NAVY,
        spaceBefore=14,
        spaceAfter=4,
    )
    return [
        Paragraph(text.upper(), style),
        HRFlowable(width=_CW, thickness=1, color=_BORDER, spaceAfter=6),
    ]


def generate_report(
    ticker: str,
    df: pd.DataFrame,
    flags: list[tuple[int, str]],
    output_dir: str = ".",
) -> str:
    """Build a polished PDF report and return the path to the saved file."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{ticker}_report.pdf")

    with tempfile.TemporaryDirectory() as chart_dir:
        chart_path = make_charts(df, ticker, chart_dir)
        elements = _build_elements(ticker, df, flags, chart_path)

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            leftMargin=_MARGIN, rightMargin=_MARGIN,
            topMargin=_MARGIN, bottomMargin=_MARGIN,
        )
        doc.build(elements)

    return pdf_path


def _build_elements(
    ticker: str,
    df: pd.DataFrame,
    flags: list[tuple[int, str]],
    chart_path: str,
) -> list:
    elements = []
    latest = df.iloc[-1]

    # ------------------------------------------------------------------
    # 1. Header bar
    # ------------------------------------------------------------------
    report_date = datetime.today().strftime("%B %d, %Y")
    header_style_L = ParagraphStyle(
        "HdrL", fontName="Helvetica-Bold", fontSize=16,
        textColor=colors.white, leading=20,
    )
    header_style_R = ParagraphStyle(
        "HdrR", fontName="Helvetica", fontSize=8,
        textColor=colors.HexColor("#BDC8DC"), alignment=TA_RIGHT,
    )
    header_table = Table(
        [[
            Paragraph(f"{ticker}", header_style_L),
            Paragraph(
                f"Financial Analysis Report<br/>Generated {report_date}",
                header_style_R,
            ),
        ]],
        colWidths=[_CW * 0.55, _CW * 0.45],
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), _NAVY),
        ("TOPPADDING",  (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",  (0, 0), (0, -1), 16),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 16),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 16))

    # ------------------------------------------------------------------
    # 2. KPI summary cards
    # ------------------------------------------------------------------
    kpi_label = ParagraphStyle(
        "KpiLabel", fontName="Helvetica", fontSize=7,
        textColor=_MUTED, alignment=TA_CENTER, spaceAfter=2,
    )
    kpi_value = ParagraphStyle(
        "KpiValue", fontName="Helvetica-Bold", fontSize=14,
        textColor=_NAVY, alignment=TA_CENTER,
    )
    kpi_sub = ParagraphStyle(
        "KpiSub", fontName="Helvetica", fontSize=7,
        textColor=_MUTED, alignment=TA_CENTER,
    )

    def kpi_cell(label: str, value: str, sub: str = "") -> list:
        cell = [Paragraph(label, kpi_label), Paragraph(value, kpi_value)]
        if sub:
            cell.append(Paragraph(sub, kpi_sub))
        return cell

    latest_year = int(latest["year"])
    kpi_data = [[
        kpi_cell("REVENUE",     fmt_currency(latest["revenue"]),  f"FY {latest_year}"),
        kpi_cell("NET MARGIN",  fmt_pct(latest["net_margin"]),    f"FY {latest_year}"),
        kpi_cell("ROE",         fmt_pct(latest["roe"]),           f"FY {latest_year}"),
        kpi_cell("CURRENT RATIO", fmt_ratio(latest["current_ratio"]), f"FY {latest_year}"),
    ]]
    kpi_col_w = _CW / 4 - 4
    kpi_table = Table(kpi_data, colWidths=[kpi_col_w] * 4, hAlign="LEFT")
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _LIGHT_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", [4]),
        ("BOX",  (0, 0), (0, -1),  0.5, _BORDER),
        ("BOX",  (1, 0), (1, -1),  0.5, _BORDER),
        ("BOX",  (2, 0), (2, -1),  0.5, _BORDER),
        ("BOX",  (3, 0), (3, -1),  0.5, _BORDER),
        ("LINEAFTER", (0, 0), (2, -1), 0.5, _BORDER),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 18))

    # ------------------------------------------------------------------
    # 3. Data table (last 5 years, formatted)
    # ------------------------------------------------------------------
    elements.extend(_section_header("Financial Summary — Last 5 Years"))

    COL_HEADERS = ["Year", "Revenue", "Net Income", "Net Margin",
                   "Current Ratio", "Debt / Equity", "ROE", "ROA"]
    recent = df.tail(5)
    rows = [COL_HEADERS]
    for _, r in recent.iterrows():
        rows.append([
            str(int(r["year"])),
            fmt_currency(r["revenue"]),
            fmt_currency(r["net_income"]),
            fmt_pct(r["net_margin"]),
            fmt_ratio(r["current_ratio"]),
            fmt_ratio(r["debt_to_equity"]),
            fmt_pct(r["roe"]),
            fmt_pct(r["roa"]),
        ])

    col_w = _CW / len(COL_HEADERS)
    data_table = Table(rows, colWidths=[col_w] * len(COL_HEADERS), repeatRows=1)

    tbl_style = [
        # Header row
        ("BACKGROUND",    (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8),
        ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        # Data rows
        ("FONTNAME",  (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 1), (-1, -1), 8),
        ("ALIGN",     (0, 1), (0, -1),  "CENTER"),   # Year centered
        ("ALIGN",     (1, 1), (-1, -1), "RIGHT"),    # Numbers right-aligned
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("RIGHTPADDING",  (1, 0), (-1, -1), 8),
        # Grid
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _BORDER),
        ("LINEAFTER", (0, 0), (0, -1),  0.4, _BORDER),
    ]
    # Alternating row shading
    for i in range(2, len(rows), 2):
        tbl_style.append(("BACKGROUND", (0, i), (-1, i), _ALT_ROW))

    data_table.setStyle(TableStyle(tbl_style))
    elements.append(data_table)
    elements.append(Spacer(1, 18))

    # ------------------------------------------------------------------
    # 4. Charts
    # ------------------------------------------------------------------
    elements.extend(_section_header("Key Metrics Over Time"))
    elements.append(Image(chart_path, width=_CW, height=_CW * 0.38))
    elements.append(Spacer(1, 18))

    # ------------------------------------------------------------------
    # 5. Risk flags
    # ------------------------------------------------------------------
    elements.extend(_section_header("Risk Indicators"))

    _RED_FLAGS  = {"Negative net income", "Current ratio below 1.0 (liquidity risk)"}

    if not flags:
        ok_style = ParagraphStyle(
            "OkText", fontName="Helvetica", fontSize=9, textColor=_GREEN,
        )
        elements.append(Paragraph("No risk indicators identified.", ok_style))
    else:
        flag_label = ParagraphStyle(
            "FlagLabel", fontName="Helvetica-Bold", fontSize=8, textColor=_DARK,
        )
        flag_sub = ParagraphStyle(
            "FlagSub", fontName="Helvetica", fontSize=8, textColor=_DARK,
        )
        flag_rows = []
        for year, msg in flags:
            is_red = any(kw in msg for kw in _RED_FLAGS)
            dot_color = _RED if is_red else _AMBER
            dot_cell = Table([[""]], colWidths=[6], rowHeights=[6])
            dot_cell.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), dot_color),
                ("ROUNDEDCORNERS", [3]),
            ]))
            flag_rows.append([
                dot_cell,
                [Paragraph(str(year), flag_label), Paragraph(msg, flag_sub)],
            ])

        flag_table = Table(
            flag_rows,
            colWidths=[14, _CW - 14],
            hAlign="LEFT",
        )
        flag_table.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (0, -1),  0),
            ("LEFTPADDING",   (1, 0), (1, -1),  8),
            ("LINEBELOW",     (0, 0), (-1, -2), 0.3, _BORDER),
        ]))
        elements.append(KeepTogether(flag_table))

    return elements


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_company(ticker: str, output_dir: str = ".") -> str:
    """
    Run the full pipeline for *ticker*:
      CIK lookup → fetch EDGAR facts → compute ratios → flag risks → PDF report.

    Returns the path to the generated PDF.

    Raises:
        TickerNotFoundError  – ticker not in SEC database
        SECDataError         – API/network failure or missing XBRL data
        InsufficientDataError – too few complete annual filings to compute ratios
    """
    ticker = ticker.upper().strip()

    print(f"[{ticker}] Looking up CIK...")
    cik = get_cik(ticker)

    print(f"[{ticker}] Fetching EDGAR data (CIK {cik})...")
    facts_data = get_company_facts(cik)
    time.sleep(0.5)  # respect SEC rate limits

    print(f"[{ticker}] Computing ratios...")
    df = build_ratio_table(facts_data)

    flags = flag_risks(df)

    print(f"[{ticker}] Generating report...")
    pdf_path = generate_report(ticker, df, flags, output_dir)
    print(f"[{ticker}] Saved: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "MSFT", "AMZN"]

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        try:
            analyze_company(ticker)
        except TickerNotFoundError as exc:
            print(f"  Ticker not found: {exc}")
        except SECDataError as exc:
            print(f"  SEC data error: {exc}")
        except InsufficientDataError as exc:
            print(f"  Insufficient data: {exc}")
