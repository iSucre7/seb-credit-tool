"""
pdf_reader.py

Extracts and structures text from corporate financial report PDFs.

Handles:
- Single and two-up page layouts
- Norwegian and English language reports
- Group vs parent statement separation
- Section splitting for targeted agent extraction
- Scanned PDF detection
- Graceful fallback when structure cannot be determined

Returns structured dict ready for downstream agents.
"""

import pdfplumber
import anthropic
import json
import os
import re

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Maximum characters per section per API call
# Haiku/Sonnet safe limit ~180k characters
MAX_CHARS_PER_SECTION = 180000

# Minimum text length to consider a page non-scanned
MIN_PAGE_TEXT_LENGTH = 50

# Number of pages to send to Haiku for TOC analysis
TOC_PAGES_TO_READ = 10


# ─────────────────────────────────────────────────────────────
# STEP 1 — RAW TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from every page of a PDF.

    Returns:
        pages       : list of {page_number, text}
        scanned_flag: True if document appears to be scanned

    Scanned detection:
        If more than 30% of pages return no text,
        the document is likely a scanned image.
        Flag it and return what we have.
    """
    pages = []
    empty_page_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"  PDF has {total_pages} pages")

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()

                if text and len(text.strip()) >= MIN_PAGE_TEXT_LENGTH:
                    pages.append({
                        "page_number": i + 1,
                        "text": text.strip()
                    })
                else:
                    empty_page_count += 1

        scanned = empty_page_count / total_pages > 0.3

        if scanned:
            print(f"  WARNING: {empty_page_count}/{total_pages} pages "
                  f"returned no text — document may be scanned")
            print(f"  Scanned PDFs cannot be processed automatically")
            print(f"  Please source a text-based version of this report")
            return pages, True

        print(f"  Extracted text from {len(pages)} pages "
              f"({empty_page_count} empty pages skipped)")
        return pages, False

    except Exception as e:
        print(f"  Error reading PDF: {e}")
        return [], False


# ─────────────────────────────────────────────────────────────
# STEP 2 — DOCUMENT STRUCTURE DETECTION (HAIKU)
# ─────────────────────────────────────────────────────────────

def get_document_structure(pages):
    """
    Sends first N pages to Haiku to detect:
    - Page layout (single or two-up)
    - Document language and currency
    - Exact page boundaries for all key sections
    - Whether parent statements exist

    Works for any language or report format.

    Returns structured dict or None if detection fails.
    """
    from config.settings import ANTHROPIC_API_KEY, MODELS
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build TOC text from first N pages
    toc_pages = pages[:TOC_PAGES_TO_READ]
    toc_text = "\n\n--- PAGE BREAK ---\n\n".join(
        [f"[PDF PAGE {p['page_number']}]\n{p['text']}"
         for p in toc_pages]
    )

    prompt = """You are analysing the opening pages of a corporate annual report.
Your job is to detect the document structure and find exact page boundaries.

TASK 1 — LAYOUT DETECTION
Some PDFs print two document pages side by side on one PDF page (two-up layout).
Look at the page numbers visible in the document text.

Examples:
- If PDF page 2 shows document page numbers "2" and "3" → two-up layout
- If PDF page 2 shows only document page number "2" → single layout

For two-up layout, determine the conversion formula by checking
multiple page number references. Common formulas:
- pdf_page = (doc_page / 2) + 1  (when cover is single page)
- pdf_page = doc_page / 2         (when all pages are two-up)

Verify your formula against at least two known page references
from the table of contents before committing to it.

TASK 2 — TABLE OF CONTENTS PARSING
Find the table of contents and extract page numbers for key sections.

Corporate annual reports often contain TWO sets of financial statements:
1. Consolidated/Group (konsern) — the main set analysts use
2. Parent company (morselskap/AS) — secondary set, less relevant

Page numbers in the TOC are DOCUMENT page numbers, not PDF page numbers.

Look for these sections in any language:
Income statement     / Resultatregnskap    / Résultat consolidé
Balance sheet        / Balanse             / Bilan
Cash flow statement  / Kontantstrøm        / Flux de trésorerie
Notes                / Noter               / Annexes
Accounting policies  / Regnskapsprinsipper / Méthodes comptables

TASK 3 — FLAG ISSUES
Flag any of these if detected:
- Page numbering restarts mid-document
- Multiple currencies or units used
- No table of contents found
- Low confidence in any page number

Return ONLY valid JSON with no preamble or explanation:
{
  "layout": "single/two-up",
  "doc_to_pdf_formula": "pdf = doc / 2 + 1",
  "formula_verified": true/false,
  "language": "Norwegian/English/Swedish/Danish/Other",
  "currency": "NOK/SEK/DKK/EUR/GBP/USD",
  "unit": "millions/thousands/billions",
  "mixed_units": false,
  "has_toc": true/false,
  "has_parent_statements": true/false,
  "page_numbering_restarts": false,
  "group_financials": {
    "income_statement_page": null,
    "balance_sheet_page": null,
    "cash_flow_page": null,
    "notes_page": null,
    "accounting_policies_page": null
  },
  "parent_financials": {
    "income_statement_page": null,
    "balance_sheet_page": null,
    "cash_flow_page": null,
    "notes_page": null
  },
  "confidence": "high/medium/low",
  "flags": [],
  "notes": null
}"""

    print(f"  Haiku analysing document structure...")

    try:
        response = client.messages.create(
            model=MODELS["extractor"],
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": (f"{prompt}\n\n"
                            f"DOCUMENT OPENING PAGES:\n{toc_text}")
            }]
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        structure = json.loads(raw)

        # Print summary
        print(f"  Layout:     {structure.get('layout')}"
              f" — formula: {structure.get('doc_to_pdf_formula')}"
              f" (verified: {structure.get('formula_verified')})")
        print(f"  Language:   {structure.get('language')}")
        print(f"  Currency:   {structure.get('currency')} "
              f"{structure.get('unit')}")
        print(f"  Has TOC:    {structure.get('has_toc')}")
        print(f"  Confidence: {structure.get('confidence')}")

        gf = structure.get("group_financials", {})
        print(f"  Group — income statement:  "
              f"doc page {gf.get('income_statement_page')}")
        print(f"  Group — notes:             "
              f"doc page {gf.get('notes_page')}")

        if structure.get("has_parent_statements"):
            pf = structure.get("parent_financials", {})
            print(f"  Parent — income statement: "
                  f"doc page {pf.get('income_statement_page')}")

        # Print any flags
        flags = structure.get("flags", [])
        if flags:
            for flag in flags:
                print(f"  FLAG: {flag}")

        if structure.get("page_numbering_restarts"):
            print(f"  WARNING: Page numbering restarts detected — "
                  f"section boundaries may be unreliable")

        if structure.get("mixed_units"):
            print(f"  WARNING: Mixed units detected in document — "
                  f"verify extraction output carefully")

        return structure

    except (json.JSONDecodeError, Exception) as e:
        print(f"  Could not parse document structure: {e}")
        print(f"  Falling back to full document extraction")
        return None


# ─────────────────────────────────────────────────────────────
# STEP 3 — PAGE NUMBER CONVERSION
# ─────────────────────────────────────────────────────────────

def doc_to_pdf_page(doc_page, structure):
    """
    Converts a document page number to a PDF page number
    using the layout formula detected by Haiku.

    Falls back to 1:1 mapping if structure unavailable.
    """
    if not structure or not doc_page:
        return doc_page

    layout = structure.get("layout", "single")

    if layout == "two-up":
        formula = structure.get(
            "doc_to_pdf_formula", "pdf = doc / 2 + 1"
        )
        # Parse and apply the formula
        # Supported: "pdf = doc / 2 + 1" or "pdf = doc / 2"
        if "+ 1" in formula:
            return max(1, round(doc_page / 2) + 1)
        else:
            return max(1, round(doc_page / 2))

    return doc_page  # single layout — 1:1 mapping


# ─────────────────────────────────────────────────────────────
# STEP 4 — SPLIT GROUP VS PARENT PAGES
# ─────────────────────────────────────────────────────────────

def split_group_parent(pages, structure):
    """
    Uses detected page boundaries to separate group
    and parent financial statement pages.

    Returns only group pages.
    Parent pages are excluded from all downstream processing.
    """
    if not structure or not structure.get("has_parent_statements"):
        print(f"  No parent statements — using all {len(pages)} pages")
        return pages

    parent_doc_page = structure.get(
        "parent_financials", {}
    ).get("income_statement_page")

    if not parent_doc_page:
        print(f"  Parent statements flagged but page not found — "
              f"using all pages")
        return pages

    parent_pdf_page = doc_to_pdf_page(parent_doc_page, structure)

    print(f"  Parent doc page {parent_doc_page} "
          f"→ PDF page {parent_pdf_page}")

    group_pages = [
        p for p in pages
        if p["page_number"] < parent_pdf_page
    ]
    parent_pages = [
        p for p in pages
        if p["page_number"] >= parent_pdf_page
    ]

    print(f"  Group pages:  {len(group_pages)} "
          f"(PDF pages 1–{parent_pdf_page - 1})")
    print(f"  Parent pages: {len(parent_pages)} "
          f"(PDF pages {parent_pdf_page}+) — excluded")

    return group_pages


# ─────────────────────────────────────────────────────────────
# STEP 5 — SPLIT INTO SECTIONS
# ─────────────────────────────────────────────────────────────

def split_into_sections(pages, structure):
    """
    Splits group pages into two targeted sections:

    financial_statements: P&L, balance sheet, cash flow,
                          equity statement
                          (from income statement to notes)

    notes_financial:      accounting policies, pension,
                          lease, finance costs, debt schedule
                          (from notes to end of group section)
    """
    gf = structure.get("group_financials", {}) if structure else {}
    income_doc = gf.get("income_statement_page")
    notes_doc  = gf.get("notes_page") or \
                 gf.get("accounting_policies_page")

    income_pdf = doc_to_pdf_page(income_doc, structure)
    notes_pdf  = doc_to_pdf_page(notes_doc, structure)

    print(f"  Income statement: doc p{income_doc} "
          f"→ PDF p{income_pdf}")
    print(f"  Notes start:      doc p{notes_doc} "
          f"→ PDF p{notes_pdf}")

    statements = []
    notes      = []

    for p in pages:
        pn = p["page_number"]

        if income_pdf is None:
            statements.append(p)
        elif notes_pdf is None:
            if pn >= income_pdf:
                statements.append(p)
        else:
            if income_pdf <= pn < notes_pdf:
                statements.append(p)
            elif pn >= notes_pdf:
                notes.append(p)

    buckets = {
        "financial_statements": statements,
        "notes_financial":      notes
    }

    result = {}
    for section, section_pages in buckets.items():

        if not section_pages:
            result[section] = ""
            print(f"  Section '{section}': no pages")
            continue

        combined = "\n\n--- PAGE BREAK ---\n\n".join(
            [p["text"] for p in section_pages]
        )

        if len(combined) > MAX_CHARS_PER_SECTION:
            cutoff = combined.rfind(
                "--- PAGE BREAK ---", 0, MAX_CHARS_PER_SECTION
            )
            combined = (combined[:cutoff] if cutoff > 0
                        else combined[:MAX_CHARS_PER_SECTION])
            combined += "\n\n[TRUNCATED — section exceeded limit]"
            print(f"  Section '{section}': "
                  f"{len(section_pages)} pages — TRUNCATED")
        else:
            print(f"  Section '{section}': "
                  f"{len(section_pages)} pages, "
                  f"{len(combined):,} chars")

        result[section] = combined

    return result

# ─────────────────────────────────────────────────────────────
# STEP 6 — REPORT TYPE FROM FILENAME
# ─────────────────────────────────────────────────────────────

def get_report_type(filename):
    """
    Identifies report type and period from filename.

    Supported formats:
        annual_YYYY.pdf    → FY annual report
        qX_YYYY.pdf        → Quarterly report
        h1_YYYY.pdf        → Half year report

    Returns dict with type, year, quarter, period,
    and applicable accounting standard (IFRS16 from 2019).
    """
    filename = filename.lower()

    annual = re.match(r"annual_(\d{4})\.pdf", filename)
    if annual:
        year = int(annual.group(1))
        return {
            "type": "annual",
            "year": year,
            "quarter": None,
            "period": f"FY{year}",
            "accounting_standard": (
                "IFRS16" if year >= 2019 else "IFRS_PRE16"
            )
        }

    quarterly = re.match(r"q([1-4])_(\d{4})\.pdf", filename)
    if quarterly:
        quarter = int(quarterly.group(1))
        year    = int(quarterly.group(2))
        return {
            "type": "quarterly",
            "year": year,
            "quarter": quarter,
            "period": f"Q{quarter} {year}",
            "accounting_standard": (
                "IFRS16" if year >= 2019 else "IFRS_PRE16"
            )
        }

    h1 = re.match(r"h1_(\d{4})\.pdf", filename)
    if h1:
        year = int(h1.group(1))
        return {
            "type": "half_year",
            "year": year,
            "quarter": None,
            "period": f"H1 {year}",
            "accounting_standard": (
                "IFRS16" if year >= 2019 else "IFRS_PRE16"
            )
        }

    return None


# ─────────────────────────────────────────────────────────────
# STEP 7 — FINANCIAL PAGE FILTER (QUARTERLY / H1)
# ─────────────────────────────────────────────────────────────

QUARTERLY_FINANCIAL_KEYWORDS = [
    # Norwegian
    "resultatregnskap", "balanse",
    "kontantstrøm", "totalresultat", "hovedtall",
    # English
    "income statement", "balance sheet",
    "cash flow", "comprehensive income",
    "key figures", "financial highlights"
]

def filter_financial_pages(pages):
    """
    For quarterly and H1 reports — returns pages
    that contain financial statement content.

    These reports are short enough that we do not
    need TOC-based boundary detection.
    """
    financial_pages = [
        p for p in pages
        if any(
            kw in p["text"].lower()
            for kw in QUARTERLY_FINANCIAL_KEYWORDS
        )
    ]
    return financial_pages


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def prepare_for_extraction(company_name, filename):
    """
    Master function — reads a PDF and prepares
    structured text for downstream agents.

    Annual reports:
        → Haiku reads TOC for layout and page boundaries
        → Group pages separated from parent pages
        → Split into three targeted sections:
           intelligence / financial_statements / notes_financial

    Quarterly / H1 reports:
        → Financial pages filtered
        → Single combined text block returned

    Returns dict with:
        company         : company name
        filename        : source filename
        report_info     : type, year, quarter, period, standard
        full_document   : True for annuals, False for quarterlies
        pages_extracted : number of pages in output

        For annuals:
            structure   : document structure from Haiku
            sections    : {intelligence, financial_statements,
                           notes_financial, metadata}

        For quarterlies:
            financial_text : combined text of financial pages

    Returns None if extraction fails.
    """
    from config.settings import DATA_DIR

    pdf_path = os.path.join(
        DATA_DIR, company_name, "raw", filename
    )

    print(f"\nProcessing: {filename}")
    print(f"  Path: {pdf_path}")

    # Validate file exists
    if not os.path.exists(pdf_path):
        print(f"  ERROR: File not found at {pdf_path}")
        return None

    # Identify report type from filename
    report_info = get_report_type(filename)
    if not report_info:
        print(f"  ERROR: Cannot identify report type — "
              f"check filename format")
        print(f"  Expected: annual_YYYY.pdf / "
              f"qX_YYYY.pdf / h1_YYYY.pdf")
        return None

    print(f"  Report type: {report_info['period']}")
    print(f"  Standard:    {report_info['accounting_standard']}")

    # Extract raw text
    all_pages, is_scanned = extract_text_from_pdf(pdf_path)

    if is_scanned:
        print(f"  SKIPPING: Scanned document — "
              f"text extraction not possible")
        return None

    if not all_pages:
        print(f"  SKIPPING: No text extracted from document")
        return None

    # ── ANNUAL REPORT PATH ──────────────────────────────────
    if report_info["type"] == "annual":

        # Detect layout and section boundaries via Haiku
        structure = get_document_structure(all_pages)

        # Graceful fallback if structure detection fails
        if not structure or structure.get("confidence") == "low":
            print(f"  WARNING: Low confidence structure detection")
            print(f"  Falling back to full document as single block")
            combined = "\n\n--- PAGE BREAK ---\n\n".join(
                [p["text"] for p in all_pages]
            )
            return {
                "company":         company_name,
                "filename":        filename,
                "report_info":     report_info,
                "structure":       structure,
                "sections": {
                    "intelligence":         "",
                    "financial_statements": combined,
                    "notes_financial":      "",
                    "metadata":             structure
                },
                "full_document":   True,
                "pages_extracted": len(all_pages),
                "fallback":        True
            }

        # Separate group and parent pages
        group_pages = split_group_parent(all_pages, structure)

        # Split into targeted sections
        sections = split_into_sections(group_pages, structure)
        sections["metadata"] = structure

        return {
            "company":         company_name,
            "filename":        filename,
            "report_info":     report_info,
            "structure":       structure,
            "sections":        sections,
            "full_document":   True,
            "pages_extracted": len(group_pages),
            "fallback":        False
        }

    # ── QUARTERLY / H1 PATH ─────────────────────────────────
    else:

        financial_pages = filter_financial_pages(all_pages)

        if not financial_pages:
            print(f"  No financial pages detected — "
                  f"using full document")
            financial_pages = all_pages
        else:
            print(f"  Financial pages found: {len(financial_pages)}")

        combined_text = "\n\n--- PAGE BREAK ---\n\n".join(
            [p["text"] for p in financial_pages]
        )
        print(f"  Text length: {len(combined_text):,} characters")

        return {
            "company":         company_name,
            "filename":        filename,
            "report_info":     report_info,
            "financial_text":  combined_text,
            "full_document":   False,
            "pages_extracted": len(financial_pages),
            "fallback":        False
        }