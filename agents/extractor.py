"""
extractor.py

Two-pass financial data extraction from prepared PDF text.

Pass 1 (financial_statements section):
    Extracts P&L, balance sheet, cash flow statement
    using Haiku.

Pass 2 (notes_financial section):
    Extracts all adjustment-relevant note disclosures
    using Haiku — pensions, leases, finance costs,
    debt schedule, segments, working capital detail.

Both passes return structured JSON saved to
data/companies/{company}/extracted/
"""

import anthropic
import json
import os
from config.settings import ANTHROPIC_API_KEY, MODELS, DATA_DIR

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ─────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────

def parse_json_response(raw):
    """
    Cleans and parses JSON from Haiku response.
    Strips markdown code fences if present.
    Returns parsed dict or error dict.
    """
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    try:
        return json.loads(clean), None
    except json.JSONDecodeError as e:
        return None, str(e)


def save_json(data, company_name, filename):
    """
    Saves extracted JSON to company's extracted/ folder.
    """
    extracted_dir = os.path.join(
        DATA_DIR, company_name, "extracted"
    )
    os.makedirs(extracted_dir, exist_ok=True)

    filepath = os.path.join(extracted_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Saved: extracted/{filename}")
    return filepath


def build_output_filename(report_info, suffix):
    """
    Builds output filename matching source PDF convention.
    e.g. annual_2024_financials.json
         q4_2025_notes.json
    """
    if report_info["type"] == "annual":
        base = f"annual_{report_info['year']}"
    elif report_info["type"] == "quarterly":
        base = f"q{report_info['quarter']}_{report_info['year']}"
    elif report_info["type"] == "half_year":
        base = f"h1_{report_info['year']}"
    else:
        base = report_info["period"].replace(" ", "_")

    return f"{base}_{suffix}.json"


# ─────────────────────────────────────────────────────────────
# PASS 1 — FINANCIAL STATEMENTS
# ─────────────────────────────────────────────────────────────

STATEMENTS_PROMPT = """You are a financial data extraction specialist
working for a Nordic investment bank credit analyst.

You are reading text extracted from a corporate financial report.
The text may be in Norwegian or English.
Table columns are often jumbled due to PDF extraction — 
use your knowledge of financial statement structure to
correctly assign figures to line items and periods.

CRITICAL RULES:
1. Extract ONLY figures explicitly stated in the text
2. Never calculate or estimate — null if not found
3. Extract ALL periods shown — current year AND prior year
4. Flag restatements — if prior year figures differ from
   what was reported in the prior year annual, note it
5. Distinguish reported vs underlying/adjusted figures
   where the company discloses both
6. All figures in the currency and unit stated
   (typically NOK millions for Norwegian companies)
7. For quarterly reports: extract BOTH standalone quarter
   AND year-to-date figures where shown

Norwegian to English mapping:
Driftsinntekter          = Revenue / Operating revenue
Energisalg               = Energy sales
Overføringsinntekter     = Grid / Transmission revenue
EBITDA                   = EBITDA
Driftsresultat           = EBIT / Operating profit
Av- og nedskrivninger    = Depreciation and amortisation
Finansinntekter          = Finance income
Finanskostnader          = Finance costs
Resultat før skatt       = Profit before tax
Skattekostnad            = Tax expense
Årsresultat/Nettoresultat = Net income
Majoritetens andel       = Majority (parent) share
Minoritetens andel       = Minority (NCI) share
Kontantstrøm fra driften = Cash flow from operations (CFO)
Investeringer            = Capital expenditure (Capex)
Rentebærende gjeld       = Interest-bearing debt
Egenkapital              = Equity
Sysselsatt kapital       = Employed / Invested capital
Betalingsmidler          = Cash and equivalents
Kundefordringer          = Trade receivables
Leverandørgjeld          = Trade payables
Utsatt skatt             = Deferred tax
Underliggende            = Underlying (management adjusted)
Høyprisavgift            = High-price levy (Norwegian tax)

REPORT INFORMATION:
Company: {company}
Period: {period}
Report type: {report_type}
Accounting standard: {accounting_standard}
Currency: {currency}
Unit: {unit}

Return ONLY valid JSON with no preamble:

{{
  "extraction_metadata": {{
    "company": "{company}",
    "period": "{period}",
    "report_type": "{report_type}",
    "accounting_standard": "{accounting_standard}",
    "currency": "{currency}",
    "unit": "{unit}",
    "confidence": "high/medium/low",
    "restatements_noted": false,
    "restatement_detail": null,
    "extraction_notes": null
  }},

  "income_statement": {{
    "current_period": {{
      "label": null,
      "revenue_reported": null,
      "revenue_underlying": null,
      "ebitda_reported": null,
      "ebitda_underlying": null,
      "depreciation_amortisation": null,
      "ebit_reported": null,
      "ebit_underlying": null,
      "share_of_associates": null,
      "finance_income": null,
      "finance_costs_reported": null,
      "unrealised_fv_movements": null,
      "profit_before_tax": null,
      "tax_expense": null,
      "net_income_reported": null,
      "net_income_underlying": null,
      "minority_interest": null,
      "majority_net_income": null,
      "high_price_levy": null,
      "other_unusual_items": null
    }},
    "prior_period": {{
      "label": null,
      "revenue_reported": null,
      "revenue_underlying": null,
      "ebitda_reported": null,
      "ebitda_underlying": null,
      "depreciation_amortisation": null,
      "ebit_reported": null,
      "ebit_underlying": null,
      "share_of_associates": null,
      "finance_income": null,
      "finance_costs_reported": null,
      "unrealised_fv_movements": null,
      "profit_before_tax": null,
      "tax_expense": null,
      "net_income_reported": null,
      "net_income_underlying": null,
      "minority_interest": null,
      "majority_net_income": null,
      "high_price_levy": null,
      "other_unusual_items": null
    }},
    "ytd_current": null,
    "ytd_prior": null
  }},

  "balance_sheet": {{
    "current_period_end": {{
      "label": null,
      "cash_and_equivalents": null,
      "restricted_cash": null,
      "trade_receivables": null,
      "inventories": null,
      "other_current_assets": null,
      "total_current_assets": null,
      "ppe_net": null,
      "right_of_use_assets": null,
      "intangible_assets": null,
      "goodwill": null,
      "investments_in_associates": null,
      "other_non_current_assets": null,
      "deferred_tax_asset": null,
      "total_assets": null,
      "trade_payables": null,
      "short_term_debt": null,
      "current_lease_liabilities": null,
      "other_current_liabilities": null,
      "total_current_liabilities": null,
      "long_term_debt": null,
      "non_current_lease_liabilities": null,
      "pension_liability": null,
      "deferred_tax_liability": null,
      "provisions": null,
      "other_non_current_liabilities": null,
      "total_liabilities": null,
      "total_equity": null,
      "minority_interest_equity": null,
      "majority_equity": null,
      "employed_capital": null,
      "net_debt_reported": null
    }},
    "prior_period_end": {{
      "label": null,
      "cash_and_equivalents": null,
      "restricted_cash": null,
      "trade_receivables": null,
      "inventories": null,
      "other_current_assets": null,
      "total_current_assets": null,
      "ppe_net": null,
      "right_of_use_assets": null,
      "intangible_assets": null,
      "goodwill": null,
      "investments_in_associates": null,
      "other_non_current_assets": null,
      "deferred_tax_asset": null,
      "total_assets": null,
      "trade_payables": null,
      "short_term_debt": null,
      "current_lease_liabilities": null,
      "other_current_liabilities": null,
      "total_current_liabilities": null,
      "long_term_debt": null,
      "non_current_lease_liabilities": null,
      "pension_liability": null,
      "deferred_tax_liability": null,
      "provisions": null,
      "other_non_current_liabilities": null,
      "total_liabilities": null,
      "total_equity": null,
      "minority_interest_equity": null,
      "majority_equity": null,
      "employed_capital": null,
      "net_debt_reported": null
    }}
  }},

  "cash_flow": {{
    "current_period": {{
      "label": null,
      "cfo_reported": null,
      "change_in_working_capital": null,
      "tax_paid": null,
      "interest_paid": null,
      "interest_received": null,
      "capex": null,
      "proceeds_from_disposals": null,
      "acquisitions": null,
      "dividends_paid_to_shareholders": null,
      "dividends_paid_to_minorities": null,
      "debt_raised": null,
      "debt_repaid": null,
      "lease_principal_repaid": null,
      "net_change_in_cash": null,
      "fcf_reported": null
    }},
    "prior_period": {{
      "label": null,
      "cfo_reported": null,
      "change_in_working_capital": null,
      "tax_paid": null,
      "interest_paid": null,
      "interest_received": null,
      "capex": null,
      "proceeds_from_disposals": null,
      "acquisitions": null,
      "dividends_paid_to_shareholders": null,
      "dividends_paid_to_minorities": null,
      "debt_raised": null,
      "debt_repaid": null,
      "lease_principal_repaid": null,
      "net_change_in_cash": null,
      "fcf_reported": null
    }}
  }}
}}"""


# ─────────────────────────────────────────────────────────────
# PASS 2 — NOTES
# ─────────────────────────────────────────────────────────────

NOTES_PROMPT = """You are a financial data extraction specialist
working for a Nordic investment bank credit analyst.

You are reading the notes to consolidated financial statements.
The text may be in Norwegian or English and may be jumbled
due to PDF extraction.

Your job is to extract all data relevant to credit analysis
adjustments. Focus on finding the specific items listed below.
Extract ONLY figures explicitly stated — null if not found.

Norwegian note references:
Note 9  = Finansinntekter og finanskostnader (Finance costs)
Note 13 = Varige driftsmidler (PP&E / Capitalised interest)
Note 14 = Tilknyttede selskaper (Associates / JVs)
Note 20 = Pensjoner (Pensions)
Note 21 = Rentebærende gjeld (Debt schedule)
Note 1  = Segmentinformasjon (Segments)
Note 16 = Fordringer (Receivables)
Note 22 = Annen kortsiktig gjeld (Other current liabilities)
Note 19 = Uopptjente inntekter (Deferred income / provisions)

REPORT INFORMATION:
Company: {company}
Period: {period}
Accounting standard: {accounting_standard}
Currency: {currency}
Unit: {unit}

Return ONLY valid JSON with no preamble:

{{
  "extraction_metadata": {{
    "company": "{company}",
    "period": "{period}",
    "accounting_standard": "{accounting_standard}",
    "currency": "{currency}",
    "unit": "{unit}",
    "confidence": "high/medium/low",
    "notes_found": [],
    "extraction_notes": null
  }},

  "finance_costs_breakdown": {{
    "current_period": {{
      "label": null,
      "interest_on_financial_debt": null,
      "interest_on_lease_liabilities": null,
      "interest_on_pension_obligations": null,
      "interest_on_provisions_unwinding": null,
      "capitalised_interest": null,
      "fx_gains_losses": null,
      "fair_value_movements_derivatives": null,
      "fair_value_movements_power_contracts": null,
      "interest_income": null,
      "other_finance_costs": null,
      "total_finance_costs_reported": null,
      "core_cash_interest_net": null
    }},
    "prior_period": {{
      "label": null,
      "interest_on_financial_debt": null,
      "interest_on_lease_liabilities": null,
      "interest_on_pension_obligations": null,
      "interest_on_provisions_unwinding": null,
      "capitalised_interest": null,
      "fx_gains_losses": null,
      "fair_value_movements_derivatives": null,
      "fair_value_movements_power_contracts": null,
      "interest_income": null,
      "other_finance_costs": null,
      "total_finance_costs_reported": null,
      "core_cash_interest_net": null
    }}
  }},

  "pension": {{
    "current_period_end": {{
      "defined_benefit_obligation": null,
      "plan_assets_fair_value": null,
      "net_pension_deficit": null,
      "pension_surplus": null,
      "service_cost_current_year": null,
      "interest_cost_on_obligation": null,
      "employer_contributions": null,
      "annual_pension_payments": null,
      "actuarial_discount_rate_pct": null,
      "salary_growth_rate_pct": null,
      "plan_type": "funded/unfunded/mixed",
      "notes": null
    }},
    "prior_period_end": {{
      "defined_benefit_obligation": null,
      "plan_assets_fair_value": null,
      "net_pension_deficit": null,
      "pension_surplus": null
    }}
  }},

  "leases_ifrs16": {{
    "current_period_end": {{
      "total_lease_liabilities": null,
      "current_lease_liabilities": null,
      "non_current_lease_liabilities": null,
      "rou_asset_carrying_value": null,
      "interest_expense_on_leases": null,
      "depreciation_of_rou_assets": null,
      "lease_principal_repaid": null,
      "short_term_lease_expense": null,
      "low_value_lease_expense": null
    }},
    "prior_period_end": {{
      "total_lease_liabilities": null,
      "current_lease_liabilities": null,
      "non_current_lease_liabilities": null
    }}
  }},

  "debt_schedule": {{
    "current_period_end": {{
      "total_interest_bearing_debt": null,
      "bank_loans": null,
      "bonds_issued": null,
      "commercial_paper": null,
      "subordinated_hybrid_debt": null,
      "shareholder_loans": null,
      "other_debt": null,
      "fixed_rate_portion": null,
      "floating_rate_portion": null,
      "weighted_average_interest_rate_pct": null,
      "maturity_profile": {{
        "within_1_year": null,
        "1_to_2_years": null,
        "2_to_3_years": null,
        "3_to_5_years": null,
        "beyond_5_years": null
      }},
      "undrawn_credit_facilities": null,
      "covenants_disclosed": null,
      "covenant_detail": null,
      "next_bond_maturity_date": null,
      "next_bond_maturity_amount": null
    }},
    "prior_period_end": {{
      "total_interest_bearing_debt": null,
      "bonds_issued": null
    }}
  }},

  "capitalised_interest": {{
    "current_period": {{
      "amount_capitalised": null,
      "capitalisation_rate_pct": null,
      "projects_note": null
    }},
    "prior_period": {{
      "amount_capitalised": null
    }}
  }},

  "segments": {{
    "segment_basis": null,
    "segments": [
      {{
        "name": null,
        "current_period": {{
          "revenue": null,
          "ebitda": null,
          "ebit": null,
          "capex": null,
          "assets": null,
          "depreciation": null
        }},
        "prior_period": {{
          "revenue": null,
          "ebitda": null,
          "ebit": null,
          "capex": null,
          "assets": null,
          "depreciation": null
        }}
      }}
    ],
    "eliminations_current": null,
    "eliminations_prior": null
  }},

  "associates_jvs": {{
    "current_period": {{
      "share_of_profit_loss": null,
      "dividends_received": null,
      "carrying_value": null,
      "material_associates": [
        {{
          "name": null,
          "ownership_pct": null,
          "carrying_value": null,
          "share_of_profit_loss": null,
          "nature": null
        }}
      ]
    }},
    "prior_period": {{
      "share_of_profit_loss": null,
      "dividends_received": null
    }}
  }},

  "working_capital_detail": {{
    "current_period_end": {{
      "trade_receivables_gross": null,
      "bad_debt_provision": null,
      "trade_receivables_net": null,
      "other_receivables": null,
      "prepayments": null,
      "inventories_breakdown": null,
      "trade_payables": null,
      "accrued_liabilities": null,
      "deferred_revenue": null,
      "other_current_liabilities": null
    }},
    "prior_period_end": {{
      "trade_receivables_net": null,
      "trade_payables": null
    }}
  }},

  "provisions_and_contingencies": {{
    "decommissioning_provision": null,
    "restructuring_provision": null,
    "legal_provision": null,
    "other_provisions": null,
    "material_contingent_liabilities": null,
    "guarantees_given": null
  }},

  "unusual_items_reconciliation": {{
    "current_period": {{
      "unrealised_power_contract_gains_losses": null,
      "unrealised_fx_gains_losses": null,
      "impairments": null,
      "disposal_gains_losses": null,
      "merger_amortisation": null,
      "restructuring_costs": null,
      "other_unusual_items": null,
      "total_adjustments_to_underlying": null
    }},
    "prior_period": {{
      "unrealised_power_contract_gains_losses": null,
      "unrealised_fx_gains_losses": null,
      "impairments": null,
      "disposal_gains_losses": null,
      "total_adjustments_to_underlying": null
    }}
  }},

  "hybrid_instruments": {{
    "exists": false,
    "description": null,
    "carrying_value": null,
    "interest_expense": null,
    "equity_credit_features": null
  }},

  "restricted_cash": {{
    "amount": null,
    "reason": null
  }},

  "factoring_securitisation": {{
    "exists": false,
    "amount": null,
    "description": null
  }}
}}"""


# ─────────────────────────────────────────────────────────────
# EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def extract_financial_statements(company_name, prepared_data):
    """
    Pass 1 — Extracts P&L, balance sheet, cash flow.
    Works on both annual (sections dict) and
    quarterly (financial_text string) prepared data.
    """
    report_info = prepared_data["report_info"]
    boundaries  = prepared_data.get("boundaries") or {}
    metadata    = (prepared_data.get("sections") or {}).get(
        "metadata", {}
    ) or {}

    currency = (
        boundaries.get("currency") or
        metadata.get("currency") or
        "NOK"
    )
    unit = (
        boundaries.get("unit") or
        metadata.get("unit") or
        "millions"
    )

    # Get the right text block
    if prepared_data.get("full_document"):
        text = prepared_data["sections"]["financial_statements"]
    else:
        text = prepared_data["financial_text"]

    if not text:
        print(f"  No financial statements text available")
        return None

    prompt = STATEMENTS_PROMPT.format(
        company=company_name,
        period=report_info["period"],
        report_type=report_info["type"],
        accounting_standard=report_info["accounting_standard"],
        currency=currency,
        unit=unit
    )

    print(f"  Pass 1: extracting financial statements...")

    response = client.messages.create(
        model=MODELS["extractor"],
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nFINANCIAL STATEMENTS TEXT:\n{text}"
        }]
    )

    result, error = parse_json_response(
        response.content[0].text
    )

    if error:
        print(f"  Parse error: {error}")
        return {"error": error,
                "raw": response.content[0].text}

    print(f"  Pass 1 complete — "
          f"confidence: "
          f"{result.get('extraction_metadata', {}).get('confidence')}")
    return result


def extract_notes(company_name, prepared_data):
    """
    Pass 2 — Extracts adjustment-relevant note disclosures.
    Annual reports only — quarterly notes are too sparse
    for full extraction.
    """
    if not prepared_data.get("full_document"):
        print(f"  Pass 2 skipped — quarterly report, "
              f"notes not detailed enough")
        return None

    report_info = prepared_data["report_info"]
    boundaries  = prepared_data.get("boundaries") or {}
    metadata    = (prepared_data.get("sections") or {}).get(
        "metadata", {}
    ) or {}

    currency = (
        boundaries.get("currency") or
        metadata.get("currency") or
        "NOK"
    )
    unit = (
        boundaries.get("unit") or
        metadata.get("unit") or
        "millions"
    )

    text = prepared_data["sections"].get("notes_financial", "")

    if not text:
        print(f"  No notes text available")
        return None

    prompt = NOTES_PROMPT.format(
        company=company_name,
        period=report_info["period"],
        accounting_standard=report_info["accounting_standard"],
        currency=currency,
        unit=unit
    )

    print(f"  Pass 2: extracting notes...")

    response = client.messages.create(
        model=MODELS["extractor"],
        max_tokens=6000,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nNOTES TEXT:\n{text}"
        }]
    )

    result, error = parse_json_response(
        response.content[0].text
    )

    if error:
        print(f"  Parse error: {error}")
        return {"error": error,
                "raw": response.content[0].text}

    print(f"  Pass 2 complete — "
          f"confidence: "
          f"{result.get('extraction_metadata', {}).get('confidence')}")
    return result


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def extract_all(company_name, prepared_data):
    """
    Runs both extraction passes and saves results.

    Returns dict with:
        financials: Pass 1 output
        notes:      Pass 2 output (annual only)
        saved_to:   list of saved file paths
    """
    report_info = prepared_data["report_info"]
    saved = []

    # Pass 1 — financial statements
    financials = extract_financial_statements(
        company_name, prepared_data
    )
    if financials and "error" not in financials:
        filename = build_output_filename(
            report_info, "financials"
        )
        path = save_json(financials, company_name, filename)
        saved.append(path)

    # Pass 2 — notes (annual only)
    notes = extract_notes(company_name, prepared_data)
    if notes and "error" not in notes:
        filename = build_output_filename(report_info, "notes")
        path = save_json(notes, company_name, filename)
        saved.append(path)

    return {
        "financials": financials,
        "notes":      notes,
        "saved_to":   saved
    }