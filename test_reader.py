from utils.pdf_reader import prepare_for_extraction
from agents.extractor import extract_all

# Test on annual report first
prepared = prepare_for_extraction("aa_energi", "annual_2024.pdf")

if prepared:
    print("\n--- RUNNING EXTRACTION ---")
    result = extract_all("aa_energi", prepared)

    # Show key figures from Pass 1
    if result["financials"] and "error" not in result["financials"]:
        print("\n--- PASS 1 KEY FIGURES ---")
        try:
            is_ = result["financials"]["income_statement"]
            cp  = is_["current_period"]
            pp  = is_["prior_period"]

            print(f"Period:           {cp['label']}")
            print(f"Revenue:          {cp['revenue_reported']}")
            print(f"EBITDA reported:  {cp['ebitda_reported']}")
            print(f"EBITDA underlying:{cp['ebitda_underlying']}")
            print(f"EBIT:             {cp['ebit_reported']}")
            print(f"Finance costs:    {cp['finance_costs_reported']}")
            print(f"PBT:              {cp['profit_before_tax']}")
            print(f"Net income:       {cp['majority_net_income']}")
            print(f"\nPrior period:     {pp['label']}")
            print(f"Revenue:          {pp['revenue_reported']}")
            print(f"EBITDA reported:  {pp['ebitda_reported']}")

            bs = result["financials"]["balance_sheet"]
            ce = bs["current_period_end"]
            print(f"\nBalance sheet ({ce['label']}):")
            print(f"Cash:             {ce['cash_and_equivalents']}")
            print(f"Total assets:     {ce['total_assets']}")
            print(f"Interest-bearing debt (ST): {ce['short_term_debt']}")
            print(f"Interest-bearing debt (LT): {ce['long_term_debt']}")
            print(f"Lease liabilities (LT): "
                  f"{ce['non_current_lease_liabilities']}")
            print(f"Pension liability:{ce['pension_liability']}")
            print(f"Total equity:     {ce['total_equity']}")

            cf = result["financials"]["cash_flow"]
            cc = cf["current_period"]
            print(f"\nCash flow ({cc['label']}):")
            print(f"CFO:              {cc['cfo_reported']}")
            print(f"Capex:            {cc['capex']}")
            print(f"Interest paid:    {cc['interest_paid']}")
            print(f"Tax paid:         {cc['tax_paid']}")
            print(f"Dividends paid:   {cc['dividends_paid_to_shareholders']}")

        except Exception as e:
            print(f"Display error: {e}")

    # Show key figures from Pass 2
    if result["notes"] and "error" not in result["notes"]:
        print("\n--- PASS 2 KEY FIGURES ---")
        try:
            notes = result["notes"]

            fc = notes["finance_costs_breakdown"]["current_period"]
            print(f"Finance costs breakdown:")
            print(f"  Core debt interest:    "
                  f"{fc['interest_on_financial_debt']}")
            print(f"  Lease interest:        "
                  f"{fc['interest_on_lease_liabilities']}")
            print(f"  Pension interest:      "
                  f"{fc['interest_on_pension_obligations']}")
            print(f"  FX gains/losses:       "
                  f"{fc['fx_gains_losses']}")
            print(f"  FV movements:          "
                  f"{fc['fair_value_movements_derivatives']}")
            print(f"  Interest income:       "
                  f"{fc['interest_income']}")

            pen = notes["pension"]["current_period_end"]
            print(f"\nPension:")
            print(f"  DBO:                   "
                  f"{pen['defined_benefit_obligation']}")
            print(f"  Plan assets:           "
                  f"{pen['plan_assets_fair_value']}")
            print(f"  Net deficit:           "
                  f"{pen['net_pension_deficit']}")
            print(f"  Interest cost:         "
                  f"{pen['interest_cost_on_obligation']}")

            debt = notes["debt_schedule"]["current_period_end"]
            print(f"\nDebt schedule:")
            print(f"  Total debt:            "
                  f"{debt['total_interest_bearing_debt']}")
            print(f"  Bonds:                 "
                  f"{debt['bonds_issued']}")
            print(f"  Avg interest rate:     "
                  f"{debt['weighted_average_interest_rate_pct']}%")
            print(f"  Undrawn facilities:    "
                  f"{debt['undrawn_credit_facilities']}")
            print(f"  Next bond maturity:    "
                  f"{debt['next_bond_maturity_date']} "
                  f"({debt['next_bond_maturity_amount']})")

            segs = notes["segments"]["segments"]
            if segs:
                print(f"\nSegments:")
                for s in segs:
                    cp = s["current_period"]
                    print(f"  {s['name']}: "
                          f"Rev {cp['revenue']}, "
                          f"EBITDA {cp['ebitda']}, "
                          f"Capex {cp['capex']}")

        except Exception as e:
            print(f"Display error: {e}")

    print(f"\nFiles saved to:")
    for path in result["saved_to"]:
        print(f"  {path}")