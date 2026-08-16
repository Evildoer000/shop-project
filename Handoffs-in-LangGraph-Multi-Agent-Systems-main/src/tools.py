"""Domain-specific tools for real estate agents.

This module provides three specialized tools:
- calculate_mortgage_affordability: For supervisor agent
- calculate_price_per_sqft: For transaction history agent
- calculate_remaining_lease: For property profile agent
"""

from langchain.tools import tool


@tool
def calculate_mortgage_affordability(
    monthly_income: float, interest_rate: float, loan_years: int = 30
) -> dict:
    """Calculate maximum affordable mortgage and monthly payment.

    Args:
        monthly_income: Monthly gross income in SGD
        interest_rate: Annual interest rate (e.g., 3.5 for 3.5%)
        loan_years: Loan duration in years (default: 30)
    """
    print(
        f"🔧 [MORTGAGE] calculate_mortgage_affordability(income=${monthly_income:,.0f}, rate={interest_rate}%)"
    )

    # 30% of income rule for monthly payment
    max_monthly_payment = monthly_income * 0.30
    monthly_rate = (interest_rate / 100) / 12
    num_payments = loan_years * 12

    # Calculate max loan amount using mortgage formula
    max_loan = max_monthly_payment * (
        (1 - (1 + monthly_rate) ** -num_payments) / monthly_rate
    )

    result = {
        "max_loan_amount": round(max_loan, 2),
        "max_monthly_payment": round(max_monthly_payment, 2),
        "loan_term_years": loan_years,
    }

    print(
        f"   → Max loan: ${result['max_loan_amount']:,.0f} (${result['max_monthly_payment']:,.0f}/month)"
    )
    return result


@tool
def calculate_price_per_sqft(total_price: float, size_sqft: float) -> dict:
    """Calculate price per square foot for property valuation."""
    print(
        f"🔧 [VALUATION] calculate_price_per_sqft(${total_price:,.0f}, {size_sqft} sqft)"
    )

    price_per_sqft = total_price / size_sqft
    tier = (
        "Premium"
        if price_per_sqft > 2500
        else "High-End"
        if price_per_sqft > 1800
        else "Mid-Range"
        if price_per_sqft > 1200
        else "Affordable"
    )

    result = {"price_per_sqft": round(price_per_sqft, 2), "tier": tier}
    print(f"   → ${result['price_per_sqft']}/sqft ({tier})")
    return result


@tool
def calculate_remaining_lease(
    lease_start_year: int, lease_duration: int = 99, current_year: int = 2025
) -> dict:
    """Calculate remaining lease years for leasehold properties."""
    print(
        f"🔧 [LEASE] calculate_remaining_lease(start={lease_start_year}, duration={lease_duration}yr)"
    )

    remaining = lease_duration - (current_year - lease_start_year)
    status = (
        "Excellent"
        if remaining > 80
        else "Good"
        if remaining > 60
        else "Fair"
        if remaining > 40
        else "Short"
    )

    result = {"remaining_years": max(0, remaining), "status": status}
    print(f"   → {result['remaining_years']} years remaining ({status})")
    return result
