"""
rule_engine.py — IRS rules, account-type eligibility, savings constraints.
"""


def apply_rules(user_info: dict, top_funds=None) -> dict:
    """
    Post-filter recommendations based on IRS rules and financial constraints.

    Args:
        user_info: age, annual_income, filing_status, has_401k,
                   has_employer_match, has_roth_ira, has_traditional_ira,
                   has_hsa, has_hdhp, emergency_fund_months, total_debt
        top_funds: DataFrame of fund recommendations (optional)

    Returns:
        dict with account_recommendations, warnings, filtered_funds
    """
    recs = []
    warns = []

    age    = user_info.get('age', 30)
    income = user_info.get('annual_income', 50_000)
    filing = user_info.get('filing_status', 'single')

    # ── Emergency fund check ──
    em = user_info.get('emergency_fund_months', 0)
    if em < 3:
        warns.append(
            "Build 3–6 months of expenses in a high-yield savings account "
            "before investing aggressively."
        )
        recs.append("🏦 Priority → High-Yield Savings Account for emergency fund")

    # ── 401(k) employer match ──
    if user_info.get('has_401k') and user_info.get('has_employer_match'):
        recs.append(
            "💰 Priority → Contribute enough to your 401(k) to capture "
            "the full employer match (free money!)"
        )

    # ── Roth IRA eligibility (2024 limits) ──
    if filing == 'single':
        if income > 161_000:
            recs.append("🚫 Roth IRA: Not eligible (income > $161k). Consider Backdoor Roth.")
        elif income > 146_000:
            recs.append("⚠️ Roth IRA: Partial contribution (phase-out $146k–$161k)")
        else:
            limit = "$8,000" if age >= 50 else "$7,000"
            recs.append(f"✅ Roth IRA: Fully eligible — max out at {limit}/yr")
    elif filing == 'married_filing_jointly':
        if income > 240_000:
            recs.append("🚫 Roth IRA: Not eligible (joint income > $240k). Consider Backdoor Roth.")
        elif income > 230_000:
            recs.append("⚠️ Roth IRA: Partial contribution (phase-out $230k–$240k)")
        else:
            recs.append("✅ Roth IRA: Fully eligible for both spouses")

    # ── HSA eligibility ──
    if user_info.get('has_hdhp'):
        recs.append("✅ HSA: Eligible — triple tax advantage, max it out!")
    elif user_info.get('has_hsa'):
        warns.append("HSA requires a High-Deductible Health Plan (HDHP). Verify eligibility.")

    # ── Traditional IRA deduction ──
    if user_info.get('has_401k'):
        if filing == 'single' and income > 87_000:
            recs.append("⚠️ Traditional IRA deduction may be limited (employer plan + high income)")
        elif filing == 'married_filing_jointly' and income > 136_000:
            recs.append("⚠️ Traditional IRA deduction may be limited (employer plan + high income)")
    else:
        recs.append("✅ Traditional IRA: Fully deductible (no employer plan)")

    # ── Tax-efficient fund placement ──
    if top_funds is not None and 'investment_type' in top_funds.columns:
        mf = (top_funds['investment_type'] == 'Mutual Fund').sum()
        if mf > 0:
            recs.append(
                "📋 Tax placement: Hold actively-managed mutual funds in "
                "tax-advantaged accounts (401k/IRA). Use ETFs in taxable accounts."
            )

    # ── Age-based nudge ──
    if age >= 55:
        recs.append("🕐 Near retirement: Consider shifting 10–20% more toward bonds / stable value.")
    elif age < 30:
        recs.append("🚀 Long horizon: Higher equity allocation is appropriate — time is on your side.")

    # ── Debt check ──
    debt = user_info.get('total_debt', 0)
    if debt > income * 0.5:
        warns.append("High debt-to-income ratio. Consider paying down debt before aggressive investing.")

    return {
        'account_recommendations': recs,
        'warnings': warns,
        'filtered_funds': top_funds,
    }
