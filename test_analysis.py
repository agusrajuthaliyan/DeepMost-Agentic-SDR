"""Test the context-aware logistic probability model across different company types."""
import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from src.agent_logic import _score_company_fit, generate_synthetic_call, analyze_call, get_provider_info

info = get_provider_info()
print(f"Provider: {info['provider']} | Model: {info['model']}")
print("=" * 70)

# Test scoring across different company profiles
test_contexts = {
    "Perfect Fit (data company with pain)": 
        "DataFlow Corp processes 50TB of data daily using legacy ETL pipelines that are slow and inefficient. "
        "They spend $200K/year on manual data processing. The CTO is actively looking for modern data solutions "
        "to replace their outdated infrastructure. They're evaluating several enterprise analytics platforms "
        "and want to automate their reporting and dashboard workflows. Revenue $50M, 500 employees globally.",
    
    "Good Fit (tech company, some signals)":
        "TechVibe is a SaaS company with 200 employees building project management tools. They use analytics "
        "heavily and are looking to scale their data pipeline. Annual revenue $20M with Series B funding.",
    
    "Moderate Fit (generic tech)":
        "CloudBase provides cloud hosting and infrastructure services to small businesses. "
        "They have a growing customer base and are expanding their platform capabilities.",
    
    "Poor Fit (no data needs)":
        "FreshBake is a local bakery chain with 15 stores. They use simple POS systems and have no "
        "significant data processing needs. Small team of 50 employees, mostly in-store staff.",
    
    "Anti-Fit (already solved)":
        "IntelliSoft is already using Snowflake and dbt for their data pipeline. They are satisfied "
        "with their current setup and not looking for new solutions. Happy with their vendor.",
}

print("\n--- FIT SCORING ACROSS COMPANY PROFILES ---\n")
print(f"{'Profile':<40} {'Score':>6} {'P(Success)':>11} {'Signals'}")
print("-" * 90)

for name, ctx in test_contexts.items():
    fit = _score_company_fit(ctx)
    signals = fit['signals']
    sig_str = f"pain={signals['pain_points']} data={signals['data_relevance']} need={signals['active_need']} budget={signals['budget_capacity']} anti={signals['anti_fit']}"
    print(f"{name:<40} {fit['fit_score']:>5}/10 {fit['success_probability']:>10.0%}   {sig_str}")

# Now run 2 actual LLM calls — one high-fit, one low-fit
print(f"\n\n{'=' * 70}")
print("LIVE TEST: Generate + Analyze (2 calls)")
print("=" * 70)

for name in ["Perfect Fit (data company with pain)", "Poor Fit (no data needs)"]:
    ctx = test_contexts[name]
    print(f"\n--- {name} ---")
    dialogue = generate_synthetic_call(ctx)
    if dialogue:
        sentiment, outcome, score, objection, feedback = analyze_call(dialogue)
        print(f"   Result: {outcome} | Score: {score} | Objection: {objection} | Sentiment: {sentiment}")
        print(f"   Feedback: {feedback[:80]}...")
    else:
        print("   FAILED to generate")

print(f"\n{'=' * 70}")
print("Done! The fit score directly influences success probability.")
