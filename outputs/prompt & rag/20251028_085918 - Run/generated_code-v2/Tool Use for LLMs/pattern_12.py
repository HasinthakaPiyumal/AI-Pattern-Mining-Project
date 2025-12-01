"""
This script defines the main orchestrator for the TORA-Finance Advisor application.
It simulates user interaction and demonstrates the flow of the TORA agent.
"""

from tora_agent import ToraFinanceAgent

def main():
    print("Welcome to TORA-Finance Advisor!\n")
    agent = ToraFinanceAgent()

    # Example 1: Retirement Planning
    user_query_1 = "I want to save for retirement. I am 30 years old, have $100,000 in savings, and can contribute $1,000 per month. I want to retire at 65 and need an annual income of $60,000 in today's money. My risk tolerance is moderate."
    print(f"User Query: {user_query_1}\n")
    financial_plan_1 = agent.reason(user_query_1)
    print("\n--- Financial Plan (Retirement) ---")
    print(financial_plan_1)
    print("\n" + "="*80 + "\n")

    # Example 2: Investment Growth Analysis
    user_query_2 = "I have $50,000 and want to invest it for 10 years with an expected annual return of 7%. What will be its future value?"
    print(f"User Query: {user_query_2}\n")
    financial_plan_2 = agent.reason(user_query_2)
    print("\n--- Financial Plan (Investment Growth) ---")
    print(financial_plan_2)
    print("\n" + "="*80 + "\n")

    # Example 3: Portfolio Recommendation (simplified)
    user_query_3 = "I have $200,000 to invest, I'm aggressive, and my goal is wealth maximization."
    print(f"User Query: {user_query_3}\n")
    financial_plan_3 = agent.reason(user_query_3)
    print("\n--- Financial Plan (Portfolio Recommendation) ---")
    print(financial_plan_3)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
