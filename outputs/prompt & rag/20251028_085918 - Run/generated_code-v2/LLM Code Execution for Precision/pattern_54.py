from llm_prompt_generator import generate_financial_script
from financial_calculator import calculate_portfolio_optimization # In a real PAL setup, this would be executed externally
from recommendation_generator import generate_recommendations

def run_personal_finance_advisor(user_data):
    print("--- Step 1: User Input ---")
    user_goals = user_data["goals"]
    risk_tolerance = user_data["risk_tolerance"]
    current_holdings = user_data["holdings"] # Example: {'AAPL': 1000, 'GOOG': 500}

    print(f"User Goals: {user_goals}")
    print(f"Risk Tolerance: {risk_tolerance}")
    print(f"Current Holdings (values): {current_holdings}")

    print("\n--- Step 2: LLM Generates Code (Simulated) ---")
    # In a real PAL system, an LLM would generate this script based on the prompt.
    # Here, we simulate it by directly calling the function that *would* be generated.
    # The actual 'script' generated would look something like:
    # "import financial_calculator; result = financial_calculator.calculate_portfolio_optimization(...)"
    
    # For this example, we'll directly pass simplified data to the calculator.
    # In a full system, the LLM would interpret user_data and formulate appropriate inputs
    # for the financial calculation functions.

    # Simulate market data for portfolio optimization (replace with real data in a real app)
    assets = list(current_holdings.keys()) if current_holdings else ['AssetA', 'AssetB'] # Example assets
    expected_returns_sim = {'AssetA': 0.12, 'AssetB': 0.08, 'AssetC': 0.15}
    cov_matrix_sim = [
        [0.01, 0.005, 0.002],
        [0.005, 0.02, 0.003],
        [0.002, 0.003, 0.015]
    ] # Example covariance matrix
    num_portfolios_sim = 10000

    # This part simulates the LLM's role in *deciding what code to generate* and *what parameters to use*.
    # For simplicity, we directly call the function with example data.
    print("Simulating LLM generating and executing financial calculation code...")
    try:
        optimal_portfolio_results = calculate_portfolio_optimization(
            expected_returns=expected_returns_sim,
            cov_matrix=cov_matrix_sim,
            num_portfolios=num_portfolios_sim
        )
        print("Financial calculations executed successfully.")
        print("Optimal Portfolio Details: ", optimal_portfolio_results)
    except Exception as e:
        print(f"Error during financial calculation: {e}")
        optimal_portfolio_results = None

    print("\n--- Step 3: LLM Synthesizes Recommendations ---")
    if optimal_portfolio_results:
        final_recommendations = generate_recommendations(
            optimal_portfolio_results,
            user_goals,
            risk_tolerance
        )
        print("\nFinal Personalized Recommendations:")
        print(final_recommendations)
    else:
        print("Could not generate recommendations due to calculation errors.")

if __name__ == "__main__":
    # Example User Data
    user_profile = {
        'goals': 'Retirement in 20 years, save for a down payment in 5 years.',
        'risk_tolerance': 'Medium',
        'holdings': {'AssetA': 5000, 'AssetB': 3000}
    }
    run_personal_finance_advisor(user_profile)

    print("\n" + "="*50 + "\n")

    user_profile_high_risk = {
        'goals': 'Aggressive growth, maximizing returns.',
        'risk_tolerance': 'High',
        'holdings': {'AssetA': 2000, 'AssetC': 8000}
    }
    run_personal_finance_advisor(user_profile_high_risk)