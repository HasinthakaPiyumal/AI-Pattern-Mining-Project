
import pandas as pd
import numpy as np
import io
import contextlib

def simulate_llm_code_generation(user_input):
    parts = user_input.split("stocks: ")
    stock_symbols = ["AAPL", "MSFT", "GOOG"]

    if len(parts) > 1:
        try:
            parsed_symbols = [s.strip().upper() for s in parts[1].split(",") if s.strip()]
            if parsed_symbols:
                stock_symbols = parsed_symbols
        except:
            pass

    code_template = """
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, efficient_frontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import datetime

np.random.seed(42)

stock_symbols = {stock_symbols_list}
num_days = 252 * 3
dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(num_days)]
price_data = {{}}
for symbol in stock_symbols:
    prices = 100 + np.cumsum(np.random.randn(num_days)) * 0.5
    price_data[symbol] = prices
df_prices = pd.DataFrame(price_data, index=dates)

latest_prices = get_latest_prices(df_prices)

mu = expected_returns.ema_returns(df_prices)
S = risk_models.exp_cov(df_prices)

ef = efficient_frontier.EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("---OPTIMIZATION_RESULTS_START---")
print(f"Optimal Weights: {{cleaned_weights}}")
print(f"Expected Annual Return: {{ef.portfolio_return:.4f}}")
print(f"Annual Volatility: {{ef.portfolio_volatility:.4f}}")
print(f"Sharpe Ratio: {{ef.sharpe_ratio:.4f}}")
print("---OPTIMIZATION_RESULTS_END---")
"""
    generated_code = code_template.format(stock_symbols_list=stock_symbols)
    return generated_code

def execute_generated_code(code_string):
    output_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_capture):
            exec(code_string, {}, {})
        return output_capture.getvalue()
    except Exception as e:
        return f"ERROR during code execution: {e}\n{output_capture.getvalue()}"

def parse_financial_results(raw_output):
    results = {}
    if "---OPTIMIZATION_RESULTS_START---" in raw_output and "---OPTIMIZATION_RESULTS_END---" in raw_output:
        start_idx = raw_output.find("---OPTIMIZATION_RESULTS_START---") + len("---OPTIMIZATION_RESULTS_START---")
        end_idx = raw_output.find("---OPTIMIZATION_RESULTS_END---")
        relevant_output = raw_output[start_idx:end_idx].strip()

        for line in relevant_output.split("\n"):
            if "Optimal Weights:" in line:
                weights_str = line.replace("Optimal Weights: ", "").strip()
                try:
                    results["Optimal Weights"] = eval(weights_str)
                except:
                    results["Optimal Weights"] = {}
            elif "Expected Annual Return:" in line:
                try:
                    results["Expected Annual Return"] = float(line.replace("Expected Annual Return: ", "").strip())
                except ValueError:
                    results["Expected Annual Return"] = 0.0
            elif "Annual Volatility:" in line:
                try:
                    results["Annual Volatility"] = float(line.replace("Annual Volatility: ", "").strip())
                except ValueError:
                    results["Annual Volatility"] = 0.0
            elif "Sharpe Ratio:" in line:
                try:
                    results["Sharpe Ratio"] = float(line.replace("Sharpe Ratio: ", "").strip())
                except ValueError:
                    results["Sharpe Ratio"] = 0.0
    elif "ERROR" in raw_output:
        results["Error"] = raw_output
    return results

def simulate_llm_explanation_generation(user_input, parsed_results):
    explanation = f"Based on your input: \"{user_input}\", here is the financial portfolio optimization result:\n\n"

    if "Error" in parsed_results:
        explanation += f"An error occurred during the optimization process: {parsed_results['Error']}\n"
        explanation += "Please check your input or the generated code for issues. Ensure 'PyPortfolioOpt' and its dependencies are installed in the execution environment."
        return explanation

    if parsed_results and parsed_results.get("Optimal Weights"):
        explanation += "Optimized Portfolio Recommendations:\n"
        explanation += "- **Optimal Asset Allocation:**\n"
        total_weight = sum(parsed_results["Optimal Weights"].values())
        if total_weight > 0:
            for asset, weight in parsed_results["Optimal Weights"].items():
                explanation += f"  - {asset}: {(weight / total_weight) * 100:.2f}%\n"
        else:
             explanation += "  - No specific asset allocation could be determined.\n"

        explanation += f"- **Projected Annual Return:** {parsed_results.get('Expected Annual Return', 0.0) * 100:.2f}%\n"
        explanation += f"- **Annual Volatility (Risk):** {parsed_results.get('Annual Volatility', 0.0) * 100:.2f}%\n"
        explanation += f"- **Sharpe Ratio (Risk-Adjusted Return):** {parsed_results.get('Sharpe Ratio', 0.0):.2f}\n\n"
        explanation += "Rationale:\n"
        explanation += "The portfolio was optimized to maximize the Sharpe Ratio, aiming for the highest possible return for a given level of risk. The recommended allocation balances growth potential with risk mitigation, as indicated by the projected return, volatility, and Sharpe Ratio. A higher Sharpe Ratio generally indicates a better risk-adjusted return.\n\n"
        explanation += "Consider reviewing these recommendations with a financial advisor, as market conditions and individual circumstances can vary. This optimization uses historical data and does not guarantee future performance."
    else:
        explanation += "No valid optimization results could be generated or parsed. This might be due to insufficient or invalid input, or issues during the financial computation. Please ensure your request is clear and provides sufficient information for optimization."

    return explanation

def main():
    user_input = "Optimize my portfolio for maximum returns with moderate risk using these stocks: AAPL, MSFT, GOOG"

    generated_python_code = simulate_llm_code_generation(user_input)
    execution_output = execute_generated_code(generated_python_code)
    parsed_results = parse_financial_results(execution_output)
    final_explanation = simulate_llm_explanation_generation(user_input, parsed_results)

    print(final_explanation)

if __name__ == "__main__":
    main()
