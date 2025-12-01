import json
import re
import io
import contextlib
import sys
import numpy as np
import pandas as pd

class CoreLLMOrchestrator:
    def _extract_parameters(self, user_query):
        params = {
            "assets": [],
            "initial_investment": 0.0,
            "contributions_per_month": 0.0,
            "time_horizon_years": 0,
            "risk_tolerance": "medium",
            "target_return": 0.07
        }

        asset_match = re.findall(r"\b([A-Z]{2,5})\b", user_query)
        if asset_match:
            params["assets"] = list(set(asset_match))

        investment_match = re.search(r"initial investment of \$?([\d,]+\.?\d*)", user_query)
        if investment_match:
            params["initial_investment"] = float(investment_match.group(1).replace(",", ""))

        contributions_match = re.search(r"monthly contributions of \$?([\d,]+\.?\d*)", user_query)
        if contributions_match:
            params["contributions_per_month"] = float(contributions_match.group(1).replace(",", ""))

        time_horizon_match = re.search(r"over (\d+) years", user_query)
        if time_horizon_match:
            params["time_horizon_years"] = int(time_horizon_match.group(1))

        return_match = re.search(r"(\d+\.?\d*)% return", user_query)
        if return_match:
            params["target_return"] = float(return_match.group(1)) / 100.0

        risk_match = re.search(r"(minimum|low|medium|high) risk", user_query)
        if risk_match:
            params["risk_tolerance"] = risk_match.group(1)

        return params

    def _generate_llm_code(self, financial_parameters):
        assets_str = json.dumps(financial_parameters["assets"])

        code_template = f"""
import json
import sys
import numpy as np
import pandas as pd

class FinancialCalculationModule:
    def get_simulated_asset_data(self, assets):
        num_assets = len(assets)
        num_days = 252 * 5
        np.random.seed(42)
        daily_returns = np.random.normal(0.0001, 0.002, (num_days, num_assets))
        for i in range(num_assets):
            daily_returns[:, i] += np.random.uniform(-0.0005, 0.0005)
        asset_returns_df = pd.DataFrame(daily_returns, columns=assets)
        return asset_returns_df

    def optimize_portfolio(self, assets, initial_investment, contributions_per_month, time_horizon_years, risk_tolerance, target_return):
        asset_data = self.get_simulated_asset_data(assets)
        mean_returns = asset_data.mean() * 252
        cov_matrix = asset_data.cov() * 252

        num_assets = len(assets)
        if num_assets == 0:
            return {{"error": "No assets provided for optimization."}}

        weights = np.ones(num_assets) / num_assets
        if target_return > 0:
            sorted_assets = mean_returns.sort_values(ascending=False).index.tolist()
            for i, asset in enumerate(sorted_assets):
                weights[assets.index(asset)] += (num_assets - 1 - i) * 0.05
            weights = weights / np.sum(weights)

        portfolio_return = np.sum(weights * mean_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        monthly_return = (1 + portfolio_return)**(1/12) - 1
        total_months = time_horizon_years * 12
        future_value = initial_investment
        for _ in range(total_months):
            future_value = future_value * (1 + monthly_return) + contributions_per_month

        return {{
            "optimal_weights": {{assets[i]: float(weights[i]) for i in range(num_assets)}},
            "portfolio_expected_annual_return": float(portfolio_return),
            "portfolio_annual_std_dev": float(portfolio_std_dev),
            "projected_future_value": float(future_value),
            "time_horizon_years": time_horizon_years
        }}

    def project_future_value(self, initial_investment, annual_return, time_horizon_years, contributions_per_month=0):
        monthly_return = (1 + annual_return)**(1/12) - 1
        total_months = time_horizon_years * 12
        future_value = initial_investment
        for _ in range(total_months):
            future_value = future_value * (1 + monthly_return) + contributions_per_month
        return float(future_value)

    def calculate_risk_metrics(self, portfolio_weights, asset_returns):
        return {{"risk_score": 0.5}}

try:
    calc_module = FinancialCalculationModule()
    result = calc_module.optimize_portfolio(
        assets={assets_str},
        initial_investment={financial_parameters["initial_investment"]},
        contributions_per_month={financial_parameters["contributions_per_month"]},
        time_horizon_years={financial_parameters["time_horizon_years"]},
        risk_tolerance="{financial_parameters["risk_tolerance"]}",
        target_return={financial_parameters["target_return"]}
    )
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
"""
        return code_template

    def _execute_code(self, code_string):
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        execution_results = {}
        try:
            exec(code_string, {}, {})
            execution_output = redirected_output.getvalue()
            execution_results = json.loads(execution_output)
        except Exception as e:
            execution_results = {"error": f"Code execution failed: {str(e)}"}
        finally:
            sys.stdout = old_stdout
        return execution_results

    def _formulate_final_response(self, user_query, computational_results):
        if "error" in computational_results:
            return f"I encountered an error while processing your request: {computational_results['error']}. Please try again."

        response_parts = [
            f"Based on your query regarding '{user_query}', here's an analysis of your investment portfolio:"
        ]

        if "optimal_weights" in computational_results:
            response_parts.append("Optimal Portfolio Allocation (Approximate Weights):")
            for asset, weight in computational_results["optimal_weights"].items():
                response_parts.append(f"- {asset}: {weight:.2%}")

        if "portfolio_expected_annual_return" in computational_results:
            response_parts.append(f"Projected Annual Return: {computational_results['portfolio_expected_annual_return']:.2%}")

        if "portfolio_annual_std_dev" in computational_results:
            response_parts.append(f"Annual Standard Deviation (Risk): {computational_results['portfolio_annual_std_dev']:.2%}")

        if "projected_future_value" in computational_results and "time_horizon_years" in computational_results:
            response_parts.append(
                f"Projected Future Value in {computational_results['time_horizon_years']} years: "
                f"${computational_results['projected_future_value']:.2f}"
            )

        response_parts.append("Please note: This is a simplified demonstration and not actual financial advice. Real investment decisions should be based on thorough research and professional consultation.")

        return "\n".join(response_parts)

    def process_financial_query(self, user_query):
        financial_parameters = self._extract_parameters(user_query)
        generated_code = self._generate_llm_code(financial_parameters)
        computational_results = self._execute_code(generated_code)
        final_response = self._formulate_final_response(user_query, computational_results)
        return final_response

if __name__ == "__main__":
    orchestrator = CoreLLMOrchestrator()
    user_query = "Optimize my portfolio for a 7% return with minimum risk given these 5 ETFs: SPY, QQQ, VOO, VTI, ARKK, an initial investment of $10,000, and monthly contributions of $500 over 20 years."
    response = orchestrator.process_financial_query(user_query)
    print(response)