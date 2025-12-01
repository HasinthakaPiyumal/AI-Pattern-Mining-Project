import pandas as pd
import numpy as np
import scipy.stats as stats
import io
import sys

class DataIngestion:
    def get_market_data(self):
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
        prices = np.random.normal(100, 5, 100).cumsum() + 50
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        return df

    def get_financial_reports(self):
        data = {
            'Year': [2021, 2022, 2023],
            'Revenue': [1000, 1100, 1200],
            'Expenses': [700, 750, 800],
            'Profit': [300, 350, 400]
        }
        df = pd.DataFrame(data)
        return df

    def get_news_sentiment(self):
        news = [
            "Positive outlook for tech stocks this quarter.",
            "Interest rate hike expected next month.",
            "Supply chain disruptions continue."
        ]
        return news

class LLMCore:
    def _generate_rationale(self, market_data, financial_reports, news_sentiment, previous_output=None):
        rationale = """Initial assessment suggests potential volatility due to market sentiment and upcoming economic announcements. Detailed quantitative analysis is needed to confirm risk levels.
        Specifically, we need to calculate daily returns and historical VaR from market data.
        Also, a quick check on profit margins from financial reports is crucial.
        """
        if previous_output:
            rationale += f"\nBased on previous calculation results: {previous_output}, further analysis might include stress testing or scenario planning."
        return rationale

    def _generate_program(self, rationale):
        if "VaR" in rationale and "profit margins" in rationale:
            code = """
import pandas as pd
import numpy as np
import scipy.stats as stats

market_data = pd.DataFrame({'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')), 'Price': np.random.normal(100, 5, 100).cumsum() + 50})
market_data['Daily_Return'] = market_data['Price'].pct_change().dropna()

var_95 = np.percentile(market_data['Daily_Return'], 5) * 100

financial_reports = pd.DataFrame({
    'Year': [2021, 2022, 2023],
    'Revenue': [1000, 1100, 1200],
    'Expenses': [700, 750, 800],
    'Profit': [300, 350, 400]
})
financial_reports['Profit_Margin'] = (financial_reports['Profit'] / financial_reports['Revenue']) * 100

print(f"Calculated 95% Historical VaR: {var_95:.2f}%")
print(f"Latest Profit Margin: {financial_reports['Profit_Margin'].iloc[-1]:.2f}%")
            """
        else:
            code = "print(\"No specific program generated based on current rationale.\")"
        return code

    def _interpret_and_refine(self, program_output, current_rationale):
        refined_rationale = current_rationale
        if "VaR:" in program_output and "Profit Margin:" in program_output:
            var_str = program_output.split("VaR: ")[1].split("%")[0]
            margin_str = program_output.split("Profit Margin: ")[1].split("%")[0]
            try:
                var = float(var_str)
                margin = float(margin_str)
                refined_rationale += f"\n\nInterpretation:\nThe 95% Historical VaR is {var:.2f}%, indicating a potential loss of this magnitude or more in 5% of cases. The latest profit margin is {margin:.2f}%. A lower VaR is generally better, and healthy profit margins are a positive sign. This suggests moderate market risk, but the company's internal profitability remains strong. Further investigation into specific market events impacting VaR is recommended."
            except ValueError:
                refined_rationale += "\n\nInterpretation: Could not parse numerical results from program output."
        else:
            refined_rationale += "\n\nInterpretation: Program output did not contain expected VaR or Profit Margin information."
        return refined_rationale

class ToolExecutionEnvironment:
    def execute_program(self, code):
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            global_scope = {'pd': pd, 'np': np, 'stats': stats}
            local_scope = {}
            exec(code, global_scope, local_scope)
            output = redirected_output.getvalue()
        except Exception as e:
            output = f"Error during program execution: {e}"
        finally:
            sys.stdout = old_stdout
        return output

class FinancialRiskAgent:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.llm_core = LLMCore()
        self.tool_executor = ToolExecutionEnvironment()

    def run_assessment(self, iterations=2):
        market_data = self.data_ingestion.get_market_data()
        financial_reports = self.data_ingestion.get_financial_reports()
        news_sentiment = self.data_ingestion.get_news_sentiment()

        print("--- Initial Data Ingested ---")
        print("Market Data Head:\n", market_data.head())
        print("Financial Reports:\n", financial_reports)
        print("News Sentiment:\n", news_sentiment)
        print("\n" + "="*50 + "\n")

        current_rationale = self.llm_core._generate_rationale(
            market_data, financial_reports, news_sentiment
        )
        print("--- LLM Initial Rationale ---")
        print(current_rationale)
        print("\n" + "="*50 + "\n")

        program_output = None

        for i in range(iterations):
            print(f"--- Iteration {i+1}: Generating and Executing Program ---")
            program_code = self.llm_core._generate_program(current_rationale)
            print("Generated Program:\n", program_code)

            program_output = self.tool_executor.execute_program(program_code)
            print("Program Output:\n", program_output)

            current_rationale = self.llm_core._interpret_and_refine(program_output, current_rationale)
            print("--- LLM Refined Rationale ---")
            print(current_rationale)
            print("\n" + "="*50 + "\n")

        print("--- Final Risk Assessment ---")
        print(current_rationale)

if __name__ == "__main__":
    agent = FinancialRiskAgent()
    agent.run_assessment()
