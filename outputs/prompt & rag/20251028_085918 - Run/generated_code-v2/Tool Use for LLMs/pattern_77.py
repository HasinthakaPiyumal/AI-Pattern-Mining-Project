import os
import sys
import importlib.util
import json
import pandas as pd
import requests
from datetime import datetime, timedelta


class ToolRegistry:
    def __init__(self, tools_dir="tools"):
        self.tools_dir = tools_dir
        os.makedirs(self.tools_dir, exist_ok=True)
        self.loaded_tools = {}

    def _get_tool_path(self, tool_name):
        return os.path.join(self.tools_dir, f"{tool_name}.py")

    def save_tool(self, tool_name, code_body):
        tool_path = self._get_tool_path(tool_name)
        with open(tool_path, "w") as f:
            f.write(code_body)
        print(f"Tool '{tool_name}' saved to {tool_path}")

    def load_tool(self, tool_name):
        if tool_name in self.loaded_tools:
            return self.loaded_tools[tool_name]

        tool_path = self._get_tool_path(tool_name)
        if not os.path.exists(tool_path):
            raise FileNotFoundError(f"Tool file for '{tool_name}' not found at {tool_path}")

        spec = importlib.util.spec_from_file_location(tool_name, tool_path)
        if spec is None:
            raise ImportError(f"Could not load spec for tool '{tool_name}'")
        module = importlib.util.module_from_spec(spec)
        sys.modules[tool_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, tool_name):
            raise AttributeError(f"Tool function '{tool_name}' not found in module '{tool_name}'")

        self.loaded_tools[tool_name] = getattr(module, tool_name)
        print(f"Tool '{tool_name}' loaded successfully.")
        return self.loaded_tools[tool_name]

    def get_tool_function(self, tool_name):
        return self.load_tool(tool_name)


class FinancialDataAPIWrapper:
    def __init__(self, api_key="YOUR_ALPHA_VANTAGE_API_KEY"): # Placeholder for a real API key
        self.api_key = api_key
        self.base_url_alpha_vantage = "https://www.alphavantage.co/query"

    def get_historical_prices(self, symbol, outputsize="compact"):
        # Mocking historical prices for demonstration
        print(f"Fetching historical prices for {symbol} (mocked)")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        data = {
            "Open": [100 + i * 0.5 + (i % 5) * 2 for i in range(len(dates))],
            "High": [102 + i * 0.5 + (i % 5) * 2.5 for i in range(len(dates))],
            "Low": [98 + i * 0.5 + (i % 5) * 1.5 for i in range(len(dates))],
            "Close": [101 + i * 0.5 + (i % 5) * 2 for i in range(len(dates))],
            "Volume": [100000 + i * 1000 + (i % 3) * 500 for i in range(len(dates))],
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "Date"
        return df

    def get_news_sentiment(self, symbol, limit=5):
        # Mocking news sentiment for demonstration
        print(f"Fetching news sentiment for {symbol} (mocked)")
        mock_sentiment = [
            {"headline": f"Good news for {symbol}", "sentiment_score": 0.7},
            {"headline": f"{symbol} sees slight dip", "sentiment_score": -0.2},
            {"headline": f"Market positive on {symbol}", "sentiment_score": 0.5},
        ]
        return mock_sentiment[:limit]


class CodeGenerationOrchestrator:
    def __init__(self, tool_registry, financial_api_wrapper):
        self.tool_registry = tool_registry
        self.financial_api_wrapper = financial_api_wrapper

    def _simulate_llm_code_generation(self, prompt):
        # In a real application, this would call an LLM (e.g., via OpenAI API or a local model)
        # to generate Python code based on the prompt.
        print(f"Simulating LLM code generation for prompt: '{prompt}'")

        if "RSI" in prompt and "calculate" in prompt:
            tool_name = "calculate_rsi"
            code = """import pandas as pd

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None
"""
        elif "moving average" in prompt and "simple" in prompt:
            tool_name = "calculate_sma"
            code = """import pandas as pd

def calculate_sma(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    return sma.iloc[-1] if not sma.empty else None
"""
        else:
            tool_name = "generic_tool"
            code = """def generic_tool(data):
    return f"Executed generic tool with data: {data.head() if isinstance(data, pd.DataFrame) else data}"
"""
        return tool_name, code

    def generate_tool_from_nl(self, natural_language_description):
        tool_name, generated_code = self._simulate_llm_code_generation(natural_language_description)
        self.tool_registry.save_tool(tool_name, generated_code)
        return tool_name


class InvestmentRecommendationEngine:
    def __init__(self, tool_registry, financial_api_wrapper):
        self.tool_registry = tool_registry
        self.financial_api_wrapper = financial_api_wrapper

    def analyze_and_recommend(self, symbol, analysis_type="basic"):
        print(f"Performing '{analysis_type}' analysis and recommendation for {symbol}")

        historical_data = self.financial_api_wrapper.get_historical_prices(symbol)

        recommendations = []

        if analysis_type == "basic":
            # Example: Use a generated RSI tool
            try:
                rsi_tool = self.tool_registry.get_tool_function("calculate_rsi")
                current_rsi = rsi_tool(historical_data)
                if current_rsi is not None:
                    recommendations.append(f"Current RSI: {current_rsi:.2f}")
                    if current_rsi < 30:
                        recommendations.append("Recommendation: Likely oversold, consider buying.")
                    elif current_rsi > 70:
                        recommendations.append("Recommendation: Likely overbought, consider selling.")
                    else:
                        recommendations.append("Recommendation: Neutral based on RSI.")
            except (FileNotFoundError, AttributeError, ImportError) as e:
                recommendations.append(f"Could not use RSI tool: {e}")

            # Example: Use a generated SMA tool
            try:
                sma_tool = self.tool_registry.get_tool_function("calculate_sma")
                current_sma = sma_tool(historical_data)
                if current_sma is not None:
                    recommendations.append(f"Current 20-day SMA: {current_sma:.2f}")
                    if historical_data['Close'].iloc[-1] > current_sma:
                        recommendations.append("Recommendation: Price above SMA, bullish sign.")
                    else:
                        recommendations.append("Recommendation: Price below SMA, bearish sign.")
            except (FileNotFoundError, AttributeError, ImportError) as e:
                recommendations.append(f"Could not use SMA tool: {e}")

            news_sentiment = self.financial_api_wrapper.get_news_sentiment(symbol)
            if news_sentiment:
                avg_sentiment = sum([n['sentiment_score'] for n in news_sentiment]) / len(news_sentiment)
                recommendations.append(f"Average news sentiment: {avg_sentiment:.2f}")
                if avg_sentiment > 0.3:
                    recommendations.append("Recommendation: Positive news sentiment.")
                elif avg_sentiment < -0.3:
                    recommendations.append("Recommendation: Negative news sentiment.")

        return "\n".join(recommendations)


if __name__ == "__main__":
    # 1. Initialize core components
    tool_registry = ToolRegistry()
    financial_api = FinancialDataAPIWrapper()
    code_orchestrator = CodeGenerationOrchestrator(tool_registry, financial_api)
    recommendation_engine = InvestmentRecommendationEngine(tool_registry, financial_api)

    test_symbol = "IBM"

    # 2. Demonstrate Tool Generation from Natural Language
    print("\n--- Demonstrating Tool Generation ---")
    rsi_tool_name = code_orchestrator.generate_tool_from_nl("create a tool to calculate the 14-day RSI for a given stock symbol")
    sma_tool_name = code_orchestrator.generate_tool_from_nl("I need a tool to calculate the simple moving average over 20 days for closing prices")

    # 3. Demonstrate Investment Recommendation using Generated Tools
    print("\n--- Demonstrating Investment Recommendation ---")
    recommendations = recommendation_engine.analyze_and_recommend(test_symbol, analysis_type="basic")
    print(f"\nInvestment Recommendations for {test_symbol}:\n{recommendations}")

    print("\n--- Attempting to load and run an unknown tool (will raise error) ---")
    try:
        tool_registry.load_tool("non_existent_tool")
    except FileNotFoundError as e:
        print(f"Error as expected: {e}")

    # Clean up generated tool files (optional)
    # import shutil
    # if os.path.exists(tool_registry.tools_dir):
    #     shutil.rmtree(tool_registry.tools_dir)
    #     print(f"Cleaned up '{tool_registry.tools_dir}' directory.")
