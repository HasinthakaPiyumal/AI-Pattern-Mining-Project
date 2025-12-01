import yfinance as yf
import pandas as pd
import numpy as np
import backtrader as bt
import quantstats as qs
import io
import sys
import importlib.util
import os

def get_historical_data(ticker, start_date, end_date, interval="1d"):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if not data.empty:
        data.index = pd.to_datetime(data.index)
        data = data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        data.columns = [col.lower().replace(" ", "_") for col in data.columns]
    return data

def calculate_indicators(df, indicators):
    if df.empty:
        return df
    df_copy = df.copy()
    if "sma_20" in indicators:
        df_copy["sma_20"] = df_copy["close"].rolling(window=20).mean()
    if "rsi_14" in indicators:
        delta = df_copy["close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss
        df_copy["rsi_14"] = 100 - (100 / (1 + rs))
    return df_copy

def resample_data(df, period):
    if df.empty:
        return df
    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "adj_close": "last",
        "volume": "sum"
    }
    return df.resample(period).apply(ohlcv).dropna()

class SimpleMovingAverageStrategy(bt.Strategy):
    params = (('sma_period', 20),)

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.sma_period)
        self.crossover = bt.indicators.CrossOver(self.data.close, self.sma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

def define_strategy(strategy_name, params=None):
    if strategy_name == "SimpleMovingAverage":
        class DynamicSMAStrategy(SimpleMovingAverageStrategy):
            pass
        if params:
            DynamicSMAStrategy.params = tuple((k, v) for k, v in params.items())
        return DynamicSMAStrategy
    return None

def run_backtest(strategy_class, data, cash=100000.0, commission=0.001):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    
    cerebro.addstrategy(strategy_class)
    
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    results = cerebro.run()
    
    strat_returns_data = results[0].analyzers.returns.get_analysis()
    strat_returns = pd.Series(strat_returns_data).fillna(0)
    
    return {
        "cerebro_results": results,
        "returns": strat_returns,
        "sharpe_ratio": results[0].analyzers.sharpe.get_analysis().get('sharperatio', 0.0),
        "max_drawdown": results[0].analyzers.drawdown.get_analysis().get('maxdrawdown', 0.0)
    }

def analyze_backtest_results(returns_series, title="Strategy Performance"):
    if not isinstance(returns_series.index, pd.DatetimeIndex):
        returns_series.index = pd.to_datetime(returns_series.index)
        
    qs_report_str = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = qs_report_str
    try:
        qs.reports.full(returns_series, benchmark=None, display=False, title=title)
    finally:
        sys.stdout = old_stdout
    return qs_report_str.getvalue()

def optimize_strategy_parameters(strategy_class, param_ranges, data, cash, commission, target_metric="sharpe"):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    
    cerebro.optstrategy(strategy_class, **param_ranges)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    optimized_runs = cerebro.run(maxcpus=1)

    best_params = None
    best_metric_value = -float('inf') if target_metric in ["sharpe", "total_return"] else float('inf')

    for run_group in optimized_runs:
        for strategy in run_group:
            p = strategy.p
            sharpe_ratio = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
            max_drawdown = strategy.analyzers.drawdown.get_analysis().get('maxdrawdown', 0.0)
            total_return_pct = strategy.analyzers.returns.get_analysis().get('rtot', 0.0) * 100

            current_metric_value = None
            if target_metric == "sharpe":
                current_metric_value = sharpe_ratio
            elif target_metric == "max_drawdown":
                current_metric_value = max_drawdown
            elif target_metric == "total_return":
                current_metric_value = total_return_pct
            
            if current_metric_value is not None:
                if target_metric in ["sharpe", "total_return"]:
                    if current_metric_value > best_metric_value:
                        best_metric_value = current_metric_value
                        best_params = {k: getattr(p, k) for k in p._getkwargs()}
                elif target_metric == "max_drawdown":
                    if current_metric_value < best_metric_value:
                        best_metric_value = current_metric_value
                        best_params = {k: getattr(p, k) for k in p._getkwargs()}
    
    return {"best_params": best_params, "best_metric_value": best_metric_value, "target_metric": target_metric}

class LLMAgent:
    def __init__(self, tools):
        self.tools = tools
        self.history = []
        self.current_context_data = None
        self.current_context_strategy_class = None
        self.current_context_backtest_report = None
        self.current_sharpe_ratio = 0.0

    def _simulate_reasoning(self, prompt, context=None):
        reasoning = f"LLM Reasoning for: '{prompt}'\n"
        if context:
            reasoning += f"Context Summary: {context}\n"
        
        if "generate a strategy" in prompt.lower() and self.current_context_data is None:
            reasoning += "Based on the goal, I need to fetch data first. Then I will propose a Simple Moving Average strategy and backtest it."
        elif "backtest results" in prompt.lower() and self.current_sharpe_ratio < 0.8 and self.current_sharpe_ratio != 0.0:
            reasoning += f"The strategy's Sharpe Ratio ({self.current_sharpe_ratio:.2f}) is low. I will try to optimize its parameters to improve performance."
        elif "backtest results" in prompt.lower() and self.current_sharpe_ratio >= 0.8:
            reasoning += "The strategy performance looks satisfactory with a good Sharpe Ratio. I will provide a final report."
        elif "optimize parameters" in prompt.lower():
            reasoning += "I will perform parameter optimization for the SMA period to maximize Sharpe Ratio, then re-backtest the optimized strategy."
        elif "optimized strategy re-backtested" in prompt.lower():
            reasoning += "The strategy has been optimized and re-backtested. I will now present the final report."
        else:
            reasoning += "Proceeding with next logical step based on current state."
        
        self.history.append({"role": "llm", "content": reasoning})
        return reasoning

    def _simulate_tool_call(self, tool_name, *args, **kwargs):
        tool_call_str = f"LLM Calling Tool: {tool_name}({', '.join(f'{k}={repr(v)}' for k, v in kwargs.items())})"
        self.history.append({"role": "llm", "content": tool_call_str})
        
        tool_func = self.tools.get(tool_name)
        if tool_func:
            return tool_func(*args, **kwargs)
        else:
            raise ValueError(f"Tool '{tool_name}' not found.")

    def run_tora_loop(self, financial_goal, max_iterations=5):
        print(f"User Goal: {financial_goal}")
        iteration = 0
        
        while iteration < max_iterations:
            print(f"\n--- Iteration {iteration + 1} ---")
            
            context_summary = f"Data fetched: {'Yes' if self.current_context_data is not None else 'No'}. " \
                              f"Strategy defined: {'Yes' if self.current_context_strategy_class is not None else 'No'}. " \
                              f"Sharpe Ratio: {self.current_sharpe_ratio:.2f}."

            reasoning = self._simulate_reasoning(
                f"What is the next step to achieve the goal: {financial_goal}?",
                context=context_summary
            )
            print(reasoning)

            if "fetch data" in reasoning.lower() and self.current_context_data is None:
                ticker = "SPY"
                start_date = "2018-01-01"
                end_date = "2023-01-01"
                interval = "1d"
                print(f"LLM decides to fetch historical data for {ticker}.")
                data = self._simulate_tool_call("get_historical_data", ticker=ticker, start_date=start_date, end_date=end_date, interval=interval)
                self.current_context_data = data
                print(f"Fetched data for {ticker}. Shape: {data.shape}")
                if data.empty:
                    print("Error: No data fetched. Exiting.")
                    return "Failed to fetch data."

            elif "propose a Simple Moving Average strategy" in reasoning.lower() and self.current_context_strategy_class is None:
                strategy_name = "SimpleMovingAverage"
                initial_params = {'sma_period': 20}
                print(f"LLM decides to define {strategy_name} strategy with params: {initial_params}")
                strategy_class = self._simulate_tool_call("define_strategy", strategy_name=strategy_name, params=initial_params)
                self.current_context_strategy_class = strategy_class
                print(f"Strategy '{strategy_name}' defined with initial params {initial_params}.")

            elif "backtest it" in reasoning.lower() and self.current_context_data is not None and self.current_context_strategy_class is not None and self.current_context_backtest_report is None:
                print("LLM decides to run initial backtest.")
                backtest_output = self._simulate_tool_call("run_backtest", strategy_class=self.current_context_strategy_class, data=self.current_context_data)
                
                analysis_report = self._simulate_tool_call("analyze_backtest_results", returns_series=backtest_output['returns'], title="Initial Backtest")
                self.current_context_backtest_report = analysis_report
                self.current_sharpe_ratio = backtest_output['sharpe_ratio']
                print(f"Backtest completed. Initial Sharpe Ratio: {self.current_sharpe_ratio:.2f}")

            elif "optimize its parameters" in reasoning.lower() and self.current_context_data is not None and self.current_context_strategy_class is not None and self.current_sharpe_ratio < 0.8:
                print("LLM decides to optimize strategy parameters.")
                param_ranges = {'sma_period': range(10, 50, 5)}
                optimization_results = self._simulate_tool_call(
                    "optimize_strategy_parameters",
                    strategy_class=self.current_context_strategy_class,
                    param_ranges=param_ranges,
                    data=self.current_context_data,
                    cash=100000.0,
                    commission=0.001,
                    target_metric="sharpe"
                )
                
                best_params = optimization_results['best_params']
                best_sharpe = optimization_results['best_metric_value']
                print(f"Parameter optimization completed. Best params: {best_params}, Best Sharpe: {best_sharpe:.2f}")
                
                self.current_context_strategy_class = self._simulate_tool_call("define_strategy", strategy_name="SimpleMovingAverage", params=best_params)
                
                print("Re-running backtest with optimized parameters.")
                backtest_output_optimized = self._simulate_tool_call("run_backtest", strategy_class=self.current_context_strategy_class, data=self.current_context_data)
                analysis_report_optimized = self._simulate_tool_call("analyze_backtest_results", returns_series=backtest_output_optimized['returns'], title="Optimized Backtest")
                
                self.current_context_backtest_report = analysis_report_optimized
                self.current_sharpe_ratio = backtest_output_optimized['sharpe_ratio']
                
                print("\n--- Final Strategy Recommendation (Optimized) ---")
                print("Based on optimization, the best strategy parameters are:")
                print(best_params)
                print("\nOptimized Performance Report:")
                print(analysis_report_optimized)
                return analysis_report_optimized

            elif "satisfactory" in reasoning.lower() and self.current_sharpe_ratio >= 0.8:
                print("\n--- Final Strategy Recommendation (Satisfactory) ---")
                print("The strategy performance is satisfactory. Here is the final report:")
                print(self.current_context_backtest_report)
                return self.current_context_backtest_report
            
            else:
                print("LLM could not determine a clear next step or deemed the goal unachievable within current logic.")
                break

            iteration += 1
        
        print("\n--- TORA Loop Ended ---")
        return "TORA loop finished without a conclusive strategy or optimization."


if __name__ == "__main__":
    tools = {
        "get_historical_data": get_historical_data,
        "calculate_indicators": calculate_indicators,
        "resample_data": resample_data,
        "define_strategy": define_strategy,
        "run_backtest": run_backtest,
        "analyze_backtest_results": analyze_backtest_results,
        "optimize_strategy_parameters": optimize_strategy_parameters,
    }

    agent = LLMAgent(tools)
    
    financial_goal = "Generate a profitable Simple Moving Average strategy for SPY with optimized parameters aiming for a Sharpe Ratio > 0.8."
    
    final_report = agent.run_tora_loop(financial_goal)
    print("\nOverall Final Result from TORA Agent:")
    print(final_report)