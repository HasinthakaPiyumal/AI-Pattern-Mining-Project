import yfinance as yf
import pandas as pd
import numpy as np

def optimize_portfolio(tickers, start_date, end_date, num_portfolios=10000):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    log_returns = np.log(data / data.shift(1))

    annual_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    portfolio_returns = []
    portfolio_volatility = []
    sharpe_ratios = []
    portfolio_weights = []

    num_assets = len(tickers)

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_weights.append(weights)

        p_return = np.sum(weights * annual_returns) * 1
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 1
        sharpe_ratio = p_return / p_std 

        portfolio_returns.append(p_return)
        portfolio_volatility.append(p_std)
        sharpe_ratios.append(sharpe_ratio)

    portfolios = pd.DataFrame({
        'Return': portfolio_returns,
        'Volatility': portfolio_volatility,
        'Sharpe Ratio': sharpe_ratios
    })
    for i, ticker in enumerate(tickers):
        portfolios[f'{ticker} Weight'] = [w[i] for w in portfolio_weights]

    max_sharpe_portfolio = portfolios.loc[portfolios['Sharpe Ratio'].idxmax()]
    min_volatility_portfolio = portfolios.loc[portfolios['Volatility'].idxmin()]

    results = {
        "max_sharpe_portfolio": {
            "return": max_sharpe_portfolio['Return'],
            "volatility": max_sharpe_portfolio['Volatility'],
            "sharpe_ratio": max_sharpe_portfolio['Sharpe Ratio'],
            "weights": {ticker: max_sharpe_portfolio[f'{ticker} Weight'] for ticker in tickers}
        },
        "min_volatility_portfolio": {
            "return": min_volatility_portfolio['Return'],
            "volatility": min_volatility_portfolio['Volatility'],
            "sharpe_ratio": min_volatility_portfolio['Sharpe Ratio'],
            "weights": {ticker: min_volatility_portfolio[f'{ticker} Weight'] for ticker in tickers}
        }
    }
    return results

# Example of how the LLM would generate and execute this code:
# tickers = ["AAPL", "MSFT", "GOOGL"]
# start_date = "2020-01-01"
# end_date = "2023-12-31"
# optimization_results = optimize_portfolio(tickers, start_date, end_date)
# print(optimization_results)
