"""Modern Portfolio Theory helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - shown in Streamlit UI.
    minimize = None

TRADING_DAYS = 252


def calculate_returns(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Convert aligned close prices into daily percentage returns."""

    returns = close_prices.ffill().dropna(how="all").pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Not enough overlapping price data for portfolio returns.")
    return returns


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    """Return annualised return, volatility, and Sharpe ratio."""

    annual_return = float(np.dot(weights, mean_returns) * TRADING_DAYS)
    annual_volatility = float(np.sqrt(weights.T @ (covariance * TRADING_DAYS) @ weights))
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility else 0.0
    return annual_return, annual_volatility, sharpe


def optimise_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    frontier_points: int = 40,
) -> dict[str, object]:
    """Find max-Sharpe allocation and efficient frontier."""

    if minimize is None:
        raise ImportError("scipy is required for portfolio optimisation. Run: pip install scipy")

    if returns.shape[1] < 2:
        raise ValueError("Select at least two assets for portfolio optimisation.")

    mean_returns = returns.mean()
    covariance = returns.cov()
    n_assets = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)
    initial_guess = np.repeat(1 / n_assets, n_assets)

    def negative_sharpe(weights: np.ndarray) -> float:
        return -portfolio_performance(weights, mean_returns, covariance, risk_free_rate)[2]

    max_sharpe_result = minimize(
        negative_sharpe,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not max_sharpe_result.success:
        raise ValueError(f"Optimisation failed: {max_sharpe_result.message}")

    optimal_weights = max_sharpe_result.x
    optimal_return, optimal_volatility, optimal_sharpe = portfolio_performance(
        optimal_weights, mean_returns, covariance, risk_free_rate
    )

    equal_weights = initial_guess
    equal_return, equal_volatility, equal_sharpe = portfolio_performance(
        equal_weights, mean_returns, covariance, risk_free_rate
    )

    min_return = float(mean_returns.min() * TRADING_DAYS)
    max_return = float(mean_returns.max() * TRADING_DAYS)
    target_returns = np.linspace(min_return, max_return, frontier_points)

    frontier_rows = []
    for target_return in target_returns:
        frontier_constraints = (
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
            {
                "type": "eq",
                "fun": lambda weights, target=target_return: (
                    np.dot(weights, mean_returns) * TRADING_DAYS
                )
                - target,
            },
        )

        result = minimize(
            lambda weights: portfolio_performance(weights, mean_returns, covariance, risk_free_rate)[1],
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=frontier_constraints,
        )
        if result.success:
            annual_return, annual_volatility, sharpe = portfolio_performance(
                result.x, mean_returns, covariance, risk_free_rate
            )
            frontier_rows.append(
                {
                    "Return": annual_return,
                    "Volatility": annual_volatility,
                    "Sharpe": sharpe,
                }
            )

    weights = pd.DataFrame(
        {
            "Asset": returns.columns,
            "Optimal Weight": optimal_weights,
            "Equal Weight": equal_weights,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "Portfolio": "Max Sharpe",
                "Annual Return": optimal_return,
                "Annual Volatility": optimal_volatility,
                "Sharpe Ratio": optimal_sharpe,
            },
            {
                "Portfolio": "Equal Weight",
                "Annual Return": equal_return,
                "Annual Volatility": equal_volatility,
                "Sharpe Ratio": equal_sharpe,
            },
        ]
    )

    return {
        "weights": weights,
        "summary": summary,
        "frontier": pd.DataFrame(frontier_rows),
    }
