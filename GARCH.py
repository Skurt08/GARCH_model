import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats
import plotly.io as pio
from scipy.optimize import minimize
from typing import Optional

pio.renderers.default = "browser"

def compute_log_returns(return_data: pd.Series) -> tuple[pd.Series, float]:
    log_returns = np.log(return_data / return_data.shift(1)).dropna()*100
    return_mean = log_returns.mean()
    log_returns -= return_mean
    return log_returns, return_mean

def plot_returns(returns: pd.Series) -> go.Figure:
    returns = returns.to_numpy()
    x_values = np.linspace(returns.min(), returns.max(), 500)
    kde = stats.gaussian_kde(returns)
    empirical_pdf = kde(x_values)
    mu, sigma = returns.mean(), returns.std()
    normal_pdf = stats.norm.pdf(x_values, mu, sigma)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=empirical_pdf,
        mode="lines",
        name="Empirical PDF (KDE)"
    ))

    fig.add_trace(go.Scatter(
        x=x_values,
        y=normal_pdf,
        mode="lines",
        name="Normal PDF"
    ))
    fig.update_layout(
        title="Distribution Comparison",
        xaxis_title="Value",
        yaxis_title="Density",
    )

    return fig

def compute_conditional_vola(para: list[float], returns: pd.Series) -> float:

    stabilisation_constant = max(0.0, para[0], para[1])
    ea = np.exp(para[0] - stabilisation_constant)
    eb = np.exp(para[1] - stabilisation_constant)
    e0 = np.exp(-stabilisation_constant)

    denominator = e0+ea+eb
    alpha = ea / denominator
    beta = eb / denominator

    omega = np.exp(para[2])
    vola = [omega / (1-alpha-beta)]
    log_likelihood = 0

    for i, ret in enumerate(returns[:-1]):
        vola_i = omega + alpha * ret ** 2 + beta * vola[i]
        log_likelihood += 0.5 * (np.log(vola_i) + returns[i+1] ** 2 / vola_i)
        vola.append(vola_i)

    return log_likelihood

def optimize_garch_params(returns: pd.Series, starting_values: Optional[dict[str, float]] = None) -> dict:

    log_returns, return_mean = compute_log_returns(returns)

    if  starting_values is None or starting_values["alpha"] + starting_values["beta"] >= 1 or min(starting_values.values()) <= 0:
        alpha_start = 0.05
        beta_start = 0.9
        omega_start = 0.05 * log_returns.values.var()
    else:
        alpha_start = starting_values["alpha"]
        beta_start = starting_values["beta"]
        omega_start = starting_values["omega"]

    alpha_start_hat = np.log(alpha_start / (1-alpha_start-beta_start))
    beta_start_hat = np.log(beta_start / (1-alpha_start-beta_start))
    omega_start_hat = np.log(omega_start)

    optimal_para_hat = minimize(compute_conditional_vola, [alpha_start_hat, beta_start_hat, omega_start_hat], args=(log_returns.values,))
    denominator = 1 + np.exp(optimal_para_hat.x[0]) + np.exp(optimal_para_hat.x[1])
    alpha = np.exp(optimal_para_hat.x[0]) / denominator
    beta = np.exp(optimal_para_hat.x[1]) / denominator
    omega = np.exp(optimal_para_hat.x[2])

    past_conditional_vola = filter_conditional_volatility(alpha, beta, omega, log_returns)
    past_z_values = log_returns / past_conditional_vola[:-1]

    garch_output = {"alpha": alpha,
                    "beta": beta,
                    "omega": omega,
                    "implied unconditional variance": omega / (1-alpha-beta),
                    "sample unconditional variance": log_returns.values.var(),
                    "expected return": return_mean,
                    "return last period": log_returns.values[-1],
                    "conditional vola t+1": past_conditional_vola[-1],
                    "past_z_values": past_z_values}

    return garch_output

def filter_conditional_volatility(alpha: float, beta: float, omega: float, returns: pd.Series) -> list[float]:

    conditional_volatility = [omega / (1-alpha-beta)]
    for i, ret in enumerate(returns):
        conditional_volatility.append(omega + alpha * ret**2 + beta * conditional_volatility[i])

    return conditional_volatility

def forecast_scenario_based_returns(scenario_z_values: list[float], garch_output: dict[str, float]) -> np.ndarray:

    conditional_vola = [garch_output["conditional vola t+1"]]
    forecasted_squared_innovations = []
    for i in range(0, len(scenario_z_values)):
        forecasted_squared_innovations.append(scenario_z_values[i] ** 2 * conditional_vola[i])
        conditional_vola.append(garch_output["omega"] + garch_output["alpha"] * forecasted_squared_innovations[i] + garch_output["beta"] * conditional_vola[i])

    return np.sqrt(conditional_vola[:-1]) * np.asarray(scenario_z_values) + garch_output["expected return"]

def get_garch_output(tickers: list[str],
                     starting_para: Optional[dict[str, float]] = None,
                     interval: str="1d",
                     period: str="5y",
                     price: str="Close") -> dict[str, dict[str, float]]:

    assets_with_garch_output = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)[price]

        if starting_para is None:
            garch_output = optimize_garch_params(data)

        else:
            garch_output = optimize_garch_params(data, starting_para)

        assets_with_garch_output[ticker] = garch_output

    return assets_with_garch_output

def plot_vola_forecast(forecasted_volatility: dict[str, list[float]]) -> go.Figure:

    vola_forecast_plot = go.Figure()

    for asset in forecasted_volatility.keys():
        vola_forecast_plot.add_trace(
            go.Scatter(
                x=list(range(1, len(forecasted_volatility[asset])+1)),
                y=np.sqrt(forecasted_volatility[asset]),
                mode="lines+markers",
                name=asset + " volatility forecast"
            )
        )
    vola_forecast_plot.update_layout(
        title="volatility forecasts (sigma)",
        xaxis_title="period",
        yaxis_title="volatility (sigma)"
    )

    return vola_forecast_plot

def plot_scenario_returns(scenario_returns: list[list[float]]) -> go.Figure:
    scenario_returns_plot = go.Figure()

    for i, scenario in enumerate(scenario_returns):
        total_return = round(sum(scenario), 2)
        scenario_returns_plot.add_trace(
            go.Scatter(
                x=list(range(1, len(scenario)+1)),
                y=scenario,
                mode="lines+markers",
                name="scenario " + str(i+1) + " with total return of " + str(total_return) + "%"
            )
        )
    scenario_returns_plot.update_layout(
        title="scenario returns forecasts",
        xaxis_title="period",
        yaxis_title="returns"
    )

    return scenario_returns_plot

def garch_1_1_volatility_forecast(garch_output: dict[str, dict[str, float]], forecast_period: int=5) -> go.Figure:

    assets_with_vola_forecasts = {}
    for asset, parameter in garch_output.items():
        vola_forecast = [parameter["conditional vola t+1"]]
        for i in range(1, forecast_period):
            vola_forecast.append(parameter["omega"] + (parameter["alpha"] + parameter["beta"]) * vola_forecast[i - 1])
        assets_with_vola_forecasts[asset] = vola_forecast

    return plot_vola_forecast(assets_with_vola_forecasts)

def deterministic_portfolio_return_forecasts(garch_output: dict[str, dict[str, float]],
                                             scenarios: list[list[float]],
                                             asset_weights: dict[str, float],
                                             main_asset: str) -> go.Figure:

    z_values_main = garch_output[main_asset]["past_z_values"]
    returns_different_scenarios = []
    for scenario in scenarios:
        scenario = np.asarray(scenario)
        scenario_returns = np.zeros(len(scenario))
        for asset, parameter in garch_output.items():
            corr = parameter["past_z_values"].corr(z_values_main)
            z_values = corr * scenario
            scenario_returns += asset_weights[asset] * forecast_scenario_based_returns(z_values, parameter)
        returns_different_scenarios.append(scenario_returns.tolist())

    return plot_scenario_returns(returns_different_scenarios)
