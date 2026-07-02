import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gammaln
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Optional

def compute_log_returns(return_data: pd.Series) -> tuple[pd.Series, float]:
    log_returns = np.log(return_data / return_data.shift(1)).dropna()*100
    return_mean = log_returns.mean()
    log_returns -= return_mean
    return log_returns, return_mean

def garch_negative_log_likelihood(para: list[float], returns: pd.Series) -> tuple[float, list[float]]:

    stabilisation_constant = max(0.0, para[0], para[1])
    ea = np.exp(para[0] - stabilisation_constant)
    eb = np.exp(para[1] - stabilisation_constant)
    e0 = np.exp(-stabilisation_constant)

    denominator = e0+ea+eb
    alpha = ea / denominator
    beta = eb / denominator

    omega = np.exp(para[2])
    nu = 2.0 + np.exp(para[3])
    log_const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    vola = [omega / (1-alpha-beta)]
    '''
    log_likelihood = (-log_const
                      + 0.5 * np.log(vola[0])
                      + ((nu + 1) / 2) * np.log(1 + returns[0]**2 / (vola[-1] * (nu - 2))))
    '''
    log_likelihood = 0

    for i, ret in enumerate(returns[:-1]):
        vola.append(omega + alpha * ret ** 2 + beta * vola[-1])
        log_likelihood += (-log_const
                           + 0.5 * np.log(vola[-1])
                           + ((nu + 1) / 2) * np.log(1 + returns[i + 1]**2 / (vola[-1] * (nu - 2))))

    vola.append(omega + alpha * returns[-1] ** 2 + beta * vola[-1])

    return log_likelihood, vola

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
    nu_start_hat = np.log(6.0)
    x0 = [alpha_start_hat, beta_start_hat, omega_start_hat, nu_start_hat]
    conditional_vola = []

    def objective(para: list[float]) -> float:
        nonlocal conditional_vola
        likelihood, conditional_vola = garch_negative_log_likelihood(para, log_returns.values)
        return likelihood


    optimal_para_hat = minimize(objective, x0)
    denominator = 1 + np.exp(optimal_para_hat.x[0]) + np.exp(optimal_para_hat.x[1])
    alpha = np.exp(optimal_para_hat.x[0]) / denominator
    beta = np.exp(optimal_para_hat.x[1]) / denominator
    omega = np.exp(optimal_para_hat.x[2])
    nu = 2.0 + np.exp(optimal_para_hat.x[3])

    past_z_values = log_returns[1:] / np.sqrt(conditional_vola[1:-1])

    garch_output = {"alpha": alpha,
                    "beta": beta,
                    "omega": omega,
                    "nu": nu,
                    "implied unconditional variance": omega / (1-alpha-beta),
                    "sample unconditional variance": log_returns.values.var(),
                    "expected return": return_mean,
                    "return last period": log_returns.values[-1],
                    "conditional vola t+1": conditional_vola[-1],
                    "past_z_values": past_z_values}

    return garch_output

def plot_returns_t(returns: pd.Series, nu: float) -> go.Figure:
    returns = returns.to_numpy()

    x_values = np.linspace(returns.min(), returns.max(), 500)

    kde = stats.gaussian_kde(returns)
    empirical_pdf = kde(x_values)
    t_dist = stats.t(df=nu)
    scale = np.sqrt(nu / (nu - 2)) if nu > 2 else np.nan

    t_pdf = t_dist.pdf(x_values / scale) / scale

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=empirical_pdf,
        mode="lines",
        name="Empirical KDE"
    ))

    fig.add_trace(go.Scatter(
        x=x_values,
        y=t_pdf,
        mode="lines",
        name=f"Student-t PDF (ν={nu:.2f})"
    ))

    fig.update_layout(
        title="Standardized Residual Distribution: KDE vs Student-t",
        xaxis_title="Value",
        yaxis_title="Density"
    )

    return fig


def diagnose_z_values(past_z_values: pd.Series, estimated_nu: float):

    plot_returns_t(past_z_values, estimated_nu).show()

    print(f'mean z values: {round(past_z_values.mean(), 4)}')
    print(f'std z values: {round(past_z_values.std(), 4)}')
    if estimated_nu > 4:
        print(f'empirical excess kurtosis z values: {past_z_values.kurtosis()}')
        print(f'theoretical excess kurtosis z values: {6 / (estimated_nu - 4)}')
    else: print('estimated degrees of freedom smaller or equal to 4, infinite kurtosis')

    lb = acorr_ljungbox(past_z_values, lags=[10, 20], return_df=True)
    print('Ljung-Box-Test for z values')
    print(lb)

    lb_sq = acorr_ljungbox(past_z_values ** 2, lags=[10, 20], return_df=True)
    print('Ljung-Box-Test for z^2 values')
    print(lb_sq)

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
