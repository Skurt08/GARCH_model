import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats
import plotly.io as pio
from scipy.optimize import minimize
from scipy.linalg import cholesky
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

    denom = e0+ea+eb
    alpha = ea / denom
    beta = eb / denom

    omega = np.exp(para[2])
    var = [returns.var()]
    log_likelihood = 0

    for i, ret in enumerate(returns[:-1]):
        vola = omega + alpha * ret ** 2 + beta * var[i]
        log_likelihood += 0.5 * (np.log(vola) + returns[i+1] ** 2 / vola)
        var.append(vola)

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
    denom = 1 + np.exp(optimal_para_hat.x[0]) + np.exp(optimal_para_hat.x[1])
    alpha = np.exp(optimal_para_hat.x[0]) / denom
    beta = np.exp(optimal_para_hat.x[1]) / denom
    omega = np.exp(optimal_para_hat.x[2])

    variance_estimate = omega / (1-alpha-beta)
    conditional_vola_last_period = filter_conditional_volatiliy(alpha, beta, omega, log_returns, True)

    garch_output = {"alpha": alpha,
                    "beta": beta,
                    "omega": omega,
                    "implied unconditional variance": variance_estimate,
                    "sample unconditional variance": log_returns.values.var(),
                    "expected return": return_mean,
                    "return last period": log_returns.values[-1],
                    "conditional vola last period": conditional_vola_last_period[-1]}

    return garch_output

def filter_conditional_volatiliy(alpha: float, beta: float, omega: float, returns: pd.Series, vola_t0_implied = True) -> list[float]:

    if vola_t0_implied:
        conditional_volatilities = [omega / (1-alpha-beta)]

    else: conditional_volatilities = [returns.values.var()]

    for i, ret in enumerate(returns[:-1]):
        conditional_volatilities.append(omega + alpha * ret**2 + beta * conditional_volatilities[i])

    return conditional_volatilities

def forecast_volatility(garch_para: dict[str, float], forecast_period: int) -> tuple[list[float], float]:

    forecast_vola = [garch_para["conditional vola last period"]]

    for i in range(1, forecast_period):
        forecast_vola.append(garch_para["omega"] + (garch_para["alpha"]+garch_para["beta"]) * forecast_vola[i-1])

    return forecast_vola, garch_para["expected return"]

def forecast_stock_volatility(ticker: str, starting_para: Optional[dict[str, float]] = None, interval: str="1d", period: str="5y", price: str="Close", forecast_period: int=7) -> tuple[list[float], float]:

    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)[price]

    if starting_para is None:
        opt_para = optimize_garch_params(data)

    else:
        opt_para = optimize_garch_params(data, starting_para)

    return forecast_volatility(opt_para, forecast_period)

def compute_z_values(returns: pd.Series, mean_return: float, conditional_volatilities: list[float]) -> list[float]:

    z_values = []
    for i, ret in enumerate(returns.to_list()):
        z_values.append((ret - mean_return) / conditional_volatilities[i])

    return z_values

def draw_scenarios(all_z_values: list[list[float]], sample_size = 500):

    l = cholesky(np.cov(all_z_values))
    z = np.random.normal(size=(sample_size, len(all_z_values)))
    return z.dot(l)

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

def garch_1_1_volatility_forecast(assets: list[str], starting_para: Optional[dict[str, float]] = None, interval: str="1d", period: str="5y", price: str="Close", forecast_period: int=7) -> go.Figure:

    assets_with_forecasts = {}
    for asset in assets:
        forecast = forecast_stock_volatility(asset, starting_para, interval, period, price, forecast_period)[0]
        assets_with_forecasts[asset] = forecast

    return plot_vola_forecast(assets_with_forecasts)

list_stocks = ["NVO", "SAP", "SI", "AAPL", "NVDA"]
plot = garch_1_1_volatility_forecast(list_stocks)
plot.show()

starting_para = {"alpha": 0.005,
                 "beta": 0.9,
                 "omega": 0.5}

#aapl_forecast = forecast_stock_volatility(ticker="AAPL")
#print(aapl_forecast[1])

#stock = yf.Ticker("AAPL")
#hist_data = stock.history(interval = "1d", period="5y")["Close"]
#returns = hist_data.pct_change().dropna() * 100
#log_returns = compute_log_returns(hist_data)[0]

#impl_model = optimize_garch_params(hist_data)

#plot = plot_returns(log_returns)
#plot.show()
