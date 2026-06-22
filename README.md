# GARCH(1,1) Volatility & Portfolio Return Forecasting

A Python implementation of a GARCH(1,1) model for volatility estimation and scenario-based portfolio return forecasting. The model is fit via maximum likelihood estimation and supports multi-asset portfolios with correlation-adjusted scenario generation.

---

## Features

- **GARCH(1,1) parameter estimation** via MLE with a numerically stable softmax reparameterisation
- **Multi-step volatility forecasting** using the analytical GARCH recursion
- **Scenario-based return forecasting** with correlation-adjusted z-values across assets
- **Portfolio-level return projections** with custom asset weights
- **Interactive Plotly charts** for volatility forecasts, scenario returns, and return distributions
- Validated against the [`arch`](https://arch.readthedocs.io/) package (see notebook)

---

## Project Structure

```
├── gitignore
├── requirements.txt
├── GARCH.py                          # Core model implementation
└── Example_Return_Forecasting.ipynb  # Usage walkthrough & arch package comparison
```

---

## Installation

```bash
pip install yfinance pandas numpy scipy plotly
```

To run the example notebook you will also need:

```bash
pip install arch jupyter
```

---

## Quick Start

```python
from GARCH import get_garch_output, garch_1_1_volatility_forecast, deterministic_portfolio_return_forecasts

# 1. Fit GARCH(1,1) to a list of tickers
assets = ["NVO", "AAPL", "SAP", "SI", "NVDA"]
output = get_garch_output(tickers=assets, interval="1d", period="5y", price="Close")

# 2. Plot a 5-period volatility forecast
fig = garch_1_1_volatility_forecast(output, forecast_period=5)
fig.show()

# 3. Scenario-based portfolio return forecasts
scenarios = [
    [0.3, -0.2, -0.1,  0.2],   # scenario 1
    [-0.2, -0.1, 0.2,  0.3],   # scenario 2
    [0.3,  0.2, -0.1, -0.2],   # scenario 3
]
weights = {"NVO": 0.3, "AAPL": 0.2, "SAP": 0.1, "SI": 0.2, "NVDA": 0.1}

fig = deterministic_portfolio_return_forecasts(
    garch_output=output,
    scenarios=scenarios,
    main_asset="NVO",
    asset_weights=weights
)
fig.show()
```

---

## API Reference

### `get_garch_output`

Fetches price data via `yfinance` and fits a GARCH(1,1) model to each ticker.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tickers` | `list[str]` | — | Yahoo Finance ticker symbols |
| `starting_para` | `dict` | `None` | Optional starting values for `alpha`, `beta`, `omega` |
| `interval` | `str` | `"1d"` | Data frequency (e.g. `"1d"`, `"1wk"`) |
| `period` | `str` | `"5y"` | Lookback period (e.g. `"5y"`, `"2y"`) |
| `price` | `str` | `"Close"` | Price column to use |

**Returns** a dict keyed by ticker, each containing:

| Key | Description |
|---|---|
| `alpha` | ARCH coefficient |
| `beta` | GARCH coefficient |
| `omega` | Constant term |
| `implied unconditional variance` | `omega / (1 - alpha - beta)` |
| `sample unconditional variance` | Empirical variance of log returns |
| `expected return` | Mean log return (subtracted before estimation) |
| `return last period` | Most recent demeaned log return |
| `conditional vola t+1` | One-step-ahead conditional variance forecast |
| `past_z_values` | Standardised residuals (used for correlation-based scenario generation) |

---

### `garch_1_1_volatility_forecast`

Produces an analytical multi-step volatility forecast and returns an interactive Plotly figure.

```python
garch_1_1_volatility_forecast(garch_output, forecast_period=5)
```

---

### `deterministic_portfolio_return_forecasts`

Generates scenario-based portfolio return paths. Scenarios are defined as sequences of z-values for a chosen main asset; all other assets receive correlation-adjusted z-values derived from their historical co-movement with the main asset.

```python
deterministic_portfolio_return_forecasts(
    garch_output=output,
    scenarios=scenarios,      # list of lists of z-values
    main_asset="NVO",         # ticker for which scenarios are specified
    asset_weights=weights     # dict of {ticker: weight}
)
```

---

## Model Details

The GARCH(1,1) variance process is:

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

**Estimation** uses a softmax reparameterisation for $\alpha$ and $\beta$ to enforce stationarity ($\alpha + \beta < 1$) and positivity during unconstrained numerical optimisation:

$$\alpha = \frac{e^{\hat\alpha}}{1 + e^{\hat\alpha} + e^{\hat\beta}}, \quad \beta = \frac{e^{\hat\beta}}{1 + e^{\hat\alpha} + e^{\hat\beta}}$$

$\omega$ is log-transformed to ensure positivity. The objective minimised is the Gaussian quasi-log-likelihood.

---

## Example Output (NVO, 5y daily)

```
alpha  : 0.137
beta   : 0.798
omega  : 0.508
Implied unconditional vol : 2.805 %
```

Results are consistent with the `arch` package reference implementation (see notebook).

---

## License

MIT
