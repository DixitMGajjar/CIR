# CIR Interest Rate Model

Python implementation of the Cox-Ingersoll-Ross model for interest rate simulation and calibration.

## Overview

The CIR model improves on Vasicek by making volatility proportional to the square root of the interest rate level. This prevents negative rates and better matches real-world rate behavior.

Model equation:
```
dr = a(b - r)dt + σ√r dW
```

The key difference from Vasicek is the `√r` term - when rates are high, volatility is high. When rates are low, volatility decreases.

## Installation

```bash
pip install pandas numpy yfinance scipy matplotlib
```

## Quick start

```python
# Download and fit model
data = dwnld_cln_data("^IRX", "2023-01-01", "2024-12-01") 
a, b, sigma = calibrate_cir(data)

# Generate rate paths
r0 = data["actual_IRX"].iloc[-1]
paths = simulate_cir_paths(r0, a, b, sigma, T=1, n_steps=252, n_paths=5000)

# Visualize
plot_paths(paths)
```

## Key differences from Vasicek

- **No negative rates**: CIR naturally keeps rates positive
- **Realistic volatility**: Volatility changes with rate level
- **Feller condition**: Need `2ab ≥ σ²` for guaranteed positive rates

The code automatically checks this condition during calibration and will warn you if it's violated.

## Parameter interpretation

Same as Vasicek but with one caveat - if the Feller condition fails, the model can still work but rates might occasionally hit zero during simulation.

Typical calibrated values:
- `a`: 0.05 - 1.5 (mean reversion)
- `b`: 0.02 - 0.10 (long-term mean)  
- `σ`: 0.1 - 0.5 (volatility coefficient)

## When to use CIR vs Vasicek

Use CIR when:
- You need positive rates only
- Modeling periods with varying volatility
- Working with low rate environments

Use Vasicek when:
- You're okay with negative rates
- Want simpler/faster calculations
- Need analytical bond pricing formulas

## Notes

The simulation uses Euler discretization with a floor at zero for numerical stability. Calibration is done via maximum likelihood estimation.

Works well with daily data over 1-5 year periods. Very long historical periods might span different market regimes and give unstable parameters.
